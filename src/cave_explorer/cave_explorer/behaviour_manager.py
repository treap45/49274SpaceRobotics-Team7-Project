"""Exploration/inspection state machine and inspection records (Planning 3).

Explore; when a confirmed artefact of an inspection class appears, approach it;
retry once if it is lost or unreachable, then give up; on success record it as
visited; resume exploring. Plain Python: the node calls ``step`` from its timer,
so the state machine is unit-tested with fakes.

Rules:

* ``VISITED`` needs the planner's ``evaluate`` to return ``SATISFIED``; a Nav2
  success alone is not an inspection.
* A Nav2 success farther than ``arrival_tolerance_m`` from the goal is not an
  arrival (Humble can report success for a preempted goal). Exploration treats
  it as ``CANCELED``; an inspection viewpoint is sent again.
* ``LOST`` only counts after the robot has reached the attempt's viewpoint. A
  retry drives to a new viewpoint, excluding every one tried.
* No progress (less than ``stuck_distance_m`` in ``stuck_timeout_sec``) gives the
  goal up: the goal is cancelled and ``on_stuck`` lets the node back away.
* A Nav2 rejection moves to another viewpoint without using up an attempt.
* An attempt has ``attempt_timeout_sec`` plus the travel time to its first
  viewpoint at ``approach_speed_mps``.
* ``DEFERRED`` is final, except that an artefact deferred for lack of a safe
  viewpoint gets one more round once exploration is exhausted and the map is
  complete. Other deferrals (mostly false detections) are not revisited.
* When the tracker merges two tracks, ``merge_records`` moves the retired id's
  record to the surviving id, so one artefact is never inspected twice.
* Goals are switched by sending the new goal, never cancel-then-send.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Callable, Protocol, Sequence

from geometry_msgs.msg import PoseStamped

from cave_explorer.interfaces import (
    ArtifactTrack,
    ExplorationGoal,
    ExplorationGoalStatus,
    InspectionAssessment,
    InspectionEvaluation,
    InspectionGoal,
    InspectionGoalStatus,
    InspectionRecord,
    InspectionState,
    NavigationProgress,
    NavigationStatus,
)


class Mode(Enum):
    EXPLORING = 'EXPLORING'
    INSPECTING = 'INSPECTING'
    EXHAUSTED = 'EXHAUSTED'       # no usable frontier; retrying periodically


# ----------------------------------------------------------------------
# What the manager needs from its collaborators
# ----------------------------------------------------------------------

class Navigator(Protocol):
    def send(self, goal: PoseStamped, timeout_sec: float | None = None) -> int: ...
    def poll(self, request_id: int) -> NavigationProgress: ...
    def cancel(self, request_id: int) -> None: ...


class Explorer(Protocol):
    frontiers: tuple

    def select_goal(self, map_msg, robot_xy, now_ns: int) -> ExplorationGoal: ...
    def report_outcome(self, goal, status, reason, now_ns: int) -> None: ...


class InspectionPlanner(Protocol):
    def make_goal(self, track: ArtifactTrack, robot_pose, map_msg, *,
                  exclude: tuple = ()) -> InspectionGoal: ...
    def evaluate(self, track: ArtifactTrack, robot_pose,
                 now_ns: int) -> InspectionAssessment: ...


@dataclass(frozen=True)
class BehaviourConfig:
    """State machine tuning; times in seconds of ROS time.

    ``inspect_classes`` empty means explore only.
    """

    inspect_classes: frozenset = frozenset()
    max_attempts: int = 2               # the first try plus one retry
    max_viewpoints: int = 3             # per attempt, after Nav2 rejections
    attempt_timeout_sec: float = 60.0   # per attempt, plus travel time
    approach_speed_mps: float = 0.3     # travel time = distance / this
    dwell_timeout_sec: float = 10.0     # arrived, waiting for SATISFIED
    explore_goal_timeout_sec: float = 120.0
    replan_period_sec: float = 4.0      # SLAM republishes the map every 2 s
    switch_distance_m: float = 2.0      # current goal counts as still valid within this
    exhausted_retry_sec: float = 5.0
    arrival_tolerance_m: float = 2.0    # Nav2 xy_goal_tolerance is 1.0 m
    stuck_timeout_sec: float = 20.0     # longer than Nav2's spin/wait recoveries
    stuck_distance_m: float = 0.5


@dataclass
class _Exploration:
    request_id: int
    goal: PoseStamped


@dataclass
class _Progress:
    """Where the robot last moved ``stuck_distance_m`` from, and when."""
    xy: tuple
    since_ns: int


@dataclass
class _Inspection:
    artifact_id: int
    started_ns: int
    tried: list = field(default_factory=list)       # viewpoints of this attempt
    excluded: list = field(default_factory=list)    # viewpoints of earlier attempts
    request_id: int | None = None
    arrived_ns: int | None = None
    deadline_ns: int | None = None                  # set with the attempt's first viewpoint


class BehaviourManager:
    """Decides, every main-loop tick, whether to explore or inspect and what to send."""

    def __init__(self, navigator: Navigator, explorer: Explorer,
                 planner: InspectionPlanner | None,
                 config: BehaviourConfig = BehaviourConfig(),
                 log: Callable[[str], None] | None = None,
                 on_stuck: Callable[[object], float] | None = None):
        """With ``planner`` ``None`` the robot only explores."""
        self._navigator = navigator
        self._explorer = explorer
        self._planner = planner
        self.config = config
        self._log = log or (lambda message: None)
        self._on_stuck = on_stuck

        self.mode = Mode.EXPLORING
        self.status = 'starting'
        self._records: dict[int, InspectionRecord] = {}
        self._exploration: _Exploration | None = None
        self._inspection: _Inspection | None = None
        self._last_plan_ns: int | None = None
        self._progress: _Progress | None = None
        self._paused_until_ns: int | None = None
        # Deferred for want of a safe viewpoint: artifact_id -> viewpoints tried.
        self._awaiting_revisit: dict[int, list] = {}
        # Revisit started: artifact_id -> viewpoints its first attempt excludes.
        self._revisit_excluded: dict[int, list] = {}
        self._revisited: set[int] = set()

    @property
    def records(self) -> tuple[InspectionRecord, ...]:
        """Snapshot of every inspection record, for the markers."""
        return tuple(self._records[key] for key in sorted(self._records))

    def merge_records(self, retired_id: int, kept_id: int) -> None:
        """The tracker merged ``retired_id`` into ``kept_id``: carry its record over."""
        retired = self._records.pop(retired_id, None)
        self._awaiting_revisit.pop(retired_id, None)
        self._revisit_excluded.pop(retired_id, None)
        if retired is None:
            return      # not a class we inspect
        kept = self._records.get(kept_id)
        kept_visited = kept is not None and kept.state is InspectionState.VISITED
        inspection = self._inspection
        if inspection is not None and inspection.artifact_id == retired_id:
            if kept_visited:
                self._log(f'artefact {retired_id} is artefact {kept_id}, already inspected')
                self._finish_inspection()
            else:
                inspection.artifact_id = kept_id
                self._set_record(kept_id, retired.state, retired.attempts,
                                 retired.last_failure_reason)
                self._log(f'artefact {retired_id} is artefact {kept_id}; inspecting it as {kept_id}')
            return
        if retired.state is InspectionState.VISITED and not kept_visited:
            if inspection is not None and inspection.artifact_id == kept_id:
                self._finish_inspection()
            self._set_record(kept_id, InspectionState.VISITED, retired.attempts)
            self._log(f'artefact {retired_id} is artefact {kept_id}, already inspected')

    def step(self, now_ns: int, robot_pose, map_msg,
             tracks: Sequence[ArtifactTrack] = ()) -> None:
        """Advance one tick. ``robot_pose`` needs ``x`` and ``y`` attributes, in ``map``."""
        targets = {track.artifact_id: track for track in tracks
                   if track.confirmed and track.class_name in self.config.inspect_classes}
        for artifact_id, track in targets.items():
            if artifact_id not in self._records:
                self._records[artifact_id] = InspectionRecord(
                    artifact_id, InspectionState.UNVISITED, 0)
                self._log(f'artefact {artifact_id} ({track.class_name}) added to inspect')

        if self._paused_until_ns is not None and now_ns < self._paused_until_ns:
            return      # recovering from being stuck: send nothing
        self._paused_until_ns = None

        if self.mode is Mode.INSPECTING:
            self._inspect(now_ns, robot_pose, map_msg, targets)
        # A target can fail at once (no safe viewpoint): try the next one in
        # the same tick. Bounded so a bug cannot hang the timer.
        for _ in range(len(targets)):
            if self.mode is Mode.INSPECTING or self._planner is None:
                break
            target = self._nearest_unvisited(robot_pose, targets)
            if target is None:
                break
            self._begin_inspection(target, now_ns)
            self._inspect(now_ns, robot_pose, map_msg, targets)
        if self.mode is not Mode.INSPECTING:
            self._explore(now_ns, robot_pose, map_msg, targets)

    # ------------------------------------------------------------------
    # Exploring
    # ------------------------------------------------------------------

    def _explore(self, now_ns, robot_pose, map_msg, targets) -> None:
        current = self._exploration
        if current is not None:
            progress = self._navigator.poll(current.request_id)
            if progress.is_terminal:
                status, reason = progress.status, progress.reason
                if progress.succeeded and not self._arrived(robot_pose, current.goal):
                    status, reason = NavigationStatus.CANCELED, 'success reported away from the goal'
                    self._log(f'exploration goal {_xy(current.goal)}: {reason}; ignoring it')
            elif self._stuck(now_ns, robot_pose):
                status = NavigationStatus.ABORTED
                reason = f'no progress for {self.config.stuck_timeout_sec:g} s'
                self._log(f'exploration goal {_xy(current.goal)}: {reason}; giving it up')
                self._recover(now_ns, robot_pose, current.request_id)
            else:
                if self._due(now_ns, self.config.replan_period_sec):
                    self._replan_while_driving(now_ns, robot_pose, map_msg, current)
                return
            self._explorer.report_outcome(current.goal, status, reason, now_ns)
            self._exploration = None
            if self._paused_until_ns is not None:
                return

        if self.mode is Mode.EXHAUSTED and not self._due(now_ns,
                                                         self.config.exhausted_retry_sec):
            return
        result = self._plan(now_ns, robot_pose, map_msg)
        if result.status is ExplorationGoalStatus.READY:
            self._send_exploration(result)
        elif result.status is ExplorationGoalStatus.NOT_READY:
            self.mode = Mode.EXPLORING
            self.status = f'waiting: {result.reason}'
        elif self._revisit_deferred(targets):
            pass        # the next tick starts the inspection
        elif self.mode is not Mode.EXHAUSTED:
            self.mode = Mode.EXHAUSTED
            self.status = f'exploration paused: {result.reason}'
            self._log(self.status)

    def _replan_while_driving(self, now_ns, robot_pose, map_msg, current) -> None:
        """Switch only if the frontier being driven to has been mapped away."""
        result = self._plan(now_ns, robot_pose, map_msg)
        if result.status is not ExplorationGoalStatus.READY:
            return
        target = _xy(current.goal)
        still_there = any(frontier.goal is not None and not frontier.rejection
                          and math.dist(frontier.goal, target) <= self.config.switch_distance_m
                          for frontier in self._explorer.frontiers)
        if not still_there:
            self._log('frontier being explored has been mapped; switching goal')
            self._send_exploration(result, keep_progress=True)

    def _send_exploration(self, result: ExplorationGoal, keep_progress: bool = False) -> None:
        request_id = self._send(result.goal, self.config.explore_goal_timeout_sec,
                                keep_progress)
        self._exploration = _Exploration(request_id, result.goal)
        self.mode = Mode.EXPLORING
        self.status = f'exploring: {result.reason}'

    def _send(self, goal: PoseStamped, timeout_sec: float, keep_progress: bool = False) -> int:
        """Send a goal; a switch while driving keeps the stuck timer running."""
        if not keep_progress:
            self._progress = None
        return self._navigator.send(goal, timeout_sec=timeout_sec)

    def _recover(self, now_ns, robot_pose, request_id) -> None:
        """Stop driving a goal the robot is stuck on and let the node back it off."""
        self._navigator.cancel(request_id)
        self._progress = None
        if self._on_stuck is not None:
            pause_sec = self._on_stuck(robot_pose)
            if pause_sec > 0.0:
                self._paused_until_ns = now_ns + int(pause_sec * 1e9)

    def _stuck(self, now_ns, robot_pose) -> bool:
        """True once the robot has moved less than ``stuck_distance_m`` for ``stuck_timeout_sec``."""
        here = (robot_pose.x, robot_pose.y)
        progress = self._progress
        if progress is None or math.dist(here, progress.xy) > self.config.stuck_distance_m:
            self._progress = _Progress(here, now_ns)
            return False
        return now_ns - progress.since_ns > self.config.stuck_timeout_sec * 1e9

    def _arrived(self, robot_pose, goal: PoseStamped) -> bool:
        return math.dist((robot_pose.x, robot_pose.y), _xy(goal)) <= self.config.arrival_tolerance_m

    def _plan(self, now_ns, robot_pose, map_msg) -> ExplorationGoal:
        self._last_plan_ns = now_ns
        return self._explorer.select_goal(map_msg, (robot_pose.x, robot_pose.y), now_ns)

    def _due(self, now_ns, period_sec) -> bool:
        return (self._last_plan_ns is None
                or now_ns - self._last_plan_ns >= period_sec * 1e9)

    # ------------------------------------------------------------------
    # Inspecting
    # ------------------------------------------------------------------

    def _nearest_unvisited(self, robot_pose, targets) -> ArtifactTrack | None:
        waiting = [track for artifact_id, track in targets.items()
                   if self._records[artifact_id].state is InspectionState.UNVISITED]
        if not waiting:
            return None
        here = (robot_pose.x, robot_pose.y)
        return min(waiting, key=lambda track: math.dist(here, track.position_map[:2]))

    def _revisit_deferred(self, targets) -> bool:
        """Put no-safe-viewpoint deferrals back for one more round. True if any."""
        ids = [artifact_id for artifact_id in self._awaiting_revisit if artifact_id in targets]
        for artifact_id in ids:
            self._revisit_excluded[artifact_id] = self._awaiting_revisit.pop(artifact_id)
            self._revisited.add(artifact_id)
            self._set_record(artifact_id, InspectionState.UNVISITED, 0,
                             self._records[artifact_id].last_failure_reason)
            self._log(f'artefact {artifact_id}: exploration exhausted; looking again for a '
                      'safe viewpoint on the final map')
        return bool(ids)

    def _begin_inspection(self, track: ArtifactTrack, now_ns: int) -> None:
        # Sending the viewpoint preempts any exploration goal.
        self._exploration = None
        self._inspection = _Inspection(
            track.artifact_id, now_ns,
            excluded=self._revisit_excluded.pop(track.artifact_id, []))
        self._set_record(track.artifact_id, InspectionState.IN_PROGRESS, 1)
        self.mode = Mode.INSPECTING
        self.status = f'inspecting artefact {track.artifact_id} ({track.class_name}), attempt 1'
        self._log(self.status)

    def _inspect(self, now_ns, robot_pose, map_msg, targets) -> None:
        inspection = self._inspection
        track = targets.get(inspection.artifact_id)
        if track is None:
            self._attempt_failed(now_ns, 'artefact no longer reported by the tracker')
            return
        deadline_ns = inspection.deadline_ns or \
            inspection.started_ns + int(self.config.attempt_timeout_sec * 1e9)
        if now_ns > deadline_ns:
            self._attempt_failed(
                now_ns, 'not inspected within '
                f'{(deadline_ns - inspection.started_ns) / 1e9:.0f} s')
            return

        assessment = self._planner.evaluate(track, robot_pose, now_ns)
        if assessment.status is InspectionEvaluation.SATISFIED:
            self._inspection_succeeded(assessment.reason)
            return
        if (assessment.status is InspectionEvaluation.LOST
                and inspection.arrived_ns is not None):
            self._attempt_failed(now_ns, f'artefact lost: {assessment.reason}')
            return

        if inspection.request_id is None:
            self._plan_viewpoint(now_ns, robot_pose, map_msg, track)
            return
        progress = self._navigator.poll(inspection.request_id)
        if not progress.is_terminal:
            if self._stuck(now_ns, robot_pose):
                self._log(f'viewpoint for artefact {inspection.artifact_id}: no progress for '
                          f'{self.config.stuck_timeout_sec:g} s; trying another')
                self._recover(now_ns, robot_pose, inspection.request_id)
                inspection.request_id = None
            return
        if progress.succeeded and not self._arrived(robot_pose, inspection.tried[-1]):
            self._log(f'viewpoint for artefact {inspection.artifact_id}: success reported '
                      'away from the goal; sending it again')
            inspection.request_id = self._send(inspection.tried[-1],
                                               self.config.attempt_timeout_sec)
            return
        if progress.succeeded:
            if inspection.arrived_ns is None:
                inspection.arrived_ns = now_ns
            elif now_ns - inspection.arrived_ns > self.config.dwell_timeout_sec * 1e9:
                self._attempt_failed(now_ns, 'reached the viewpoint but inspection was '
                                     f'never satisfied ({assessment.reason})')
            return
        # The viewpoint failed, not the attempt: try another one.
        self._log(f'viewpoint for artefact {inspection.artifact_id} failed '
                  f'({progress.status.value}: {progress.reason}); trying another')
        inspection.request_id = None

    def _plan_viewpoint(self, now_ns, robot_pose, map_msg, track) -> None:
        inspection = self._inspection
        if len(inspection.tried) >= self.config.max_viewpoints:
            self._attempt_failed(
                now_ns, f'no reachable viewpoint after {len(inspection.tried)} tries')
            return
        plan = self._planner.make_goal(track, robot_pose, map_msg,
                                       exclude=tuple(inspection.excluded + inspection.tried))
        if plan.status is InspectionGoalStatus.NOT_READY:
            return     # bounded by the attempt timeout
        if plan.status is InspectionGoalStatus.NO_SAFE_GOAL:
            self._attempt_failed(now_ns, f'no safe viewpoint: {plan.reason}')
            return
        inspection.tried.append(plan.goal)
        if inspection.deadline_ns is None:
            travel_sec = math.dist((robot_pose.x, robot_pose.y), _xy(plan.goal)) \
                / self.config.approach_speed_mps
            inspection.deadline_ns = inspection.started_ns + int(
                (self.config.attempt_timeout_sec + travel_sec) * 1e9)
        inspection.request_id = self._send(
            plan.goal, (inspection.deadline_ns - now_ns) / 1e9)

    def _attempt_failed(self, now_ns, reason: str) -> None:
        artifact_id = self._inspection.artifact_id
        attempts = self._records[artifact_id].attempts
        if attempts < self.config.max_attempts:
            self._log(f'artefact {artifact_id} attempt {attempts} failed ({reason}); retrying')
            previous = self._inspection
            self._inspection = _Inspection(artifact_id, now_ns,
                                           excluded=previous.excluded + previous.tried)
            self._set_record(artifact_id, InspectionState.IN_PROGRESS, attempts + 1, reason)
            self.status = f'inspecting artefact {artifact_id}, attempt {attempts + 1}'
            return
        self._set_record(artifact_id, InspectionState.DEFERRED, attempts, reason)
        self._log(f'artefact {artifact_id} deferred after {attempts} attempts ({reason})')
        if reason.startswith('no safe viewpoint') and artifact_id not in self._revisited:
            previous = self._inspection
            self._awaiting_revisit[artifact_id] = previous.excluded + previous.tried
        self._finish_inspection()

    def _inspection_succeeded(self, reason: str) -> None:
        artifact_id = self._inspection.artifact_id
        attempts = self._records[artifact_id].attempts
        self._set_record(artifact_id, InspectionState.VISITED, attempts)
        self._log(f'artefact {artifact_id} inspected' + (f' ({reason})' if reason else ''))
        self._finish_inspection()

    def _finish_inspection(self) -> None:
        # The next exploration goal preempts any running viewpoint goal.
        self._inspection = None
        self.mode = Mode.EXPLORING
        self._last_plan_ns = None

    def _set_record(self, artifact_id, state, attempts, reason='') -> None:
        self._records[artifact_id] = InspectionRecord(artifact_id, state, attempts, reason)


def _xy(pose: PoseStamped) -> tuple[float, float]:
    return (pose.pose.position.x, pose.pose.position.y)
