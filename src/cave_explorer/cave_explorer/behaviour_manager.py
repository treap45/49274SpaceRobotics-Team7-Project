"""Exploration/inspection state machine and inspection records.

Owner: C.
Contract: repository-root Workplan.md, sections 4.4-4.6.

Planning 3. Alternates between exploring (Planning 1) and close-range
inspection (Planning 2), following the task sheet's sequence:

1. Explore with frontier goals.
2. When a confirmed artefact of a chosen class appears, stop exploring and
   approach it.
3. If it is lost or cannot be reached, retry once; if that also fails, give up
   on it and resume exploring.
4. After a successful inspection, record it as visited and resume exploring.

The manager is plain Python with no ROS runtime: the main node calls ``step``
from its timer with the current time, robot pose, map and B's track snapshot,
and the collaborators are passed in. That keeps the whole state machine
testable in milliseconds with fakes.

Rules carried over from Workplan.md:

* C owns inspection records; B owns observations (4.6). Records exist for
  confirmed tracks of the inspection classes only.
* ``VISITED`` needs B's ``evaluate`` to say ``SATISFIED``. A Nav2 success alone
  is not an inspection (4.4).
* A Nav2 rejection moves on to another viewpoint, excluding the ones already
  tried; it does not by itself use up an attempt (4.4).
* ``DEFERRED`` is terminal for the run (4.6).
* Goals are switched by sending the new goal, never cancel-then-send:
  navigation_client explains why, and reports the preempted goal as
  ``CANCELED`` so nothing gets penalised for it.

Assumption pending B's review: ``InspectionPlanner.make_goal`` accepts an
``exclude`` keyword listing viewpoints already tried (Workplan 4.4 offers this
or a candidate list). If B returns a list instead, only ``_plan_viewpoint``
changes.
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
    """Tuning for the state machine. Times are seconds of ROS (simulation) time.

    ``inspect_classes`` is the team's choice of 2-3 reliably detected classes
    (Planning 2) and has no default: an empty set means explore only.
    """

    inspect_classes: frozenset = frozenset()
    max_attempts: int = 2               # the first try plus one retry
    max_viewpoints: int = 3             # per attempt, after Nav2 rejections
    attempt_timeout_sec: float = 60.0   # one approach, start to finish
    dwell_timeout_sec: float = 10.0     # arrived, waiting for SATISFIED
    explore_goal_timeout_sec: float = 120.0
    replan_period_sec: float = 4.0      # SLAM republishes the map every 2 s
    switch_distance_m: float = 2.0      # current goal counts as still valid within this
    exhausted_retry_sec: float = 5.0


@dataclass
class _Exploration:
    request_id: int
    goal: PoseStamped


@dataclass
class _Inspection:
    artifact_id: int
    started_ns: int
    tried: list = field(default_factory=list)
    request_id: int | None = None
    arrived_ns: int | None = None


class BehaviourManager:
    """Decides, every main-loop tick, whether to explore or inspect and what to send."""

    def __init__(self, navigator: Navigator, explorer: Explorer,
                 planner: InspectionPlanner | None,
                 config: BehaviourConfig = BehaviourConfig(),
                 log: Callable[[str], None] | None = None):
        """``planner`` may be ``None`` until B's module exists: the robot then only explores."""
        self._navigator = navigator
        self._explorer = explorer
        self._planner = planner
        self.config = config
        self._log = log or (lambda message: None)

        self.mode = Mode.EXPLORING
        self.status = 'starting'
        self._records: dict[int, InspectionRecord] = {}
        self._exploration: _Exploration | None = None
        self._inspection: _Inspection | None = None
        self._last_plan_ns: int | None = None

    @property
    def records(self) -> tuple[InspectionRecord, ...]:
        """Read-only snapshot of every inspection record, for B's markers (Workplan 4.6)."""
        return tuple(self._records[key] for key in sorted(self._records))

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

        if self.mode is Mode.INSPECTING:
            self._inspect(now_ns, robot_pose, map_msg, targets)
        # A target can fail at once (no safe viewpoint); move straight on to the
        # next one rather than sending an exploration goal only to preempt it.
        # Each pass should take a record off UNVISITED; the bound keeps a bug in
        # that from freezing the node's timer.
        for _ in range(len(targets)):
            if self.mode is Mode.INSPECTING or self._planner is None:
                break
            target = self._nearest_unvisited(robot_pose, targets)
            if target is None:
                break
            self._begin_inspection(target, now_ns)
            self._inspect(now_ns, robot_pose, map_msg, targets)
        if self.mode is not Mode.INSPECTING:
            self._explore(now_ns, robot_pose, map_msg)

    # ------------------------------------------------------------------
    # Exploring
    # ------------------------------------------------------------------

    def _explore(self, now_ns, robot_pose, map_msg) -> None:
        current = self._exploration
        if current is not None:
            progress = self._navigator.poll(current.request_id)
            if not progress.is_terminal:
                if self._due(now_ns, self.config.replan_period_sec):
                    self._replan_while_driving(now_ns, robot_pose, map_msg, current)
                return
            self._explorer.report_outcome(current.goal, progress.status, progress.reason,
                                          now_ns)
            self._exploration = None

        if self.mode is Mode.EXHAUSTED and not self._due(now_ns,
                                                         self.config.exhausted_retry_sec):
            return
        result = self._plan(now_ns, robot_pose, map_msg)
        if result.status is ExplorationGoalStatus.READY:
            self._send_exploration(result)
        elif result.status is ExplorationGoalStatus.NOT_READY:
            self.mode = Mode.EXPLORING
            self.status = f'waiting: {result.reason}'
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
            self._send_exploration(result)

    def _send_exploration(self, result: ExplorationGoal) -> None:
        request_id = self._navigator.send(result.goal,
                                          timeout_sec=self.config.explore_goal_timeout_sec)
        self._exploration = _Exploration(request_id, result.goal)
        self.mode = Mode.EXPLORING
        self.status = f'exploring: {result.reason}'

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

    def _begin_inspection(self, track: ArtifactTrack, now_ns: int) -> None:
        # Any exploration goal is simply abandoned: sending the viewpoint
        # preempts it, and its CANCELED result is never reported.
        self._exploration = None
        self._inspection = _Inspection(track.artifact_id, now_ns)
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
        if now_ns - inspection.started_ns > self.config.attempt_timeout_sec * 1e9:
            self._attempt_failed(
                now_ns, f'not inspected within {self.config.attempt_timeout_sec:g} s')
            return

        assessment = self._planner.evaluate(track, robot_pose, now_ns)
        if assessment.status is InspectionEvaluation.SATISFIED:
            self._inspection_succeeded(assessment.reason)
            return
        if assessment.status is InspectionEvaluation.LOST:
            self._attempt_failed(now_ns, f'artefact lost: {assessment.reason}')
            return

        if inspection.request_id is None:
            self._plan_viewpoint(now_ns, robot_pose, map_msg, track)
            return
        progress = self._navigator.poll(inspection.request_id)
        if not progress.is_terminal:
            return
        if progress.succeeded:
            if inspection.arrived_ns is None:
                inspection.arrived_ns = now_ns
            elif now_ns - inspection.arrived_ns > self.config.dwell_timeout_sec * 1e9:
                self._attempt_failed(now_ns, 'reached the viewpoint but inspection was '
                                     'never satisfied')
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
                                       exclude=tuple(inspection.tried))
        if plan.status is InspectionGoalStatus.NOT_READY:
            return     # bounded by the attempt timeout
        if plan.status is InspectionGoalStatus.NO_SAFE_GOAL:
            self._attempt_failed(now_ns, f'no safe viewpoint: {plan.reason}')
            return
        inspection.tried.append(plan.goal)
        inspection.request_id = self._navigator.send(
            plan.goal, timeout_sec=self.config.attempt_timeout_sec)

    def _attempt_failed(self, now_ns, reason: str) -> None:
        artifact_id = self._inspection.artifact_id
        attempts = self._records[artifact_id].attempts
        if attempts < self.config.max_attempts:
            self._log(f'artefact {artifact_id} attempt {attempts} failed ({reason}); retrying')
            self._inspection = _Inspection(artifact_id, now_ns)
            self._set_record(artifact_id, InspectionState.IN_PROGRESS, attempts + 1, reason)
            self.status = f'inspecting artefact {artifact_id}, attempt {attempts + 1}'
            return
        self._set_record(artifact_id, InspectionState.DEFERRED, attempts, reason)
        self._log(f'artefact {artifact_id} deferred after {attempts} attempts ({reason})')
        self._finish_inspection()

    def _inspection_succeeded(self, reason: str) -> None:
        artifact_id = self._inspection.artifact_id
        attempts = self._records[artifact_id].attempts
        self._set_record(artifact_id, InspectionState.VISITED, attempts)
        self._log(f'artefact {artifact_id} inspected' + (f' ({reason})' if reason else ''))
        self._finish_inspection()

    def _finish_inspection(self) -> None:
        # Whatever viewpoint goal is still running is preempted by the next
        # exploration goal.
        self._inspection = None
        self.mode = Mode.EXPLORING
        self._last_plan_ns = None

    def _set_record(self, artifact_id, state, attempts, reason='') -> None:
        self._records[artifact_id] = InspectionRecord(artifact_id, state, attempts, reason)


def _xy(pose: PoseStamped) -> tuple[float, float]:
    return (pose.pose.position.x, pose.pose.position.y)
