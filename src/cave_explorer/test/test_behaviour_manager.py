"""Tests for the exploration/inspection state machine.

Navigation, exploration and B's inspection planner are fakes the tests drive by
hand, so every scenario in Workplan 4.6 and the task sheet's Planning 3
sequence can be stepped through tick by tick. No ROS runtime is needed.
"""

from types import SimpleNamespace

from geometry_msgs.msg import PoseStamped
import pytest

from cave_explorer.behaviour_manager import BehaviourConfig, BehaviourManager, Mode
from cave_explorer.interfaces import (
    ArtifactTrack,
    ExplorationGoal,
    ExplorationGoalStatus,
    GREEN_ALIEN,
    InspectionAssessment,
    InspectionEvaluation,
    InspectionGoal,
    InspectionGoalStatus,
    InspectionState,
    NavigationProgress,
    NavigationStatus,
    WHITE_SPHERE,
)

SECOND = 10**9
ROBOT = SimpleNamespace(x=0.0, y=0.0)
MAP = object()     # the manager only passes the map through


def pose(x, y=0.0):
    goal = PoseStamped()
    goal.header.frame_id = 'map'
    goal.pose.position.x, goal.pose.position.y = float(x), float(y)
    goal.pose.orientation.w = 1.0
    return goal


def track(artifact_id=1, x=5.0, class_name=GREEN_ALIEN, confirmed=True):
    return ArtifactTrack(artifact_id, class_name, (x, 0.0, 0.5), 0, 3, confirmed)


class FakeNavigator:
    def __init__(self):
        self.sent = []          # (request_id, goal, timeout)
        self.status = {}

    def send(self, goal, timeout_sec=None):
        request_id = len(self.sent) + 1
        self.sent.append((request_id, goal, timeout_sec))
        self.status[request_id] = NavigationProgress(NavigationStatus.ACTIVE)
        return request_id

    def poll(self, request_id):
        return self.status[request_id]

    def finish(self, request_id, status, reason=''):
        self.status[request_id] = NavigationProgress(status, reason=reason)

    @property
    def last_id(self):
        return self.sent[-1][0]

    @property
    def last_goal(self):
        return self.sent[-1][1]


class FakeExplorer:
    """Returns frontier goals at x = 100, 101, ...; ``status`` overrides that."""

    def __init__(self):
        self.status = ExplorationGoalStatus.READY
        self.reason = ''
        self.next_x = 100.0
        self.reports = []
        self.frontiers = ()

    def select_goal(self, map_msg, robot_xy, now_ns):
        if self.status is not ExplorationGoalStatus.READY:
            return ExplorationGoal(self.status, None, self.reason or 'scripted')
        goal = pose(self.next_x)
        self.next_x += 1.0
        return ExplorationGoal(ExplorationGoalStatus.READY, goal, 'scripted')

    def report_outcome(self, goal, status, reason, now_ns):
        self.reports.append((goal.pose.position.x, status))

    def keep_frontier_at(self, x):
        self.frontiers = (SimpleNamespace(goal=(x, 0.0), rejection=''),)


class FakePlanner:
    """Viewpoints at x = -1, -2, ...; ``assessment`` is what evaluate returns."""

    def __init__(self):
        self.assessment = InspectionEvaluation.PENDING
        self.goal_status = InspectionGoalStatus.READY
        self.excluded = []
        self._viewpoints = 0

    def make_goal(self, track, robot_pose, map_msg, *, exclude=()):
        self.excluded.append(tuple(goal.pose.position.x for goal in exclude))
        if self.goal_status is not InspectionGoalStatus.READY:
            return InspectionGoal(self.goal_status, None, 'scripted')
        self._viewpoints += 1
        return InspectionGoal(InspectionGoalStatus.READY, pose(-self._viewpoints), '')

    def evaluate(self, track, robot_pose, now_ns):
        return InspectionAssessment(self.assessment, 'scripted')


class World:
    """A manager wired to fakes, with a clock that advances one tick per step."""

    def __init__(self, inspect=frozenset({GREEN_ALIEN}), planner=True, **config):
        self.nav = FakeNavigator()
        self.explorer = FakeExplorer()
        self.planner = FakePlanner() if planner else None
        self.log = []
        self.manager = BehaviourManager(
            self.nav, self.explorer, self.planner,
            BehaviourConfig(inspect_classes=inspect, **config), log=self.log.append)
        self.now = 0
        self.robot = ROBOT

    def step(self, tracks=(), seconds=0.2):
        self.now += int(seconds * SECOND)
        self.manager.step(self.now, self.robot, MAP, tracks)

    def record(self, artifact_id=1):
        return next(r for r in self.manager.records if r.artifact_id == artifact_id)


# ----------------------------------------------------------------------
# Exploring
# ----------------------------------------------------------------------

def test_explores_and_reports_each_outcome_to_the_explorer():
    world = World()
    world.step()
    assert world.manager.mode is Mode.EXPLORING
    assert world.nav.last_goal.pose.position.x == 100.0

    world.nav.finish(1, NavigationStatus.ABORTED, 'stuck')
    world.step()
    assert world.explorer.reports == [(100.0, NavigationStatus.ABORTED)]
    assert world.nav.last_goal.pose.position.x == 101.0


def test_waits_quietly_for_a_map():
    world = World()
    world.explorer.status = ExplorationGoalStatus.NOT_READY
    for _ in range(5):
        world.step()
    assert world.nav.sent == []
    assert world.manager.status.startswith('waiting')


def test_no_frontier_pauses_then_retries_and_resumes():
    world = World(exhausted_retry_sec=5.0)
    world.explorer.status = ExplorationGoalStatus.NO_REACHABLE_FRONTIER
    world.step()
    assert world.manager.mode is Mode.EXHAUSTED

    world.explorer.status = ExplorationGoalStatus.READY
    world.step(seconds=1.0)
    assert world.nav.sent == []                       # not due yet
    world.step(seconds=5.0)
    assert world.manager.mode is Mode.EXPLORING
    assert len(world.nav.sent) == 1


def test_keeps_driving_while_its_frontier_still_exists():
    world = World(replan_period_sec=4.0)
    world.step()
    world.explorer.keep_frontier_at(100.0)
    world.step(seconds=5.0)
    assert len(world.nav.sent) == 1


def test_switches_goal_once_its_frontier_has_been_mapped():
    world = World(replan_period_sec=4.0)
    world.step()
    world.explorer.frontiers = ()
    world.step(seconds=5.0)
    assert len(world.nav.sent) == 2
    assert world.explorer.reports == []    # the preempted goal is not reported


def test_explores_only_when_the_planner_is_missing():
    world = World(planner=False)
    world.step(tracks=[track()])
    assert world.manager.mode is Mode.EXPLORING
    assert world.record(1).state is InspectionState.UNVISITED


# ----------------------------------------------------------------------
# Choosing what to inspect
# ----------------------------------------------------------------------

def test_confirmed_target_interrupts_exploration_by_sending_a_viewpoint():
    world = World()
    world.step()
    world.step(tracks=[track()])
    assert world.manager.mode is Mode.INSPECTING
    assert world.nav.last_goal.pose.position.x == -1.0
    assert world.record(1).state is InspectionState.IN_PROGRESS
    assert world.record(1).attempts == 1

    world.nav.finish(1, NavigationStatus.CANCELED, 'superseded by request 2')
    world.step(tracks=[track()])
    assert world.explorer.reports == []    # the interrupted exploration goal


@pytest.mark.parametrize('ignored', [
    track(class_name=WHITE_SPHERE),        # not one of the chosen classes
    track(confirmed=False),
])
def test_other_classes_and_unconfirmed_tracks_are_ignored(ignored):
    world = World()
    world.step(tracks=[ignored])
    assert world.manager.mode is Mode.EXPLORING
    assert world.manager.records == ()


def test_nearest_target_goes_first():
    world = World()
    world.step(tracks=[track(1, x=20.0), track(2, x=3.0)])
    assert world.record(2).state is InspectionState.IN_PROGRESS
    assert world.record(1).state is InspectionState.UNVISITED


# ----------------------------------------------------------------------
# Success
# ----------------------------------------------------------------------

def test_satisfied_marks_visited_and_is_never_retriggered():
    world = World()
    world.step(tracks=[track()])
    world.planner.assessment = InspectionEvaluation.SATISFIED
    world.step(tracks=[track()])
    assert world.record(1).state is InspectionState.VISITED
    assert world.manager.mode is Mode.EXPLORING

    sent = len(world.nav.sent)
    world.planner.assessment = InspectionEvaluation.PENDING
    for _ in range(5):
        world.step(tracks=[track()])
    assert world.manager.mode is Mode.EXPLORING
    assert all(goal.pose.position.x >= 100 for _, goal, _ in world.nav.sent[sent:])


def test_nav2_success_alone_is_not_an_inspection():
    world = World(dwell_timeout_sec=10.0, max_attempts=1)
    world.step(tracks=[track()])
    world.nav.finish(world.nav.last_id, NavigationStatus.SUCCEEDED)
    world.step(tracks=[track()])
    world.step(tracks=[track()], seconds=5.0)
    assert world.record(1).state is InspectionState.IN_PROGRESS

    world.step(tracks=[track()], seconds=6.0)
    record = world.record(1)
    assert record.state is InspectionState.DEFERRED
    assert 'never satisfied (scripted)' in record.last_failure_reason   # the planner's reason


def test_satisfied_before_arriving_counts():
    world = World()
    world.step(tracks=[track()])
    world.planner.assessment = InspectionEvaluation.SATISFIED
    world.step(tracks=[track()])
    assert world.record(1).state is InspectionState.VISITED


# ----------------------------------------------------------------------
# Failure: retry once, then defer for the rest of the run
# ----------------------------------------------------------------------

def arrive(world):
    """Finish the current viewpoint goal and let the manager see the arrival."""
    world.nav.finish(world.nav.last_id, NavigationStatus.SUCCEEDED)
    world.step(tracks=[track()])


def test_lost_twice_defers_and_a_deferred_target_never_returns():
    world = World()
    world.step(tracks=[track()])
    world.planner.assessment = InspectionEvaluation.LOST
    arrive(world)
    world.step(tracks=[track()])
    assert world.record(1).state is InspectionState.IN_PROGRESS
    assert world.record(1).attempts == 2

    world.step(tracks=[track()])
    arrive(world)
    world.step(tracks=[track()])
    record = world.record(1)
    assert record.state is InspectionState.DEFERRED
    assert record.attempts == 2
    assert 'lost' in record.last_failure_reason
    assert world.manager.mode is Mode.EXPLORING

    world.planner.assessment = InspectionEvaluation.PENDING
    for _ in range(5):
        world.step(tracks=[track()])            # still visible, still deferred
    assert world.record(1).state is InspectionState.DEFERRED
    assert world.manager.mode is Mode.EXPLORING


def test_rejected_viewpoint_tries_another_without_using_an_attempt():
    world = World()
    world.step(tracks=[track()])
    world.nav.finish(world.nav.last_id, NavigationStatus.REJECTED)
    world.step(tracks=[track()])
    world.step(tracks=[track()])
    assert world.nav.last_goal.pose.position.x == -2.0
    assert world.planner.excluded == [(), (-1.0,)]
    assert world.record(1).attempts == 1


def test_running_out_of_viewpoints_uses_up_an_attempt():
    world = World(max_viewpoints=2)
    world.step(tracks=[track()])
    for _ in range(2):
        world.nav.finish(world.nav.last_id, NavigationStatus.ABORTED)
        world.step(tracks=[track()])
        world.step(tracks=[track()])
    record = world.record(1)
    assert record.attempts == 2
    assert 'no reachable viewpoint' in record.last_failure_reason
    world.step(tracks=[track()])
    assert world.planner.excluded[-1] == (-1.0, -2.0)   # the retry avoids both


def test_no_safe_viewpoint_fails_the_attempt():
    world = World(max_attempts=1)
    world.planner.goal_status = InspectionGoalStatus.NO_SAFE_GOAL
    world.step(tracks=[track()])
    assert world.record(1).state is InspectionState.DEFERRED
    assert world.manager.mode is Mode.EXPLORING


def test_attempt_times_out():
    world = World(attempt_timeout_sec=60.0, max_attempts=1)
    world.step(tracks=[track()])
    world.step(tracks=[track()], seconds=61.0)
    assert world.record(1).state is InspectionState.DEFERRED
    assert 'within 60 s' in world.record(1).last_failure_reason


def test_target_dropped_by_the_tracker_fails_the_attempt():
    world = World(max_attempts=1)
    world.step(tracks=[track()])
    world.step(tracks=[])
    assert world.record(1).state is InspectionState.DEFERRED
    assert 'no longer reported' in world.record(1).last_failure_reason


def test_next_target_starts_right_after_one_is_deferred():
    world = World(max_attempts=1)
    world.planner.goal_status = InspectionGoalStatus.NO_SAFE_GOAL
    tracks = [track(1, x=2.0), track(2, x=9.0)]
    world.step(tracks=tracks)
    assert world.record(1).state is InspectionState.DEFERRED
    assert world.record(2).state is InspectionState.DEFERRED
    assert world.manager.mode is Mode.EXPLORING
    # Only the exploration goal sent after both failed; none wasted in between.
    assert [goal.pose.position.x for _, goal, _ in world.nav.sent] == [100.0]


def test_records_are_a_read_only_snapshot():
    world = World()
    world.step(tracks=[track()])
    snapshot = world.manager.records
    assert isinstance(snapshot, tuple)
    with pytest.raises(Exception):
        snapshot[0].state = InspectionState.VISITED


# ----------------------------------------------------------------------
# Nav2 reporting success away from the goal
# ----------------------------------------------------------------------

def test_exploration_success_away_from_goal_is_ignored():
    world = World()
    world.step()
    goal_x = world.nav.last_goal.pose.position.x
    world.robot = SimpleNamespace(x=goal_x - 9.0, y=0.0)
    world.nav.finish(world.nav.last_id, NavigationStatus.SUCCEEDED)
    world.step()
    assert world.explorer.reports[-1] == (goal_x, NavigationStatus.CANCELED)

    world.robot = SimpleNamespace(x=world.nav.last_goal.pose.position.x - 0.5, y=0.0)
    world.nav.finish(world.nav.last_id, NavigationStatus.SUCCEEDED)
    world.step()
    assert world.explorer.reports[-1][1] is NavigationStatus.SUCCEEDED


def test_viewpoint_success_away_from_goal_resends_it():
    world = World()
    world.robot = SimpleNamespace(x=-10.0, y=0.0)
    world.step(tracks=[track()])
    viewpoint = world.nav.last_goal
    world.nav.finish(world.nav.last_id, NavigationStatus.SUCCEEDED)
    world.step(tracks=[track()])
    assert world.nav.last_goal is viewpoint
    assert len(world.nav.sent) == 2
    assert world.record(1).state is InspectionState.IN_PROGRESS


def test_lost_before_reaching_the_viewpoint_is_ignored():
    """Regression: a retry used to fail on its first tick, still not seeing it."""
    world = World()
    world.step(tracks=[track()])
    world.planner.assessment = InspectionEvaluation.LOST
    for _ in range(5):
        world.step(tracks=[track()])
    assert world.record(1).state is InspectionState.IN_PROGRESS
    assert world.record(1).attempts == 1


def test_retry_drives_to_a_new_viewpoint():
    world = World()
    world.step(tracks=[track()])
    first = world.nav.last_goal
    world.planner.assessment = InspectionEvaluation.LOST
    arrive(world)
    world.step(tracks=[track()])          # attempt 1 fails
    world.step(tracks=[track()])          # attempt 2 plans its viewpoint
    assert world.record(1).attempts == 2
    assert world.nav.last_goal is not first
    assert world.planner.excluded[-1] == (first.pose.position.x,)


# ----------------------------------------------------------------------
# Stuck watchdog
# ----------------------------------------------------------------------

def test_exploration_goal_without_progress_is_given_up():
    """Regression: the robot sat on green_crystals for three 120 s timeouts."""
    world = World(stuck_timeout_sec=20.0)
    world.step()
    stuck_goal = world.nav.last_goal
    world.explorer.keep_frontier_at(stuck_goal.pose.position.x)
    for _ in range(5):
        world.step(seconds=5.0)              # the clock starts on the first check
    assert world.nav.last_goal is stuck_goal
    world.step(seconds=5.0)                  # now more than 20 s without moving
    assert world.explorer.reports[-1] == (stuck_goal.pose.position.x, NavigationStatus.ABORTED)
    assert world.nav.last_goal is not stuck_goal
    assert any('no progress' in line for line in world.log)


def test_a_moving_robot_is_not_stuck():
    world = World(stuck_timeout_sec=20.0)
    world.step()
    goal = world.nav.last_goal
    world.explorer.keep_frontier_at(goal.pose.position.x)
    for i in range(10):
        world.robot = SimpleNamespace(x=0.6 * (i + 1), y=0.0)
        world.step(seconds=5.0)
    assert world.nav.last_goal is goal
    assert world.explorer.reports == []


def test_a_new_goal_gets_the_full_timeout():
    world = World(stuck_timeout_sec=20.0)
    world.step()
    world.explorer.keep_frontier_at(world.nav.last_goal.pose.position.x)
    world.step(seconds=15.0)
    world.nav.finish(world.nav.last_id, NavigationStatus.SUCCEEDED)
    world.robot = SimpleNamespace(x=world.nav.last_goal.pose.position.x, y=0.0)
    world.step()                             # reached; next goal sent from here
    second = world.nav.last_goal
    world.explorer.keep_frontier_at(second.pose.position.x)
    world.step(seconds=10.0)
    world.step(seconds=8.0)
    world.step(seconds=8.0)                  # 16 s since the first check: not stuck yet
    assert world.nav.last_goal is second


def test_stuck_on_the_way_to_a_viewpoint_tries_another():
    world = World(stuck_timeout_sec=20.0)
    world.step(tracks=[track()])
    first = world.nav.last_goal
    for _ in range(6):
        world.step(tracks=[track()], seconds=5.0)
    world.step(tracks=[track()])             # plans the next viewpoint
    assert world.nav.last_goal is not first
    assert world.record(1).state is InspectionState.IN_PROGRESS
    assert world.record(1).attempts == 1


def test_switching_goals_while_stuck_does_not_reset_the_watchdog():
    world = World(stuck_timeout_sec=20.0)
    world.step()
    for _ in range(6):
        world.step(seconds=5.0)              # the frontier keeps "moving": goal switches
    assert any('no progress' in line for line in world.log)
