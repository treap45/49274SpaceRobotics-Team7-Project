"""Tests for observation poses and inspection evaluation (Workplan 4.4).

The last test drives the real planner through BehaviourManager, so a signature
drift between B's module and the state machine fails here, not in simulation.
"""

import math
from types import SimpleNamespace

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest

from cave_explorer.behaviour_manager import BehaviourConfig, BehaviourManager
from cave_explorer.inspection_planner import InspectionPlanner
from cave_explorer.interfaces import (
    ArtifactTrack,
    GREEN_ALIEN,
    GREEN_CRYSTALS,
    InspectionEvaluation,
    InspectionGoalStatus,
)

SECOND = 10**9
RESOLUTION = 0.1


def make_map(cells, origin=(-5.0, -5.0)):
    """``cells`` is a (rows, cols) int array; row 0 is the lowest y."""
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.resolution = RESOLUTION
    msg.info.height, msg.info.width = cells.shape
    msg.info.origin.position.x, msg.info.origin.position.y = origin
    msg.info.origin.orientation.w = 1.0
    msg.data = [int(v) for v in cells.ravel()]
    return msg


def open_room(size=100):
    """10 m x 10 m free room centred on the origin, walls round the edge."""
    cells = np.zeros((size, size), dtype=np.int16)
    cells[0, :] = cells[-1, :] = cells[:, 0] = cells[:, -1] = 100
    return cells


def track(x=0.0, y=0.0, last_seen_s=0.0, artifact_id=1, class_name=GREEN_ALIEN):
    return ArtifactTrack(artifact_id, class_name, (x, y, 0.3),
                         int(last_seen_s * SECOND), 3, True)


def robot(x, y, theta=0.0):
    return SimpleNamespace(x=x, y=y, theta=theta)


def xy(goal):
    return goal.pose.position.x, goal.pose.position.y


def yaw(goal):
    q = goal.pose.orientation
    return 2.0 * math.atan2(q.z, q.w)


# --------------------------------------------------------------------------
# make_goal
# --------------------------------------------------------------------------

def test_goal_on_robot_side_at_stand_off_facing_artefact():
    planner = InspectionPlanner()
    plan = planner.make_goal(track(), robot(-4.0, 0.0), make_map(open_room()))
    assert plan.status is InspectionGoalStatus.READY
    assert xy(plan.goal) == pytest.approx((-1.2, 0.0), abs=1e-6)
    assert yaw(plan.goal) == pytest.approx(0.0, abs=1e-6)
    assert plan.goal.header.frame_id == 'map'


def test_keep_out_type_is_viewed_from_outside_its_disk():
    """green_crystals' 2.7 m keep-out disk plus the 1.0 m margin: 3.7 m."""
    planner = InspectionPlanner()
    assert planner.stand_off_for(GREEN_ALIEN) == pytest.approx(1.2)
    assert planner.stand_off_for(GREEN_CRYSTALS) == pytest.approx(3.7)
    plan = planner.make_goal(track(class_name=GREEN_CRYSTALS), robot(-4.5, 0.0),
                             make_map(open_room()))
    assert plan.status is InspectionGoalStatus.READY
    assert xy(plan.goal) == pytest.approx((-3.7, 0.0), abs=1e-6)


def test_keep_out_type_is_satisfied_at_its_own_stand_off():
    """Sightings at 4.5 m count as close for a 3.7 m viewpoint, beyond sighting_range_m."""
    planner = InspectionPlanner(hold_time_sec=0.0, sighting_range_m=4.0)
    crystals = dict(class_name=GREEN_CRYSTALS)
    planner.evaluate(track(last_seen_s=0, **crystals), robot(-8.0, 0.0), 0)
    planner.evaluate(track(last_seen_s=1, **crystals), robot(-4.5, 0.0), SECOND)
    assert planner.evaluate(track(last_seen_s=1, **crystals), robot(-3.7, 0.0),
                            2 * SECOND).status is InspectionEvaluation.SATISFIED
    # 1.2 m away is far off the 3.7 m stand-off.
    assert planner.evaluate(track(last_seen_s=1, **crystals), robot(-1.2, 0.0),
                            3 * SECOND).status is InspectionEvaluation.PENDING


def test_exclude_gives_a_different_pose():
    planner = InspectionPlanner()
    room = make_map(open_room())
    first = planner.make_goal(track(), robot(-4.0, 0.0), room).goal
    second = planner.make_goal(track(), robot(-4.0, 0.0), room, exclude=(first,)).goal
    assert math.dist(xy(first), xy(second)) > 0.5
    assert math.dist(xy(second), (0.0, 0.0)) == pytest.approx(1.2, abs=1e-6)
    # Still faces the artefact.
    assert yaw(second) == pytest.approx(
        math.atan2(-second.pose.position.y, -second.pose.position.x), abs=1e-6)


def test_retry_looks_from_another_side():
    """As in simulation: green_crystals viewed from 3.7 m, and the robot has
    moved, so the same bearing gives a pose 0.7 m from the first. A distance
    exclusion lets it through; the angle exclusion does not."""
    planner = InspectionPlanner()
    room = make_map(open_room())
    crystals = track(class_name=GREEN_CRYSTALS)
    first = planner.make_goal(crystals, robot(-4.5, 0.0), room).goal
    second = planner.make_goal(crystals, robot(-4.5, -0.9), room, exclude=(first,)).goal
    first_bearing = math.atan2(first.pose.position.y, first.pose.position.x)
    second_bearing = math.atan2(second.pose.position.y, second.pose.position.x)
    assert abs(math.remainder(second_bearing - first_bearing, math.tau)) >= math.radians(30)


def test_wall_next_to_preferred_pose_moves_the_goal():
    cells = open_room()
    cells[:, 37:40] = 100            # wall at x = -1.3 .. -1.0 m
    plan = InspectionPlanner().make_goal(track(), robot(-4.0, 0.0), make_map(cells))
    assert plan.status is InspectionGoalStatus.READY
    # Not on the blocked side.
    assert plan.goal.pose.position.x > -0.6


def test_unknown_cells_are_not_used():
    cells = np.full((100, 100), -1, dtype=np.int16)
    plan = InspectionPlanner().make_goal(track(), robot(-4.0, 0.0), make_map(cells))
    assert plan.status is InspectionGoalStatus.NO_SAFE_GOAL
    assert 'not free' in plan.reason


def test_line_of_sight_through_a_wall_is_rejected():
    cells = open_room()
    cells[:, 45:47] = 100            # thin wall at x = -0.5 m, between pose and artefact
    cells[:, :45] = np.where(cells[:, :45] == 100, 100, 0)
    planner = InspectionPlanner(bearing_offsets_deg=(0,), extra_stand_off_m=(0.0,))
    plan = planner.make_goal(track(), robot(-4.0, 0.0), make_map(cells))
    assert plan.status is InspectionGoalStatus.NO_SAFE_GOAL


def test_no_map_is_not_ready():
    plan = InspectionPlanner().make_goal(track(), robot(0.0, 0.0), None)
    assert plan.status is InspectionGoalStatus.NOT_READY


def test_artefact_outside_map_is_not_ready():
    plan = InspectionPlanner().make_goal(track(x=20.0), robot(0.0, 0.0),
                                         make_map(open_room()))
    assert plan.status is InspectionGoalStatus.NOT_READY


def test_every_candidate_excluded_is_no_safe_goal():
    planner = InspectionPlanner()
    room = make_map(open_room())
    tried = []
    for _ in range(40):
        plan = planner.make_goal(track(), robot(-4.0, 0.0), room, exclude=tuple(tried))
        if plan.status is not InspectionGoalStatus.READY:
            break
        tried.append(plan.goal)
    assert plan.status is InspectionGoalStatus.NO_SAFE_GOAL
    assert len(tried) == 8           # one per bearing: a tried bearing excludes both stand-offs


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------

def test_satisfied_only_after_hold_time_in_ros_time():
    planner = InspectionPlanner(hold_time_sec=2.0)
    here = robot(-1.2, 0.0, theta=0.0)
    planner.evaluate(track(last_seen_s=9), here, 9 * SECOND)        # first call: baseline
    assert planner.evaluate(track(last_seen_s=10), here, 10 * SECOND).status \
        is InspectionEvaluation.PENDING                             # new sighting, hold starts
    assert planner.evaluate(track(last_seen_s=11), here, 11 * SECOND).status \
        is InspectionEvaluation.PENDING
    assert planner.evaluate(track(last_seen_s=12), here, 12 * SECOND).status \
        is InspectionEvaluation.SATISFIED


def test_wrong_heading_or_distance_is_pending():
    planner = InspectionPlanner(hold_time_sec=0.0)
    assert planner.evaluate(track(last_seen_s=1), robot(-1.2, 0.0, math.pi), SECOND).status \
        is InspectionEvaluation.PENDING
    assert planner.evaluate(track(last_seen_s=1), robot(-3.0, 0.0), SECOND).status \
        is InspectionEvaluation.PENDING


def test_never_seen_up_close_is_not_satisfied():
    planner = InspectionPlanner(hold_time_sec=0.0, sighting_range_m=4.0)
    # Last seen from 8 m away, then the robot drove in without seeing it again.
    planner.evaluate(track(last_seen_s=0), robot(-8.0, 0.0), 0)
    result = planner.evaluate(track(last_seen_s=0), robot(-1.2, 0.0), 3 * SECOND)
    assert result.status is InspectionEvaluation.PENDING
    assert 'not seen within 4 m' in result.reason


def test_seen_on_the_approach_counts_without_a_detection_at_the_viewpoint():
    """The detector loses artefacts inside ~2 m; a sighting at 3 m during the
    approach keeps the artefact "in view" for the task sheet's Planning 2."""
    planner = InspectionPlanner(hold_time_sec=2.0, sighting_window_sec=15.0)
    planner.evaluate(track(last_seen_s=9), robot(-5.0, 0.0), 9 * SECOND)
    planner.evaluate(track(last_seen_s=10), robot(-3.0, 0.0), 10 * SECOND)   # new, at 3 m
    here = robot(-1.2, 0.0)
    assert planner.evaluate(track(last_seen_s=10), here, 14 * SECOND).status \
        is InspectionEvaluation.PENDING
    assert planner.evaluate(track(last_seen_s=10), here, 16 * SECOND).status \
        is InspectionEvaluation.SATISFIED


def test_old_close_sighting_expires():
    def sighted_at_3m():
        planner = InspectionPlanner(hold_time_sec=0.0, sighting_window_sec=15.0)
        planner.evaluate(track(last_seen_s=-1), robot(-5.0, 0.0), -SECOND)
        planner.evaluate(track(last_seen_s=0), robot(-3.0, 0.0), 0)
        return planner

    result = sighted_at_3m().evaluate(track(last_seen_s=0), robot(-1.2, 0.0), 14 * SECOND)
    assert result.status is InspectionEvaluation.SATISFIED
    result = sighted_at_3m().evaluate(track(last_seen_s=0), robot(-1.2, 0.0), 16 * SECOND)
    assert result.status is not InspectionEvaluation.SATISFIED


def test_lost_when_close_and_unseen_but_not_when_far():
    planner = InspectionPlanner(sighting_window_sec=15.0)
    close, far = robot(-1.2, 0.0), robot(-4.0, 0.0)
    assert planner.evaluate(track(last_seen_s=0), close, 0).status \
        is not InspectionEvaluation.LOST
    assert planner.evaluate(track(last_seen_s=0), close, 10 * SECOND).status \
        is not InspectionEvaluation.LOST
    assert planner.evaluate(track(last_seen_s=0), close, 16 * SECOND).status \
        is InspectionEvaluation.LOST
    assert planner.evaluate(track(last_seen_s=0), far, 30 * SECOND).status \
        is InspectionEvaluation.PENDING


def test_lost_clock_starts_when_the_robot_comes_within_range():
    """An artefact last seen 30 s ago, while another was inspected, is not LOST
    the moment the robot reaches its viewpoint."""
    planner = InspectionPlanner(sighting_window_sec=15.0)
    planner.evaluate(track(last_seen_s=0), robot(-8.0, 0.0), 20 * SECOND)
    assert planner.evaluate(track(last_seen_s=0), robot(-1.2, 0.0), 30 * SECOND).status \
        is InspectionEvaluation.PENDING
    assert planner.evaluate(track(last_seen_s=0), robot(-1.2, 0.0), 44 * SECOND).status \
        is InspectionEvaluation.PENDING
    assert planner.evaluate(track(last_seen_s=0), robot(-1.2, 0.0), 46 * SECOND).status \
        is InspectionEvaluation.LOST


def test_new_viewpoint_restarts_the_lost_clock():
    planner = InspectionPlanner(sighting_window_sec=15.0)
    here = robot(-1.2, 0.0)
    planner.evaluate(track(last_seen_s=0), here, 0)
    assert planner.evaluate(track(last_seen_s=0), here, 16 * SECOND).status \
        is InspectionEvaluation.LOST
    planner.make_goal(track(), robot(-4.0, 0.0), make_map(open_room()))
    assert planner.evaluate(track(last_seen_s=0), here, 17 * SECOND).status \
        is InspectionEvaluation.PENDING


def test_hold_timer_is_per_artefact():
    planner = InspectionPlanner(hold_time_sec=2.0)
    here = robot(-1.2, 0.0)
    planner.evaluate(track(last_seen_s=0, artifact_id=1), here, 0)
    # Artefact 2 starts its own timer instead of inheriting artefact 1's.
    planner.evaluate(track(last_seen_s=2, artifact_id=2), here, 2 * SECOND)
    assert planner.evaluate(track(last_seen_s=3, artifact_id=2), here, 3 * SECOND).status \
        is InspectionEvaluation.PENDING


def test_hold_resets_when_conditions_break():
    planner = InspectionPlanner(hold_time_sec=2.0)
    here, away = robot(-1.2, 0.0), robot(-1.2, 0.0, math.pi)
    planner.evaluate(track(last_seen_s=0), here, 0)
    planner.evaluate(track(last_seen_s=1), away, SECOND)
    assert planner.evaluate(track(last_seen_s=2), here, 2 * SECOND).status \
        is InspectionEvaluation.PENDING


# --------------------------------------------------------------------------
# Wired into the state machine
# --------------------------------------------------------------------------

class _Navigator:
    def __init__(self):
        self.sent = []

    def send(self, goal, timeout_sec=None):
        self.sent.append(goal)
        return len(self.sent)

    def poll(self, request_id):
        from cave_explorer.interfaces import NavigationProgress, NavigationStatus
        return NavigationProgress(request_id, NavigationStatus.ACTIVE, '', None)

    def cancel(self, request_id):
        pass


class _Explorer:
    def select_goal(self, map_msg, robot_xy, now_ns):
        from cave_explorer.interfaces import ExplorationGoal, ExplorationGoalStatus
        return ExplorationGoal(ExplorationGoalStatus.EXHAUSTED, None, 'test', ())

    def report_outcome(self, goal, status, reason, now_ns):
        pass


def test_real_planner_works_inside_behaviour_manager():
    navigator = _Navigator()
    manager = BehaviourManager(
        navigator, _Explorer(), InspectionPlanner(),
        BehaviourConfig(inspect_classes=frozenset({GREEN_ALIEN})))
    room = make_map(open_room())
    manager.step(SECOND, robot(-4.0, 0.0), room, (track(last_seen_s=1),))
    assert navigator.sent, manager.status
    assert isinstance(navigator.sent[-1], PoseStamped)
    assert xy(navigator.sent[-1]) == pytest.approx((-1.2, 0.0), abs=1e-6)
