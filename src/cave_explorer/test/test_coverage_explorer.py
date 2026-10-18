"""Tests for camera coverage and the two-phase explorer.

Maps are synthetic OccupancyGrids with 0.3 m cells, the simulation's SLAM
resolution, so region sizes and clearances behave as they do in the cave.
"""

import math

from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest

from cave_explorer.coverage_explorer import CoverageExplorer, TwoPhaseExplorer
from cave_explorer.interfaces import (
    ExplorationGoal,
    ExplorationGoalStatus,
    NavigationStatus,
)

RES = 0.3
SECOND = 10**9


def make_map(cells, origin=(0.0, 0.0)):
    """``cells`` is a (rows, cols) array; row 0 is the lowest y."""
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.resolution = RES
    msg.info.height, msg.info.width = cells.shape
    msg.info.origin.position.x, msg.info.origin.position.y = origin
    msg.info.origin.orientation.w = 1.0
    msg.data = [int(v) for v in cells.ravel()]
    return msg


def room(rows=40, cols=100):
    """Free room 30 m x 12 m with walls round the edge."""
    cells = np.zeros((rows, cols), dtype=np.int16)
    cells[0, :] = cells[-1, :] = cells[:, 0] = cells[:, -1] = 100
    return cells


def seen_at(cov, x, y):
    return bool(cov._seen[int(y / RES), int(x / RES)])


def xy(goal):
    return goal.pose.position.x, goal.pose.position.y


# --------------------------------------------------------------------------
# observe
# --------------------------------------------------------------------------

def test_sees_ahead_within_range_but_not_behind_or_beyond():
    cov = CoverageExplorer(view_range_m=6.0)
    world = make_map(room())
    cov.observe(world, (5.0, 6.0), 0.0)
    assert seen_at(cov, 9.0, 6.0)               # 4 m ahead
    assert not seen_at(cov, 13.0, 6.0)          # 8 m ahead: beyond range
    assert not seen_at(cov, 2.0, 6.0)           # behind the camera


def test_walls_block_the_view():
    cells = room()
    cells[5:35, 30] = 100                       # wall at x = 9 m
    cov = CoverageExplorer(view_range_m=8.0)
    cov.observe(make_map(cells), (6.0, 6.0), 0.0)
    assert seen_at(cov, 8.5, 6.0)
    assert seen_at(cov, 9.1, 6.0)               # the wall itself is seen
    assert not seen_at(cov, 11.0, 6.0)          # behind it is not


def test_unknown_cells_are_not_marked_seen():
    cells = room()
    cells[:, 35:] = -1
    cov = CoverageExplorer(view_range_m=8.0)
    cov.observe(make_map(cells), (6.0, 6.0), 0.0)
    assert not seen_at(cov, 12.0, 6.0)


def test_seen_cells_survive_the_map_growing():
    cov = CoverageExplorer(view_range_m=6.0)
    cov.observe(make_map(room()), (5.0, 6.0), 0.0)
    grown = np.full((50, 110), -1, dtype=np.int16)
    grown[5:45, 10:110] = room()                # same room, origin moved by (-3, -1.5)
    msg = make_map(grown, origin=(-3.0, -1.5))
    cov.seen_fraction(msg)                      # realigns
    row, col = int((6.0 + 1.5) / RES), int((9.0 + 3.0) / RES)
    assert cov._seen[row, col]


# --------------------------------------------------------------------------
# select_goal
# --------------------------------------------------------------------------

def test_goal_is_in_the_unseen_part_facing_into_it():
    cov = CoverageExplorer(view_range_m=7.0)
    world = make_map(room())
    for yaw in (0.0, math.pi / 2, math.pi, -math.pi / 2):
        cov.observe(world, (3.0, 6.0), yaw)     # everything within 7 m
    result = cov.select_goal(world, (2.5, 6.0), 0)
    assert result.status is ExplorationGoalStatus.READY
    gx, gy = xy(result.goal)
    assert 9.0 < gx < 13.0                      # just past what was seen
    q = result.goal.pose.orientation
    assert abs(2 * math.atan2(q.z, q.w)) < math.radians(60)   # facing further in


def test_everything_seen_means_done():
    cov = CoverageExplorer(view_range_m=40.0, fov_rad=2 * math.pi)
    world = make_map(room())
    cov.observe(world, (15.0, 6.0), 0.0)
    result = cov.select_goal(world, (15.0, 6.0), 0)
    assert result.status is ExplorationGoalStatus.NO_REACHABLE_FRONTIER
    assert 'seen 100%' in result.reason


def test_small_shadow_regions_are_ignored():
    cells = room(rows=20, cols=20)
    world = make_map(cells)
    cov = CoverageExplorer(view_range_m=40.0, fov_rad=2 * math.pi, min_region_m2=2.0)
    cov.observe(world, (3.0, 3.0), 0.0)
    cov._seen[10:12, 10:12] = False             # a 0.36 m2 hole
    result = cov.select_goal(world, (3.0, 3.0), 0)
    assert result.status is ExplorationGoalStatus.NO_REACHABLE_FRONTIER
    assert 'too small' in result.reason


def test_unreachable_unseen_space_is_not_a_goal():
    cells = room()
    cells[:, 50] = 100                          # sealed wall at x = 15 m
    cov = CoverageExplorer(view_range_m=40.0, fov_rad=2 * math.pi)
    world = make_map(cells)
    cov.observe(world, (5.0, 6.0), 0.0)         # sees the near half only
    result = cov.select_goal(world, (5.0, 6.0), 0)
    assert result.status is ExplorationGoalStatus.NO_REACHABLE_FRONTIER


def test_reached_goal_is_not_chosen_again_but_failed_one_cools_down():
    cov = CoverageExplorer(view_range_m=6.0, failure_cooldown_sec=10.0)
    world = make_map(room())
    cov.observe(world, (3.0, 6.0), 0.0)
    cov.observe(world, (3.0, 6.0), math.pi)
    first = cov.select_goal(world, (2.5, 6.0), 0).goal
    cov.report_outcome(first, NavigationStatus.ABORTED, '', 0)
    second = cov.select_goal(world, (2.5, 6.0), SECOND).goal
    assert second is None or math.dist(xy(first), xy(second)) > 1.5
    later = cov.select_goal(world, (2.5, 6.0), 20 * SECOND).goal
    assert math.dist(xy(first), xy(later)) < 0.01


def test_coverage_grid_marks_unseen_free_cells():
    cov = CoverageExplorer(view_range_m=6.0)
    world = make_map(room())
    cov.observe(world, (5.0, 6.0), 0.0)
    data = np.array(cov.coverage_grid(world).data).reshape(40, 100)
    assert data[int(6.0 / RES), int(9.0 / RES)] == 0
    assert data[int(6.0 / RES), int(20.0 / RES)] == 100
    assert data[0, 0] == 0                      # walls are never "unseen free"


# --------------------------------------------------------------------------
# TwoPhaseExplorer
# --------------------------------------------------------------------------

class ScriptedFrontier:
    def __init__(self, status, cost=0.0):
        self.status = status
        self.frontiers, self.chosen, self.frame_id = ('f',), None, 'map'
        self.cost = cost
        self.reports = []

    def select_goal(self, map_msg, robot_xy, now_ns):
        if self.status is ExplorationGoalStatus.READY:
            from geometry_msgs.msg import PoseStamped
            from types import SimpleNamespace
            self.chosen = SimpleNamespace(cost=self.cost)
            return ExplorationGoal(self.status, PoseStamped(), 'frontier')
        self.chosen = None
        return ExplorationGoal(self.status, None, 'scripted')

    def report_outcome(self, goal, status, reason, now_ns):
        self.reports.append(status)


def test_frontiers_first_then_coverage_and_outcomes_go_back_to_their_source():
    frontier = ScriptedFrontier(ExplorationGoalStatus.READY)
    coverage = CoverageExplorer(view_range_m=6.0)
    explorer = TwoPhaseExplorer(frontier, coverage)
    world = make_map(room())

    first = explorer.select_goal(world, (2.5, 6.0), 0)
    assert first.reason == 'frontier' and explorer.frontiers == ('f',)

    frontier.status = ExplorationGoalStatus.NO_REACHABLE_FRONTIER
    second = explorer.select_goal(world, (2.5, 6.0), 0)
    assert second.reason.startswith('coverage') and explorer.active is coverage

    explorer.report_outcome(first.goal, NavigationStatus.SUCCEEDED, '', 0)
    explorer.report_outcome(second.goal, NavigationStatus.ABORTED, '', 0)
    assert frontier.reports == [NavigationStatus.SUCCEEDED]
    assert len(coverage._failures) == 1


def test_not_ready_frontier_does_not_fall_through_to_coverage():
    frontier = ScriptedFrontier(ExplorationGoalStatus.NOT_READY)
    explorer = TwoPhaseExplorer(frontier, CoverageExplorer())
    result = explorer.select_goal(make_map(room()), (2.5, 6.0), 0)
    assert result.status is ExplorationGoalStatus.NOT_READY
    assert explorer.active is frontier


def test_interleaved_picks_the_cheaper_of_frontier_and_unseen_region():
    world = make_map(room())
    coverage = CoverageExplorer(view_range_m=6.0)
    near = coverage.select_goal(world, (2.5, 6.0), 0)       # unseen room, cost ~ -4
    assert near.status is ExplorationGoalStatus.READY and coverage.chosen.cost < 0

    far = ScriptedFrontier(ExplorationGoalStatus.READY, cost=30.0)
    explorer = TwoPhaseExplorer(far, CoverageExplorer(view_range_m=6.0), interleave=True)
    result = explorer.select_goal(world, (2.5, 6.0), 0)
    assert result.reason.startswith('coverage') and explorer.active is explorer.coverage
    # Replanning checks the goal being driven to against both lists.
    assert explorer.frontiers[0] == 'f' and len(explorer.frontiers) > 1
    explorer.report_outcome(result.goal, NavigationStatus.ABORTED, '', 0)
    assert len(explorer.coverage._failures) == 1 and far.reports == []

    close = ScriptedFrontier(ExplorationGoalStatus.READY, cost=-30.0)
    explorer = TwoPhaseExplorer(close, CoverageExplorer(view_range_m=6.0), interleave=True)
    assert explorer.select_goal(world, (2.5, 6.0), 0).reason == 'frontier'


def test_not_interleaved_ignores_unseen_regions_while_frontiers_remain():
    far = ScriptedFrontier(ExplorationGoalStatus.READY, cost=30.0)
    explorer = TwoPhaseExplorer(far, CoverageExplorer(view_range_m=6.0))
    assert explorer.select_goal(make_map(room()), (2.5, 6.0), 0).reason == 'frontier'
    assert explorer.frontiers == ('f',)


def test_region_gain_is_capped_at_max_gain_m2():
    """A huge unseen region must not outbid a nearby small one by tens of metres."""
    view_m2 = 0.5 * 2.0944 * 6.0 ** 2
    cov = CoverageExplorer(view_range_m=6.0, gain_per_m2=0.05, max_gain_m2=view_m2)
    world = make_map(room(rows=80, cols=200))                 # 60 m x 24 m, all unseen
    cov.select_goal(world, (2.5, 6.0), 0)
    assert cov.chosen.length_m > 10 * view_m2
    assert cov.chosen.cost == pytest.approx(cov.chosen.path_m - 0.05 * view_m2)


@pytest.mark.parametrize('robot', [(2.5, 6.0), (27.0, 3.0)])
def test_coverage_converges_on_a_room_with_pillars(robot):
    """Driving to each goal in turn ends with the room seen, in a bounded number of goals."""
    cells = room(rows=60, cols=120)
    cells[15:25, 30:40] = 100
    cells[35:45, 70:80] = 100
    world = make_map(cells)
    cov = CoverageExplorer(view_range_m=6.0)
    x, y = robot
    for step in range(60):
        result = cov.select_goal(world, (x, y), step * 30 * SECOND)
        if result.status is not ExplorationGoalStatus.READY:
            break
        gx, gy = xy(result.goal)
        heading = math.atan2(gy - y, gx - x)
        for i in range(1, 11):
            cov.observe(world, (x + (gx - x) * i / 10, y + (gy - y) * i / 10), heading)
        q = result.goal.pose.orientation
        cov.observe(world, (gx, gy), 2 * math.atan2(q.z, q.w))
        cov.report_outcome(result.goal, NavigationStatus.SUCCEEDED, '', step * 30 * SECOND)
        x, y = gx, gy
    assert result.status is ExplorationGoalStatus.NO_REACHABLE_FRONTIER
    assert cov.seen_fraction(world) > 0.95
    assert step < 40


def test_unseen_cells_behind_a_reached_goal_do_not_end_coverage():
    """Regression: the region's nearest cell was the goal just reached, so the
    whole region was skipped as visited and coverage stopped at 10 %."""
    cov = CoverageExplorer(view_range_m=6.0)
    world = make_map(room())
    first = cov.select_goal(world, (2.5, 6.0), 0)
    cov.observe(world, (2.5, 6.0), 0.0)
    cov.report_outcome(first.goal, NavigationStatus.SUCCEEDED, '', 0)
    second = cov.select_goal(world, (2.5, 6.0), SECOND)
    assert second.status is ExplorationGoalStatus.READY
    assert math.dist(xy(second.goal), (2.5, 6.0)) > 1.5
