"""Tests for information-gain next-best-view and tour exploration.

Maps are synthetic OccupancyGrids with 0.3 m cells, the simulation's SLAM
resolution. "Seen" is set directly on the shared CoverageExplorer.
"""

import itertools
import math
import time

from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest

from cave_explorer.coverage_explorer import CoverageExplorer
from cave_explorer.frontier_explorer import frontier_markers
from cave_explorer.interfaces import ExplorationGoalStatus, NavigationStatus
from cave_explorer.nbv_explorer import (
    InformationGainExplorer,
    InformationTourExplorer,
    _bisect,
    _open_tour,
)

RES = 0.3
SECOND = 10**9


def make_map(cells):
    """``cells`` is a (rows, cols) array; row 0 is the lowest y."""
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.resolution = RES
    msg.info.height, msg.info.width = cells.shape
    msg.info.origin.orientation.w = 1.0
    msg.data = [int(v) for v in cells.ravel()]
    return msg


def room(rows=40, cols=100):
    """Free room 30 m x 12 m with walls round the edge."""
    cells = np.zeros((rows, cols), dtype=np.int16)
    cells[0, :] = cells[-1, :] = cells[:, 0] = cells[:, -1] = 100
    return cells


def explorer_on(cells, seen_cols=None, **kwargs):
    """Explorer whose camera has seen every free cell in columns ``seen_cols``."""
    world = make_map(cells)
    cov = CoverageExplorer(view_range_m=6.0)
    cov.seen_fraction(world)                    # aligns the seen grid
    if seen_cols is not None:
        cov._seen[:, seen_cols] = True
    return InformationGainExplorer(cov, **kwargs), world


def xy(goal):
    return goal.pose.position.x, goal.pose.position.y


def yaw(goal):
    q = goal.pose.orientation
    return 2 * math.atan2(q.z, q.w)


def test_goal_looks_into_the_unseen_end_of_the_room():
    nbv, world = explorer_on(room(), seen_cols=slice(0, 50))   # x < 15 m seen
    result = nbv.select_goal(world, (3.0, 6.0), 0)
    assert result.status is ExplorationGoalStatus.READY
    gx, _ = xy(result.goal)
    assert 9.0 < gx < 22.0                      # near the unseen half, not inside the seen one
    assert abs(yaw(result.goal)) < math.radians(35)     # facing +x, into it
    assert 'information gain' in result.reason


def test_everything_seen_and_mapped_means_done():
    nbv, world = explorer_on(room(), seen_cols=slice(None))
    result = nbv.select_goal(world, (3.0, 6.0), 0)
    assert result.status is ExplorationGoalStatus.NO_REACHABLE_FRONTIER
    assert 'seen 100%' in result.reason


def test_nearer_of_two_equal_unseen_ends_wins():
    nbv, world = explorer_on(room(), seen_cols=slice(25, 75))  # both ends unseen
    gx, _ = xy(nbv.select_goal(world, (9.0, 6.0), 0).goal)
    assert gx < 15.0
    nbv, world = explorer_on(room(), seen_cols=slice(25, 75))
    gx, _ = xy(nbv.select_goal(world, (21.0, 6.0), 0).goal)
    assert gx > 15.0


def test_unknown_space_beyond_an_opening_draws_the_robot_there():
    cells = room()
    cells[10:30, -1] = -1                       # 6 m opening in the east wall
    cells = np.hstack([cells, np.full((40, 30), -1, dtype=np.int16)])
    nbv, world = explorer_on(cells, seen_cols=slice(None))     # every known cell seen
    result = nbv.select_goal(world, (3.0, 6.0), 0)
    assert result.status is ExplorationGoalStatus.READY
    assert xy(result.goal)[0] > 20.0


def test_unknown_behind_a_one_cell_gap_is_worth_nothing():
    cells = room()
    cells[20, -1] = -1                          # 0.3 m crack in the east wall
    cells = np.hstack([cells, np.full((40, 30), -1, dtype=np.int16)])
    nbv, world = explorer_on(cells, seen_cols=slice(None))
    result = nbv.select_goal(world, (3.0, 6.0), 0)
    assert result.status is ExplorationGoalStatus.NO_REACHABLE_FRONTIER


def test_small_unseen_patch_is_not_worth_a_trip():
    nbv, world = explorer_on(room(), seen_cols=slice(None))
    nbv.coverage._seen[10:13, 60:63] = False    # 0.8 m2 shadow
    result = nbv.select_goal(world, (3.0, 6.0), 0)
    assert result.status is ExplorationGoalStatus.NO_REACHABLE_FRONTIER


def test_reached_viewpoint_is_not_chosen_again_until_its_cooldown_ends():
    nbv, world = explorer_on(room(), seen_cols=slice(0, 50), visit_cooldown_sec=10.0)
    first = nbv.select_goal(world, (3.0, 6.0), 0).goal
    nbv.report_outcome(first, NavigationStatus.SUCCEEDED, '', 0)
    second = nbv.select_goal(world, (3.0, 6.0), SECOND).goal
    assert math.dist(xy(first), xy(second)) > 1.5
    later = nbv.select_goal(world, (3.0, 6.0), 20 * SECOND).goal
    assert math.dist(xy(first), xy(later)) < 0.01


def test_previous_goal_is_still_listed_while_it_pays():
    """BehaviourManager keeps driving to a goal only while the explorer lists it."""
    nbv, world = explorer_on(room(), seen_cols=slice(0, 50))
    first = xy(nbv.select_goal(world, (3.0, 6.0), 0).goal)
    nbv.select_goal(world, (6.0, 6.0), SECOND)
    assert any(math.dist(f.goal, first) < 0.01 for f in nbv.frontiers)


def test_drives_a_cave_to_completion_in_bounded_goals():
    """Driving to each goal in turn ends with the room seen."""
    cells = room(rows=60, cols=120)
    cells[15:25, 30:40] = 100
    cells[35:45, 70:80] = 100
    world = make_map(cells)
    cov = CoverageExplorer(view_range_m=6.0)
    nbv = InformationGainExplorer(cov)
    x, y = 2.5, 6.0
    for step in range(60):
        result = nbv.select_goal(world, (x, y), step * 30 * SECOND)
        if result.status is not ExplorationGoalStatus.READY:
            break
        gx, gy = xy(result.goal)
        heading = math.atan2(gy - y, gx - x)
        for i in range(1, 11):
            cov.observe(world, (x + (gx - x) * i / 10, y + (gy - y) * i / 10), heading)
        cov.observe(world, (gx, gy), yaw(result.goal))
        nbv.report_outcome(result.goal, NavigationStatus.SUCCEEDED, '', step * 30 * SECOND)
        x, y = gx, gy
    assert result.status is ExplorationGoalStatus.NO_REACHABLE_FRONTIER
    assert cov.seen_fraction(world) > 0.95
    assert step < 30


def test_full_size_map_is_fast():
    """The cave map is about 216 x 171 cells; the node plans in a 0.2 s loop."""
    rng = np.random.default_rng(0)
    cells = np.zeros((171, 216), dtype=np.int16)
    cells[0, :] = cells[-1, :] = cells[:, 0] = cells[:, -1] = 100
    for _ in range(25):
        r, c = rng.integers(10, 160), rng.integers(10, 205)
        cells[r:r + 8, c:c + 8] = 100
    cells[:, 150:] = -1
    world = make_map(cells)
    cov = CoverageExplorer()
    nbv = InformationGainExplorer(cov)
    cov.observe(world, (5.0, 5.0), 0.0)
    start = time.perf_counter()
    result = nbv.select_goal(world, (5.0, 5.0), 0)
    assert result.status is ExplorationGoalStatus.READY
    assert time.perf_counter() - start < 0.3


# --------------------------------------------------------------------------
# InformationTourExplorer
# --------------------------------------------------------------------------

def tour_on(cells, seen_cols=None, **kwargs):
    nbv, world = explorer_on(cells, seen_cols)
    return InformationTourExplorer(nbv.coverage, **kwargs), world


def _random_tours(n, count, seed):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        points = rng.uniform(0, 30, size=(n, 2))
        robot = rng.uniform(0, 30, size=2)
        start = np.hypot(*(points - robot).T)
        between = np.hypot(*(points[:, None] - points[None, :]).transpose(2, 0, 1))

        def length(o, start=start, between=between):
            return start[o[0]] + sum(between[a, b] for a, b in zip(o, o[1:]))
        best = min(length(p) for p in itertools.permutations(range(n)))
        yield start, between, length, best


def test_open_tour_is_exact_for_few_clusters():
    for start, between, length, best in _random_tours(6, 20, seed=1):
        order = _open_tour(start, between)
        assert sorted(order.tolist()) == list(range(6))
        assert length(order) == pytest.approx(best)


def test_open_tour_heuristic_is_near_shortest():
    for start, between, length, best in _random_tours(7, 20, seed=2):
        order = _open_tour(start, between, exact_up_to=0)
        assert sorted(order.tolist()) == list(range(7))
        assert length(order) <= 1.05 * best


def test_bisect_cuts_a_long_strip_into_pieces_within_radius():
    width = 200
    strip = np.arange(150) + 5 * width             # 150 cells in a row
    pieces = _bisect(strip, width, 20.0)
    assert sum(p.size for p in pieces) == 150 and len(pieces) >= 4
    for p in pieces:
        cols = p % width
        assert np.abs(cols - cols.mean()).max() <= 20.0


def test_tour_clears_the_pocket_behind_before_the_far_end():
    """Greedy next-best-view goes for the big view ahead and comes back for the
    pocket behind; the tour covers the pocket on the way."""
    cells = room(rows=40, cols=120)                 # 36 m x 12 m
    seen = np.zeros(120, dtype=bool)
    seen[12:50] = True                              # 3.6 m - 15 m seen; behind and far end not
    greedy, world = explorer_on(cells, seen_cols=seen)
    tour, _ = tour_on(cells, seen_cols=seen)
    robot = (6.0, 6.0)
    assert xy(greedy.select_goal(world, robot, 0).goal)[0] > robot[0]
    result = tour.select_goal(world, robot, 0)
    assert result.status is ExplorationGoalStatus.READY and 'tour' in result.reason
    assert xy(result.goal)[0] < robot[0]
    assert tour.tour[0] == robot and len(tour.tour) == len(tour.frontiers) + 1
    assert xy(result.goal) == tour.tour[1]


def test_tour_lists_every_cluster_viewpoint_and_draws_the_route():
    tour, world = tour_on(room(rows=40, cols=120), seen_cols=slice(12, 50))
    tour.select_goal(world, (6.0, 6.0), 0)
    assert len(tour.frontiers) >= 2
    assert all(f.goal is not None and not f.rejection for f in tour.frontiers)
    namespaces = [m.ns for m in frontier_markers(tour).markers]
    assert 'exploration_tour' in namespaces and 'frontier_chosen' in namespaces


def test_tour_done_when_everything_seen():
    tour, world = tour_on(room(), seen_cols=slice(None))
    assert tour.select_goal(world, (3.0, 6.0), 0).status is ExplorationGoalStatus.NO_REACHABLE_FRONTIER


def test_tour_drives_a_cave_to_completion_in_bounded_goals():
    cells = room(rows=60, cols=120)
    cells[15:25, 30:40] = 100
    cells[35:45, 70:80] = 100
    world = make_map(cells)
    cov = CoverageExplorer(view_range_m=6.0)
    tour = InformationTourExplorer(cov)
    x, y = 2.5, 6.0
    for step in range(60):
        result = tour.select_goal(world, (x, y), step * 30 * SECOND)
        if result.status is not ExplorationGoalStatus.READY:
            break
        gx, gy = xy(result.goal)
        heading = math.atan2(gy - y, gx - x)
        for i in range(1, 11):
            cov.observe(world, (x + (gx - x) * i / 10, y + (gy - y) * i / 10), heading)
        cov.observe(world, (gx, gy), yaw(result.goal))
        tour.report_outcome(result.goal, NavigationStatus.SUCCEEDED, '', step * 30 * SECOND)
        x, y = gx, gy
    assert result.status is ExplorationGoalStatus.NO_REACHABLE_FRONTIER
    assert cov.seen_fraction(world) > 0.95
    assert step < 40


def test_tour_on_a_full_size_map_is_fast():
    rng = np.random.default_rng(0)
    cells = np.zeros((171, 216), dtype=np.int16)
    cells[0, :] = cells[-1, :] = cells[:, 0] = cells[:, -1] = 100
    for _ in range(25):
        r, c = rng.integers(10, 160), rng.integers(10, 205)
        cells[r:r + 8, c:c + 8] = 100
    cells[:, 150:] = -1
    world = make_map(cells)
    cov = CoverageExplorer()
    tour = InformationTourExplorer(cov)
    cov.observe(world, (5.0, 5.0), 0.0)
    start = time.perf_counter()
    assert tour.select_goal(world, (5.0, 5.0), 0).status is ExplorationGoalStatus.READY
    assert time.perf_counter() - start < 0.3


def test_unknown_still_unknown_after_reaching_its_viewpoint_is_given_up():
    """A pocket the lidar cannot map from its viewpoints must not keep drawing
    the robot back (it did, 68 m, in simulation): each reached viewpoint gives
    up what it had in sight, so the trips end."""
    cells = room()
    cells[10:30, -1] = -1                       # opening in the east wall ...
    cells = np.hstack([cells, np.full((40, 30), -1, dtype=np.int16)])  # ... that never gets mapped
    for planner in (InformationGainExplorer, InformationTourExplorer):
        nbv, world = explorer_on(cells, seen_cols=slice(None))
        explorer = planner(nbv.coverage)
        robot = (3.0, 6.0)
        for trip in range(6):
            result = explorer.select_goal(world, robot, trip * 200 * SECOND)
            if result.status is not ExplorationGoalStatus.READY:
                break
            explorer.report_outcome(result.goal, NavigationStatus.SUCCEEDED, '', trip * 200 * SECOND)
            robot = xy(result.goal)
        assert result.status is ExplorationGoalStatus.NO_REACHABLE_FRONTIER, planner.__name__
        assert 1 <= trip <= 3


def test_unknown_is_not_given_up_before_slam_has_mapped_the_view():
    cells = room()
    cells[10:30, -1] = -1
    cells = np.hstack([cells, np.full((40, 30), -1, dtype=np.int16)])
    nbv, world = explorer_on(cells, seen_cols=slice(None))
    first = nbv.select_goal(world, (3.0, 6.0), 0)
    nbv.report_outcome(first.goal, NavigationStatus.SUCCEEDED, '', 0)
    assert nbv.select_goal(world, xy(first.goal), 5 * SECOND).status is ExplorationGoalStatus.READY
