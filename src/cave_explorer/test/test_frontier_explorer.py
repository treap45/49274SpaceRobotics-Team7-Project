"""Tests for frontier_explorer on hand-drawn occupancy grids.

Maps are written as ASCII with the top row at the highest y, the way RViz shows
them: ``#`` occupied, ``.`` free, ``?`` unknown, ``R`` the robot on a free
cell. Cells are 0.3 m, as in slam_params.yaml, so a goal needs to be at least
three cells from a wall to clear the default 0.7 m. Every expected answer can be
worked out from the drawing.

No ROS runtime is involved - only message types - so these run in well under a
second.
"""

import math
import random
import time

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
import pytest

from cave_explorer.frontier_explorer import FrontierExplorer, frontier_markers
from cave_explorer.interfaces import ExplorationGoalStatus, NavigationStatus

RES = 0.3
SECOND = 10**9
CELL_VALUE = {'#': 100, '.': 0, '?': -1, 'R': 0}


def ascii_map(rows, origin=(0.0, 0.0)):
    """Build an OccupancyGrid and the robot position from an ASCII drawing."""
    height, width = len(rows), len(rows[0])
    assert all(len(row) == width for row in rows), 'ragged drawing'
    data, robot = [], None
    for y_index in range(height):
        line = rows[height - 1 - y_index]
        for x_index, symbol in enumerate(line):
            data.append(CELL_VALUE[symbol])
            if symbol == 'R':
                robot = cell_centre(x_index, y_index, origin)
    grid = OccupancyGrid()
    grid.header.frame_id = 'map'
    grid.info.resolution = RES
    grid.info.width, grid.info.height = width, height
    grid.info.origin.position.x, grid.info.origin.position.y = origin
    grid.info.origin.orientation.w = 1.0
    grid.data = data
    return grid, robot


def cell_centre(x_index, y_index, origin=(0.0, 0.0)):
    return (origin[0] + (x_index + 0.5) * RES, origin[1] + (y_index + 0.5) * RES)


def goal_xy(result):
    return (result.goal.pose.position.x, result.goal.pose.position.y)


def pose_at(xy):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.pose.position.x, pose.pose.position.y = xy
    return pose


# A corridor whose top opens into unknown space. The only clear frontier cell is
# the middle one: the cells beside it are 1 or 2 cells (0.3 / 0.6 m) from a wall.
DEAD_END = [
    '???????',
    '???????',
    '#.....#',
    '#.....#',
    '#.....#',
    '#..R..#',
    '#######',
]

# A corridor open at both ends; the robot stands near the left one.
TWO_EXITS = [
    '#########################',
    '?.......................?',
    '?.......................?',
    '?..R....................?',
    '?.......................?',
    '?.......................?',
    '#########################',
]
LEFT_EXIT = cell_centre(1, 3)
RIGHT_EXIT = cell_centre(23, 3)


@pytest.fixture
def explorer():
    return FrontierExplorer()


# ----------------------------------------------------------------------
# Status: no map, nothing left, a usable goal
# ----------------------------------------------------------------------

def test_no_map_is_not_ready(explorer):
    result = explorer.select_goal(None, (0.0, 0.0), 0)
    assert result.status is ExplorationGoalStatus.NOT_READY


def test_map_without_free_space_is_not_ready(explorer):
    grid, _ = ascii_map(['?????', '?????', '?????'])
    result = explorer.select_goal(grid, (0.5, 0.5), 0)
    assert result.status is ExplorationGoalStatus.NOT_READY
    assert 'no free cells' in result.reason


def test_robot_off_the_known_map_is_not_ready(explorer):
    grid, _ = ascii_map(DEAD_END)
    result = explorer.select_goal(grid, (50.0, 50.0), 0)
    assert result.status is ExplorationGoalStatus.NOT_READY
    assert 'robot' in result.reason


def test_enclosed_room_has_no_frontier(explorer):
    grid, robot = ascii_map([
        '#######',
        '#.....#',
        '#..R..#',
        '#.....#',
        '#######',
    ])
    result = explorer.select_goal(grid, robot, 0)
    assert result.status is ExplorationGoalStatus.NO_REACHABLE_FRONTIER
    assert explorer.frontiers == ()


def test_dead_end_goal_is_the_clear_middle_of_the_opening(explorer):
    grid, robot = ascii_map(DEAD_END)
    result = explorer.select_goal(grid, robot, 0)
    assert result.status is ExplorationGoalStatus.READY
    assert goal_xy(result) == pytest.approx(cell_centre(3, 4))
    assert result.goal.header.frame_id == 'map'
    assert explorer.chosen.path_m == pytest.approx(3 * RES)


def test_goal_faces_the_direction_of_travel(explorer):
    grid, robot = ascii_map(DEAD_END)
    result = explorer.select_goal(grid, robot, 0)
    orientation = result.goal.pose.orientation
    yaw = 2.0 * math.atan2(orientation.z, orientation.w)
    assert yaw == pytest.approx(math.pi / 2)    # the opening is straight up
    assert orientation.z ** 2 + orientation.w ** 2 == pytest.approx(1.0)


# ----------------------------------------------------------------------
# What makes a frontier unusable
# ----------------------------------------------------------------------

def test_gap_narrower_than_the_robot_is_ignored(explorer):
    grid, robot = ascii_map([
        '???????????',
        '####..#####',
        '#.........#',
        '#.........#',
        '#....R....#',
        '#.........#',
        '###########',
    ])
    result = explorer.select_goal(grid, robot, 0)
    assert result.status is ExplorationGoalStatus.NO_REACHABLE_FRONTIER
    assert [f.rejection for f in explorer.frontiers] == ['too_narrow']
    assert 'too narrow' in result.reason


def test_frontier_behind_a_wall_is_unreachable(explorer):
    grid, robot = ascii_map([
        '###########',
        '#.....#...?',
        '#.....#...?',
        '#..R..#...?',
        '#.....#...?',
        '#.....#...?',
        '###########',
    ])
    result = explorer.select_goal(grid, robot, 0)
    assert result.status is ExplorationGoalStatus.NO_REACHABLE_FRONTIER
    assert [f.rejection for f in explorer.frontiers] == ['unreachable']


def test_opening_hugging_a_wall_has_no_clear_goal(explorer):
    # Four cells wide: every frontier cell is within 0.6 m of a wall.
    grid, robot = ascii_map([
        '??????',
        '#....#',
        '#....#',
        '#.R..#',
        '######',
    ])
    result = explorer.select_goal(grid, robot, 0)
    assert result.status is ExplorationGoalStatus.NO_REACHABLE_FRONTIER
    assert [f.rejection for f in explorer.frontiers] == ['no_clearance']


# ----------------------------------------------------------------------
# Choosing between frontiers, and learning from outcomes
# ----------------------------------------------------------------------

def test_nearer_exit_wins_and_both_are_listed(explorer):
    grid, robot = ascii_map(TWO_EXITS)
    result = explorer.select_goal(grid, robot, 0)
    assert goal_xy(result) == pytest.approx(LEFT_EXIT)
    goals = sorted(f.goal for f in explorer.frontiers)
    assert goals == [pytest.approx(LEFT_EXIT), pytest.approx(RIGHT_EXIT)]


def test_path_cost_follows_walls_not_straight_lines(explorer):
    # The opening at the top left is 1.6 m away as the crow flies, but the
    # dividing wall forces a 23-cell detour along the bottom. The right-hand
    # frontier is 15 cells straight ahead and should win.
    rows = ['#?????' + '#' * 19]
    for y in range(11, 1, -1):
        rows.append('#.....#' + ('.R' + '.' * 15 if y == 9 else '.' * 17) + '?')
    rows += ['#' + '.' * 23 + '?', '#' * 25]
    grid, robot = ascii_map(rows)

    result = explorer.select_goal(grid, robot, 0)
    assert goal_xy(result) == pytest.approx(cell_centre(23, 9))
    top_left = next(f for f in explorer.frontiers if f.goal[0] < 2.0)
    assert top_left.goal == pytest.approx(cell_centre(3, 11))
    assert math.dist(top_left.goal, robot) < 2.0
    assert top_left.path_m == pytest.approx(23 * RES)


def test_failed_goal_is_avoided_until_its_cooldown_ends(explorer):
    grid, robot = ascii_map(TWO_EXITS)
    first = explorer.select_goal(grid, robot, 0)
    explorer.report_outcome(first.goal, NavigationStatus.ABORTED, 'aborted', 0)

    retry = explorer.select_goal(grid, robot, 1 * SECOND)
    assert goal_xy(retry) == pytest.approx(RIGHT_EXIT)
    assert sorted(f.rejection for f in explorer.frontiers) == ['', 'failed_recently']

    later = explorer.select_goal(grid, robot, 91 * SECOND)
    assert goal_xy(later) == pytest.approx(LEFT_EXIT)


@pytest.mark.parametrize('status', [NavigationStatus.REJECTED, NavigationStatus.ERROR])
def test_rejection_and_timeout_count_as_failures(explorer, status):
    grid, robot = ascii_map(TWO_EXITS)
    first = explorer.select_goal(grid, robot, 0)
    explorer.report_outcome(first.goal, status, '', 0)
    assert goal_xy(explorer.select_goal(grid, robot, 0)) == pytest.approx(RIGHT_EXIT)


def test_cancelled_goal_is_not_penalised(explorer):
    """navigation_client reports our own preemptions as CANCELED."""
    grid, robot = ascii_map(TWO_EXITS)
    first = explorer.select_goal(grid, robot, 0)
    explorer.report_outcome(first.goal, NavigationStatus.CANCELED, 'superseded', 0)
    assert goal_xy(explorer.select_goal(grid, robot, 0)) == pytest.approx(LEFT_EXIT)


def test_recently_visited_goal_is_penalised_and_the_penalty_fades(explorer):
    # A frontier can survive a visit, e.g. when the unknown beyond it cannot be seen.
    grid, robot = ascii_map(TWO_EXITS)
    explorer.report_outcome(pose_at(LEFT_EXIT), NavigationStatus.SUCCEEDED, '', 0)
    assert goal_xy(explorer.select_goal(grid, robot, 0)) == pytest.approx(RIGHT_EXIT)
    assert goal_xy(explorer.select_goal(grid, robot, 121 * SECOND)) == pytest.approx(LEFT_EXIT)


# ----------------------------------------------------------------------
# Geometry details
# ----------------------------------------------------------------------

def test_robot_on_a_non_free_cell_still_finds_its_way(explorer):
    grid, _ = ascii_map(DEAD_END)
    on_the_wall = cell_centre(3, 0)
    result = explorer.select_goal(grid, on_the_wall, 0)
    assert result.status is ExplorationGoalStatus.READY


def test_map_origin_offset_moves_the_goal(explorer):
    origin = (-10.0, 5.0)
    grid, robot = ascii_map(DEAD_END, origin=origin)
    result = explorer.select_goal(grid, robot, 0)
    assert goal_xy(result) == pytest.approx(cell_centre(3, 4, origin))


def test_markers_show_candidates_rejections_and_the_choice(explorer):
    grid, robot = ascii_map([
        '???????????????????',
        '####..#############',
        '#.................?',
        '#.................?',
        '#....R............?',
        '#.................?',
        '#.................?',
        '###################',
    ])
    explorer.select_goal(grid, robot, 0)
    markers = {m.ns: m for m in frontier_markers(explorer).markers}
    assert markers[''].action == markers[''].DELETEALL
    assert len(markers['frontier_cells'].points) == 5
    assert len(markers['frontier_rejected'].points) == 2
    assert len(markers['frontier_goals'].points) == 1
    assert len(markers['frontier_chosen'].points) == 1


def test_full_size_map_is_fast(explorer):
    """A 60 x 60 m cave at 0.3 m is 40 000 cells; selection must not stall the main loop."""
    rng = random.Random(7)
    size = 200
    rows = []
    for y in range(size):
        row = []
        for x in range(size):
            if x in (0, size - 1) or y in (0, size - 1):
                row.append('#')
            elif x > 150:
                row.append('?')
            else:
                row.append('#' if rng.random() < 0.05 else '.')
        rows.append(''.join(row))
    rows[100] = rows[100][:100] + 'R' + rows[100][101:]
    grid, robot = ascii_map(rows)

    started = time.perf_counter()
    explorer.select_goal(grid, robot, 0)
    assert time.perf_counter() - started < 0.5
