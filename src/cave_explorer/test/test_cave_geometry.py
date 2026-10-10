"""Hand-constructed maps for the ROS-independent cave geometry computation."""

import math

import numpy as np
import pytest

from cave_explorer.cave_geometry import analyze_geometry, grid_to_world, world_to_grid


@pytest.fixture
def rooms():
    """Two rooms connected by a five-cell (1.5 m) horizontal corridor."""
    grid = np.full((65, 110), 100, dtype=np.int16)
    grid[15:46, 5:36] = 0
    grid[5:56, 65:106] = 0
    grid[28:33, 35:66] = 0
    return grid


def test_corridor_width(rooms):
    result = analyze_geometry(rooms, 0.3, (30, 20))
    assert result.bottleneck_position_rc is not None
    r, c = result.bottleneck_position_rc
    assert 28 <= r < 33 and 36 <= c < 65
    assert abs(result.bottleneck_width_m - 1.5) <= 0.3 + 1e-9
    assert all(rooms[p] >= 50 for p in result.bottleneck_wall_points_rc)
    assert result.saddle_candidates


def test_widest_in_larger_room(rooms):
    result = analyze_geometry(rooms, 0.3, (30, 20))
    r, c = result.widest_position_rc
    assert 5 <= r < 56 and 65 <= c < 106


@pytest.mark.parametrize('end', [100, -1])
def test_dead_end_and_unknown(end, rooms):
    rooms[:, 65:] = end
    result = analyze_geometry(rooms, 0.3, (30, 20))
    assert result.bottleneck_position_rc is None


def test_unknown_walls_are_not_confirmed(rooms):
    rooms[27, 36:65] = -1
    rooms[33, 36:65] = -1
    assert analyze_geometry(rooms, 0.3, (30, 20)).bottleneck_position_rc is None


def test_disconnected(rooms):
    rooms[:, 50] = 100
    result = analyze_geometry(rooms, 0.3, (30, 20))
    assert not result.reachable_mask[:, 65:].any()
    assert result.widest_position_rc[1] < 50
    assert not result.ridge_mask[:, 65:].any()


def test_clearance_disconnects(rooms):
    result = analyze_geometry(rooms, 0.3, (30, 20), robot_radius_m=0.9)
    assert not result.reachable_mask[:, 65:].any()
    assert result.bottleneck_position_rc is None


@pytest.mark.parametrize('robot', [None, (-1, 0), (100, 0), (0, 0), (15, 5)])
def test_invalid_robot(rooms, robot):
    result = analyze_geometry(rooms, 0.3, robot)
    assert not result.reachable_mask.any()
    assert result.widest_position_rc is None
    assert result.reason


def test_unknown():
    result = analyze_geometry(np.full((20, 20), -1), 0.3, (10, 10))
    assert not result.distance_map.any()
    assert result.bottleneck_position_rc is None


def test_all_free_edges():
    result = analyze_geometry(np.zeros((21, 21)), 0.3, (10, 10))
    assert result.distance_map[0, 10] == pytest.approx(0.3)
    assert result.distance_map[-1, 10] == pytest.approx(0.3)
    assert result.widest_radius_m == pytest.approx(3.3)
    assert result.bottleneck_position_rc is None


@pytest.mark.parametrize('shape', [(1, 10), (10, 1), (1, 1)])
def test_small_maps(shape):
    result = analyze_geometry(np.zeros(shape), 0.3, (0, 0), robot_radius_m=0)
    assert result.widest_position_rc is not None


@pytest.mark.parametrize('yaw', [0, math.pi / 2, -0.7])
def test_coordinates(yaw):
    assert world_to_grid(grid_to_world((4, 8), 0.3, (-10, 5), yaw),
                         0.3, (-10, 5), yaw) == (4, 8)
    assert grid_to_world((0, 0), 0.3) == pytest.approx((0.15, 0.15))
    assert world_to_grid((-0.01, -0.01), 0.3) == (-1, -1)


def test_invalid_input():
    with pytest.raises(ValueError):
        analyze_geometry(np.zeros((0, 2)), 0.3, (0, 0))
    with pytest.raises(ValueError):
        analyze_geometry(np.zeros((2, 2)), 0, (0, 0))


def test_markers(rooms):
    pytest.importorskip('rclpy')
    from builtin_interfaces.msg import Time
    from nav_msgs.msg import MapMetaData
    from visualization_msgs.msg import Marker
    from cave_explorer.cave_geometry_node import geometry_markers

    info = MapMetaData(resolution=0.3, width=110, height=65)
    info.origin.orientation.w = 1.0
    result = analyze_geometry(rooms, 0.3, (30, 20))
    markers = geometry_markers(result, info, Time()).markers
    assert markers[0].action == Marker.DELETEALL
    assert all(m.header.frame_id == 'map' for m in markers)
    assert len({(m.ns, m.id) for m in markers}) == len(markers)
    line = next(m for m in markers if m.ns == 'bottleneck')
    assert len(line.points) == 2
    empty = analyze_geometry(rooms, 0.3, None)
    assert not any(m.ns == 'bottleneck' for m in geometry_markers(empty, info, Time()).markers)


def test_even_width_corridor(rooms):
    rooms[33, 35:66] = 0
    result = analyze_geometry(rooms, 0.3, (30, 20))
    assert result.bottleneck_position_rc is not None
    assert abs(result.bottleneck_width_m - 1.8) <= 0.3 + 1e-9


def test_vertical_corridor(rooms):
    grid = np.rot90(rooms)
    result = analyze_geometry(grid, 0.3, (rooms.shape[1] - 1 - 20, 30))
    assert result.bottleneck_position_rc is not None
    assert abs(result.bottleneck_width_m - 1.5) <= 0.3 + 1e-9


def test_occupancy_threshold_and_corner_cutting():
    grid = np.full((9, 9), 50)
    grid[1:4, 1:4] = 49
    grid[4:8, 4:8] = 0
    result = analyze_geometry(grid, 0.3, (2, 2), robot_radius_m=0)
    assert result.reachable_mask[2, 2]
    assert not result.reachable_mask[5, 5]
    assert not result.reachable_mask[0, 0]
