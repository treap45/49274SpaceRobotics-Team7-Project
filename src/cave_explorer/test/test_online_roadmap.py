"""Hand-constructed maps for the ROS-independent online roadmap."""

import math

import numpy as np
import pytest

from cave_explorer.online_roadmap import (MapSnapshot, RoadmapBuilder, attach, failure_reason,
                                          path_length, plan_path, shortcut)

RES = 0.3


def grid_with(*boxes, shape=(120, 160), fill=100):
    """Grid with free boxes (x0, y0, x1, y1) in metres; origin at (0, 0)."""
    grid = np.full(shape, fill, dtype=np.int16)
    for x0, y0, x1, y1 in boxes:
        grid[int(y0 / RES):int(y1 / RES), int(x0 / RES):int(x1 / RES)] = 0
    return grid


def snapshot(*boxes, **kwargs):
    return MapSnapshot(grid_with(*boxes, **kwargs), RES)


# A 4 m wide corridor along y = 6 with a 6 m wide room at its east end.
CORRIDOR = (1.0, 4.0, 30.0, 8.0)
ROOM = (30.0, 1.0, 42.0, 13.0)


def drive(builder, snap, start, end, step=0.25, t0=0.0):
    start, end = np.asarray(start, float), np.asarray(end, float)
    count = int(math.dist(start, end) / step)
    for k in range(count + 1):
        builder.update(snap, start + (end - start) * k / count, t0 + 0.1 * k)


def all_edges_valid(builder, snap):
    roadmap = builder.roadmap
    return all(snap.segments_clear(roadmap.xy(a), [roadmap.xy(b)])[0] for a, b in roadmap.edge_list())


def test_clearance_and_traversability():
    snap = snapshot(CORRIDOR)
    assert snap.is_traversable(np.array([10.0, 6.0]))
    # 0.3 m from the wall: free but too close for a 0.5 m robot half-width.
    assert snap.is_free(np.array([10.0, 4.3])) and not snap.is_traversable(np.array([10.0, 4.3]))
    assert snap.clearance.max() == pytest.approx(2.0, abs=RES)


def test_unknown_space_blocks_edges():
    grid = grid_with(CORRIDOR)
    grid[int(4 / RES):int(8 / RES), int(15 / RES):int(16 / RES)] = -1
    snap = MapSnapshot(grid, RES)
    assert not snap.segments_clear((10.0, 6.0), [(20.0, 6.0)])[0]
    assert snap.segments_clear((5.0, 6.0), [(12.0, 6.0)])[0]


def test_medial_axis_is_the_corridor_centre():
    snap = snapshot(CORRIDOR)
    rows, cols = np.nonzero(snap.medial_axis(0.8, 2.0))
    x, y = (cols + 0.5) * RES, (rows + 0.5) * RES
    # Away from the ends, where the axis branches into the corners.
    middle = (x > 5.0) & (x < 26.0)
    assert middle.sum() > 50 and np.all(np.abs(y[middle] - 6.0) <= RES)


def test_trail_nodes_follow_the_robot():
    snap = snapshot(CORRIDOR)
    builder = RoadmapBuilder('trail', trail_spacing_m=2.0)
    drive(builder, snap, (3.0, 6.0), (25.0, 6.0))
    roadmap = builder.roadmap
    xs = sorted(node.x for node in roadmap.nodes.values())
    assert all(node.kind == 'trail' for node in roadmap.nodes.values())
    assert len(xs) == 12 and all(1.99 <= b - a <= 2.3 for a, b in zip(xs, xs[1:]))
    assert len(set(roadmap.components().values())) == 1
    assert all_edges_valid(builder, snap)


def test_observed_nodes_cover_unvisited_space():
    snap = snapshot(CORRIDOR, ROOM)
    builder = RoadmapBuilder('observed')
    # The robot never leaves the first metres of the corridor.
    drive(builder, snap, (3.0, 6.0), (5.0, 6.0))
    nodes = list(builder.roadmap.nodes.values())
    assert any(n.kind == 'observed' and n.x > 30 for n in nodes), 'room has no nodes'
    assert len(set(builder.roadmap.components().values())) == 1
    assert all_edges_valid(builder, snap)
    path = plan_path(builder.roadmap, snap, (3.0, 6.0), (38.0, 10.0))
    assert path is not None and path_length(path) < 1.2 * math.dist((3, 6), (38, 10)) + 2


def test_spacing_adapts_to_clearance():
    builder = RoadmapBuilder('observed', spacing_per_clearance=1.5, min_spacing_m=1.5,
                             max_spacing_m=6.0)
    assert builder.spacing(0.5) == 1.5
    assert builder.spacing(2.0) == pytest.approx(3.0)
    assert builder.spacing(10.0) == 6.0


def test_edges_never_cross_a_wall():
    # Two parallel corridors separated by a 1.2 m wall, joined at the east end.
    snap = snapshot((1.0, 2.0, 25.0, 6.0), (1.0, 7.2, 25.0, 11.2), (25.0, 2.0, 29.0, 11.2))
    builder = RoadmapBuilder('observed')
    drive(builder, snap, (3.0, 4.0), (4.0, 4.0))
    assert all_edges_valid(builder, snap)
    roadmap = builder.roadmap
    for a, b in roadmap.edge_list():
        (xa, ya), (xb, yb) = roadmap.xy(a), roadmap.xy(b)
        crosses = (ya - 6.6) * (yb - 6.6) < 0 and min(xa, xb) < 24.0
        assert not crosses
    path = plan_path(roadmap, snap, (3.0, 4.0), (3.0, 9.2))
    assert path is not None and path_length(path) > 40


def test_map_change_invalidates_and_reconnects():
    open_grid = grid_with(CORRIDOR)
    builder = RoadmapBuilder('observed')
    drive(builder, MapSnapshot(open_grid, RES), (3.0, 6.0), (4.0, 6.0))
    before = len(builder.roadmap.edge_list())
    # A rock fall blocks the corridor around x = 15 m.
    blocked = open_grid.copy()
    blocked[:, int(14.5 / RES):int(15.5 / RES)] = 100
    snap = MapSnapshot(blocked, RES)
    stats = builder.update(snap, (4.0, 6.0), 100.0)
    assert stats.removed_edges > 0 and len(builder.roadmap.edge_list()) < before
    assert all_edges_valid(builder, snap)
    assert plan_path(builder.roadmap, snap, (3.0, 6.0), (25.0, 6.0)) is None
    # The rock is cleared again: the two halves join on the next map.
    builder.update(MapSnapshot(open_grid, RES), (4.0, 6.0), 101.0)
    assert len(set(builder.roadmap.components().values())) == 1


def test_attach_and_shortcut():
    snap = snapshot(CORRIDOR)
    builder = RoadmapBuilder('observed')
    drive(builder, snap, (3.0, 6.0), (4.0, 6.0))
    assert attach(builder.roadmap, snap, (20.0, 6.0), 4.0)
    assert not attach(builder.roadmap, snap, (20.0, 6.0), 0.01)
    path = plan_path(builder.roadmap, snap, (2.0, 6.0), (28.0, 6.0))
    smooth = shortcut(path, snap)
    assert len(smooth) == 2 and path_length(smooth) == pytest.approx(26.0)


def test_invalid_mode():
    with pytest.raises(ValueError):
        RoadmapBuilder('random')


def test_failure_reasons():
    snap = snapshot(CORRIDOR)
    builder = RoadmapBuilder('trail')
    drive(builder, snap, (3.0, 6.0), (9.0, 6.0))
    roadmap = builder.roadmap
    assert failure_reason(roadmap, snap, (3, 6), (20, 20)) == 'goal is not known free space'
    assert 'closer to a wall' in failure_reason(roadmap, snap, (3, 6), (20, 4.2))
    assert 'in reach of the goal' in failure_reason(roadmap, snap, (3, 6), (25, 6))
