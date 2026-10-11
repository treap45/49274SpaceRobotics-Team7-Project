"""Online roadmap construction from the SLAM occupancy grid (Advanced 4).

The roadmap is an undirected graph of collision-free straight edges, grown as
the robot explores and checked again whenever the map changes.

Validity
--------
A point is *traversable* when its clearance, the distance from the cell
centre to the nearest cell that is not known free minus half a cell, is at
least ``robot_radius_m``. Unknown cells count as obstacles. An edge is valid
when every point sampled along it, a quarter cell apart, is traversable, so
the robot body (half-width 0.5 m) fits along the whole edge and never crosses
an obstacle or unmapped space.

Strategies (``RoadmapBuilder.mode``)
------------------------------------
``trail``
    The baseline: a node at the robot's position whenever it is at least
    ``trail_spacing_m`` from every node, joined to nearby nodes.

``observed``
    The trail plus nodes in observed but unvisited free space, placed on the
    medial axis of the free space (the cave's centre lines). Candidates are taken
    widest first and kept only if no node lies within a spacing that grows
    with the local clearance, so narrow passages get dense nodes and open
    chambers sparse ones. New map areas create new candidates, so the roadmap
    grows ahead of the robot.

Each update also removes nodes and edges that the newest map invalidates
(SLAM corrections, newly seen obstacles) and reconnects what remains.
"""

from dataclasses import dataclass, field
import heapq
import math
import time

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass
class RoadmapNode:
    x: float
    y: float
    kind: str          # 'trail' or 'observed'
    created_s: float


class MapSnapshot:
    """Clearance and traversability of one occupancy grid, queried in metres."""

    def __init__(self, grid, resolution, origin=(0.0, 0.0), yaw=0.0, robot_radius_m=0.5):
        grid = np.asarray(grid)
        if grid.ndim != 2 or not grid.size or not resolution > 0:
            raise ValueError('expected nonempty 2-D grid and positive resolution')
        self.resolution = float(resolution)
        self.origin = (float(origin[0]), float(origin[1]))
        self.cos, self.sin = math.cos(yaw), math.sin(yaw)
        self.shape = grid.shape
        self.free = (grid >= 0) & (grid < 50)
        # Pad so the map edge counts as unknown space.
        padded = np.pad(self.free, 1, constant_values=False)
        edt, nearest = ndimage.distance_transform_edt(padded, return_indices=True)
        edt = edt[1:-1, 1:-1]
        # Nearest non-free cell of every cell (padded coordinates; only differences are used).
        self.nearest_obstacle = nearest[:, 1:-1, 1:-1]
        self.clearance = np.maximum(edt * self.resolution - 0.5 * self.resolution, 0.0)
        self.clearance[~self.free] = 0.0
        self.robot_radius = float(robot_radius_m)
        self.traversable = self.clearance >= self.robot_radius

    def cells(self, xy):
        xy = np.asarray(xy, dtype=float)
        dx, dy = xy[..., 0] - self.origin[0], xy[..., 1] - self.origin[1]
        col = np.floor((self.cos * dx + self.sin * dy) / self.resolution).astype(int)
        row = np.floor((-self.sin * dx + self.cos * dy) / self.resolution).astype(int)
        return row, col

    def centres(self, rows, cols):
        lx = (np.asarray(cols) + 0.5) * self.resolution
        ly = (np.asarray(rows) + 0.5) * self.resolution
        return np.stack([self.origin[0] + self.cos * lx - self.sin * ly,
                         self.origin[1] + self.sin * lx + self.cos * ly], axis=-1)

    def _lookup(self, mask, xy):
        row, col = self.cells(xy)
        inside = (row >= 0) & (row < self.shape[0]) & (col >= 0) & (col < self.shape[1])
        out = np.zeros(row.shape, dtype=bool)
        out[inside] = mask[row[inside], col[inside]]
        return out

    def is_traversable(self, xy):
        return self._lookup(self.traversable, xy)

    def is_free(self, xy):
        return self._lookup(self.free, xy)

    def segments_clear(self, a, targets, relaxed=False):
        """Per target: is the straight segment from ``a`` traversable throughout?

        ``relaxed`` only requires known free cells, for attaching a robot that
        stands closer to a wall than the robot radius.
        """
        a = np.asarray(a, dtype=float)
        targets = np.atleast_2d(np.asarray(targets, dtype=float))
        if not len(targets):
            return np.zeros(0, dtype=bool)
        offsets = targets - a
        length = np.hypot(offsets[:, 0], offsets[:, 1]).max()
        count = max(2, int(math.ceil(length / (0.25 * self.resolution))) + 1)
        t = np.linspace(0.0, 1.0, count)
        points = a + t[None, :, None] * offsets[:, None, :]
        ok = self.is_free(points) if relaxed else self.is_traversable(points)
        return ok.all(axis=1)

    def medial_axis(self, min_clearance_m, min_separation_m):
        """Cells whose neighbours have nearest obstacles far apart: the centre lines.

        Two adjacent cells on opposite sides of a passage's centre line have
        their nearest obstacle on opposite walls. Bumps on one wall give nearby
        obstacle points, so ``min_separation_m`` removes their short spurs.
        """
        height, width = self.shape
        axis = np.zeros(self.shape, dtype=bool)
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            rows = slice(0, height - dr)
            a_cols = slice(max(0, -dc), width - max(0, dc))
            b_cols = slice(max(0, dc), width - max(0, -dc))
            a = self.nearest_obstacle[:, rows, a_cols]
            b = self.nearest_obstacle[:, dr:, b_cols]
            separated = np.hypot(a[0] - b[0], a[1] - b[1]) * self.resolution >= min_separation_m
            # Mark whichever of the pair is further from the walls.
            ca, cb = self.clearance[rows, a_cols], self.clearance[dr:, b_cols]
            axis[rows, a_cols] |= separated & (ca >= cb)
            axis[dr:, b_cols] |= separated & (cb > ca)
        return axis & self.traversable & (self.clearance >= min_clearance_m)


class Roadmap:
    """Undirected graph with integer node ids and Euclidean edge lengths."""

    def __init__(self):
        self.nodes: dict[int, RoadmapNode] = {}
        self.edges: dict[int, dict[int, float]] = {}
        self._next_id = 0

    def add_node(self, node: RoadmapNode) -> int:
        node_id = self._next_id
        self._next_id += 1
        self.nodes[node_id] = node
        self.edges[node_id] = {}
        return node_id

    def remove_node(self, node_id: int) -> None:
        for other in self.edges.pop(node_id):
            del self.edges[other][node_id]
        del self.nodes[node_id]

    def add_edge(self, a: int, b: int) -> None:
        length = math.dist(self.xy(a), self.xy(b))
        self.edges[a][b] = self.edges[b][a] = length

    def remove_edge(self, a: int, b: int) -> None:
        del self.edges[a][b]
        del self.edges[b][a]

    def xy(self, node_id):
        node = self.nodes[node_id]
        return (node.x, node.y)

    def arrays(self):
        """Node ids and an (N, 2) array of positions, in matching order."""
        ids = np.fromiter(self.nodes, dtype=int, count=len(self.nodes))
        xy = np.array([self.xy(i) for i in ids], dtype=float).reshape(-1, 2)
        return ids, xy

    def edge_list(self):
        return [(a, b) for a, others in self.edges.items() for b in others if a < b]

    def components(self):
        """Map node id -> component index (largest component first)."""
        label, groups = {}, []
        for start in self.nodes:
            if start in label:
                continue
            group, stack = [], [start]
            label[start] = -1
            while stack:
                node = stack.pop()
                group.append(node)
                for other in self.edges[node]:
                    if other not in label:
                        label[other] = -1
                        stack.append(other)
            groups.append(group)
        groups.sort(key=len, reverse=True)
        for index, group in enumerate(groups):
            for node in group:
                label[node] = index
        return label


@dataclass
class UpdateStats:
    added_nodes: int = 0
    removed_nodes: int = 0
    added_edges: int = 0
    removed_edges: int = 0
    seconds: float = 0.0


@dataclass
class BuilderTotals:
    updates: int = 0
    removed_nodes: int = 0
    removed_edges: int = 0
    seconds: float = 0.0
    max_seconds: float = 0.0
    history: list = field(default_factory=list)


class RoadmapBuilder:
    """Grow and maintain a roadmap from map snapshots and robot poses."""

    MODES = ('trail', 'observed')

    def __init__(self, mode='observed', trail_spacing_m=2.0, attach_radius_m=8.0,
                 max_neighbours=4, spacing_per_clearance=2.5, min_spacing_m=1.5,
                 max_spacing_m=6.0, axis_min_clearance_m=0.8, min_separation_m=2.0,
                 connect_per_spacing=2.5):
        if mode not in self.MODES:
            raise ValueError(f'mode must be one of {self.MODES}')
        if not (trail_spacing_m > 0 and attach_radius_m > 0 and max_neighbours > 0
                and 0 < min_spacing_m <= max_spacing_m):
            raise ValueError('invalid roadmap parameters')
        self.mode = mode
        self.trail_spacing_m = float(trail_spacing_m)
        # Radius for joining a start or goal pose to the roadmap when planning.
        self.attach_radius_m = float(attach_radius_m)
        self.max_neighbours = int(max_neighbours)
        self.spacing_per_clearance = float(spacing_per_clearance)
        self.min_spacing_m = float(min_spacing_m)
        self.max_spacing_m = float(max_spacing_m)
        self.axis_min_clearance_m = float(axis_min_clearance_m)
        self.min_separation_m = float(min_separation_m)
        self.connect_per_spacing = float(connect_per_spacing)
        self.roadmap = Roadmap()
        self.totals = BuilderTotals()
        self._snapshot = None

    def update(self, snapshot: MapSnapshot, robot_xy, stamp_s) -> UpdateStats:
        """Fold in a map snapshot (if new) and the robot pose; return what changed."""
        started = time.perf_counter()
        stats = UpdateStats()
        new_nodes = []
        if snapshot is not self._snapshot:
            self._snapshot = snapshot
            self._validate(snapshot, stats)
            # Nodes cut off by removed edges (or never joined) get another try
            # on every new map: space that was unknown may have become free.
            labels = self.roadmap.components()
            for node_id, label in labels.items():
                if label > 0:
                    stats.added_edges += self._connect(snapshot, node_id)
            if self.mode == 'observed':
                new_nodes += self._add_observed(snapshot, stamp_s)
        if robot_xy is not None:
            new_nodes += self._add_trail(snapshot, robot_xy, stamp_s)
        stats.added_nodes = len(new_nodes)
        for node_id in new_nodes:
            stats.added_edges += self._connect(snapshot, node_id)
        stats.seconds = time.perf_counter() - started
        totals = self.totals
        totals.updates += 1
        totals.removed_nodes += stats.removed_nodes
        totals.removed_edges += stats.removed_edges
        totals.seconds += stats.seconds
        totals.max_seconds = max(totals.max_seconds, stats.seconds)
        return stats

    def _validate(self, snapshot, stats):
        """Drop nodes and edges that the newest map no longer supports."""
        roadmap = self.roadmap
        ids, xy = roadmap.arrays()
        if len(ids):
            for node_id in ids[~snapshot.is_traversable(xy)]:
                stats.removed_edges += len(roadmap.edges[node_id])
                roadmap.remove_node(int(node_id))
                stats.removed_nodes += 1
        for a, b in roadmap.edge_list():
            if not snapshot.segments_clear(roadmap.xy(a), [roadmap.xy(b)])[0]:
                roadmap.remove_edge(a, b)
                stats.removed_edges += 1

    def _add_trail(self, snapshot, robot_xy, stamp_s):
        roadmap = self.roadmap
        robot_xy = np.asarray(robot_xy, dtype=float)
        if not snapshot.is_traversable(robot_xy):
            return []
        _, xy = roadmap.arrays()
        if len(xy) and np.hypot(*(xy - robot_xy).T).min() < self.trail_spacing_m:
            return []
        return [roadmap.add_node(RoadmapNode(float(robot_xy[0]), float(robot_xy[1]),
                                             'trail', float(stamp_s)))]

    def spacing(self, clearance_m):
        """Node spacing for a given clearance: dense in passages, sparse in chambers."""
        return np.clip(self.spacing_per_clearance * np.asarray(clearance_m),
                       self.min_spacing_m, self.max_spacing_m)

    def _add_observed(self, snapshot, stamp_s):
        """Adaptive Poisson-disk sampling of the medial axis, widest first."""
        roadmap = self.roadmap
        cells = np.argwhere(snapshot.medial_axis(self.axis_min_clearance_m, self.min_separation_m))
        if not len(cells):
            return []
        clearance = snapshot.clearance[cells[:, 0], cells[:, 1]]
        order = np.argsort(-clearance, kind='stable')
        cells, clearance = cells[order], clearance[order]
        xy = snapshot.centres(cells[:, 0], cells[:, 1])
        radius = self.spacing(clearance)
        _, existing = roadmap.arrays()
        if len(existing):
            nearest, _ = cKDTree(existing).query(xy)
            keep = nearest >= radius
            xy, radius = xy[keep], radius[keep]
        accepted = []
        for point, r in zip(xy, radius):
            if accepted and np.hypot(*(np.array(accepted) - point).T).min() < r:
                continue
            accepted.append(point)
        return [roadmap.add_node(RoadmapNode(float(p[0]), float(p[1]), 'observed', float(stamp_s)))
                for p in accepted]

    def _connect(self, snapshot, node_id):
        """Join a node to its nearest valid neighbours and to other components."""
        roadmap = self.roadmap
        ids, xy = roadmap.arrays()
        point = np.array(roadmap.xy(node_id))
        if roadmap.nodes[node_id].kind == 'trail':
            spacing = self.trail_spacing_m
        else:
            row, col = snapshot.cells(point)
            spacing = float(self.spacing(snapshot.clearance[row, col]))
        # Poisson-disk neighbours can be up to twice the spacing apart, so the
        # factor (default 2.5) keeps neighbouring samples within reach.
        radius = self.connect_per_spacing * spacing
        gaps = np.hypot(*(xy - point).T)
        near = np.flatnonzero((gaps <= radius) & (ids != node_id))
        near = near[np.argsort(gaps[near])]
        clear = snapshot.segments_clear(point, xy[near])
        if not clear.any():
            # Nothing valid close by: look as far as the widest spacing allows.
            far = self.connect_per_spacing * max(self.max_spacing_m, self.trail_spacing_m)
            near = np.flatnonzero((gaps <= far) & (ids != node_id))
            near = near[np.argsort(gaps[near])]
            clear = snapshot.segments_clear(point, xy[near])
        added = 0
        components = roadmap.components()
        joined = {components[node_id]}
        for index, ok in zip(near, clear):
            other = int(ids[index])
            if not ok or other in roadmap.edges[node_id]:
                continue
            group = components[other]
            # Nearest neighbours up to the limit; beyond it, only edges that
            # join a component this node is not yet part of.
            if added < self.max_neighbours or group not in joined:
                roadmap.add_edge(node_id, other)
                joined.add(group)
                added += 1
        return added


def attach(roadmap, snapshot, xy, radius_m, limit=5, relaxed=False):
    """Up to ``limit`` nodes reachable from ``xy`` by a valid straight segment."""
    ids, positions = roadmap.arrays()
    if not len(ids):
        return []
    gaps = np.hypot(*(positions - np.asarray(xy, dtype=float)).T)
    near = np.flatnonzero(gaps <= radius_m)
    near = near[np.argsort(gaps[near])][:4 * limit]
    if not len(near):
        return []
    ok = snapshot.segments_clear(xy, positions[near], relaxed=relaxed)
    return [(int(ids[i]), float(gaps[i])) for i in near[ok][:limit]]


def plan_path(roadmap, snapshot, start_xy, goal_xy, radius_m=8.0, relaxed_start=True):
    """A* from start to goal through the roadmap; list of (x, y) or None."""
    start_links = attach(roadmap, snapshot, start_xy, radius_m, relaxed=relaxed_start)
    goal_links = dict(attach(roadmap, snapshot, goal_xy, radius_m))
    if not start_links or not goal_links:
        return None
    goal = np.asarray(goal_xy, dtype=float)

    def h(node_id):
        return math.dist(roadmap.xy(node_id), goal)

    best = {}
    parent = {}
    queue = []
    for node_id, cost in start_links:
        best[node_id] = cost
        parent[node_id] = None
        heapq.heappush(queue, (cost + h(node_id), cost, node_id))
    finish, finish_cost = None, math.inf
    while queue:
        estimate, cost, node_id = heapq.heappop(queue)
        if estimate >= finish_cost:
            break
        if cost > best.get(node_id, math.inf):
            continue
        if node_id in goal_links and cost + goal_links[node_id] < finish_cost:
            finish, finish_cost = node_id, cost + goal_links[node_id]
        for other, length in roadmap.edges[node_id].items():
            new = cost + length
            if new < best.get(other, math.inf):
                best[other] = new
                parent[other] = node_id
                heapq.heappush(queue, (new + h(other), new, other))
    if finish is None:
        return None
    path = [tuple(goal)]
    node_id = finish
    while node_id is not None:
        path.append(roadmap.xy(node_id))
        node_id = parent[node_id]
    path.append(tuple(np.asarray(start_xy, dtype=float)))
    return path[::-1]


def failure_reason(roadmap, snapshot, start_xy, goal_xy, radius_m=8.0):
    """Why ``plan_path`` found nothing, in words for the operator."""
    goal = np.asarray(goal_xy, dtype=float)
    if not snapshot.is_free(goal):
        return 'goal is not known free space'
    if not snapshot.is_traversable(goal):
        return 'goal is closer to a wall than the robot radius'
    if not attach(roadmap, snapshot, goal, radius_m):
        return f'no roadmap node in reach of the goal within {radius_m:.0f} m'
    if not attach(roadmap, snapshot, start_xy, radius_m, relaxed=True):
        return f'no roadmap node in reach of the robot within {radius_m:.0f} m'
    return 'robot and goal are on different roadmap components'


def shortcut(path, snapshot):
    """Skip waypoints while the straight segment stays valid (greedy smoothing)."""
    if path is None or len(path) < 3:
        return path
    points = np.asarray(path, dtype=float)
    out, i = [points[0]], 0
    while i < len(points) - 1:
        ahead = points[i + 1:]
        clear = snapshot.segments_clear(points[i], ahead, relaxed=(i == 0))
        # Furthest waypoint reachable in a straight line.
        j = i + 1 + int(np.flatnonzero(clear)[-1]) if clear.any() else i + 1
        out.append(points[j])
        i = j
    return [tuple(p) for p in out]


def path_length(path):
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))
