"""Frontier extraction and exploration goal selection.

Owner: C.
Contract: repository-root Workplan.md, section 4.5.

Planning 1. Given the SLAM occupancy grid and the robot's position, choose the
next place worth driving to: a reachable point on the boundary between known
free space and unknown space (a *frontier*). Driving there lets the lidar see
past it, the map grows, and new frontiers appear further on.

Pipeline, all on the grid:

1. Classify cells: unknown (-1), occupied (>= 50), free (the rest).
2. Frontier cells are free cells with an unknown 4-neighbour.
3. Group touching frontier cells (8-connectivity) into frontiers. Drop any
   narrower than the robot - a one-cell hole in a wall is noise, not a passage.
4. Measure path distance from the robot through known free space (a BFS on the
   grid, so walls count; straight-line distance would not). Frontiers the robot
   cannot reach through known free space are dropped.
5. In each frontier, the goal is the nearest reachable cell that is at least
   ``min_clearance_m`` from any occupied cell, so Nav2 does not reject it.
6. Score: path length, minus a bonus for frontier length, plus a decaying
   penalty near recently visited goals. Goals near a recent failure are
   excluded for a cooldown. The lowest score wins.

This is the first version: effectively nearest-first. Path cost, frontier size
and revisit avoidance are the incremental features the task sheet suggests.

Defaults follow the simulation (Workplan 5.1, ``slam_params.yaml``,
``nav2_params.yaml``): 0.3 m cells, a robot radius of 0.5 m, Nav2 inflation and
goal tolerance of 1.0 m, and ``allow_unknown: true`` so Nav2 will plan up to
and into unknown space.

Known limitation, left for the next version: frontiers are driven by a 40 m
lidar, but the camera localises artefacts only within 10 m. A side chamber the
lidar has mapped from the main passage has no frontier left, yet the camera may
never have been near it.

Penalties use ``now_ns`` from the caller's ROS clock, as every timeout in this
project does (Workplan 4.1).
"""

from collections import Counter, deque
from dataclasses import dataclass
import math

from geometry_msgs.msg import Point, PoseStamped
import numpy as np
from scipy import ndimage
from visualization_msgs.msg import Marker, MarkerArray

from cave_explorer.interfaces import (
    ExplorationGoal,
    ExplorationGoalStatus,
    NavigationStatus,
)

OCCUPIED_THRESHOLD = 50

_FAILURE_STATUSES = frozenset({
    NavigationStatus.ABORTED,
    NavigationStatus.REJECTED,
    NavigationStatus.ERROR,
})

_REJECTION_TEXT = {
    'too_narrow': 'too narrow',
    'unreachable': 'not reachable through known free space',
    'no_clearance': 'no point clear of walls',
    'failed_recently': 'failed recently',
}


@dataclass(frozen=True)
class Frontier:
    """One frontier from the latest ``select_goal`` call, kept for RViz and tests.

    ``goal``, ``path_m`` and ``cost`` are ``None`` when the frontier was
    rejected before they could be computed. ``rejection`` is empty for a usable
    candidate, otherwise one of the keys of ``_REJECTION_TEXT``.
    """

    points: tuple[tuple[float, float], ...]
    length_m: float
    goal: tuple[float, float] | None
    path_m: float | None
    cost: float | None
    rejection: str


class FrontierExplorer:
    """Chooses exploration goals from the occupancy grid.

    Keeps no map of its own: every call works from the snapshot passed in, so
    the result always reflects the latest SLAM output. The only state carried
    between calls is the visit and failure history fed back through
    ``report_outcome``.
    """

    def __init__(self, *,
                 min_frontier_m: float = 1.0,
                 min_clearance_m: float = 0.7,
                 gain_weight: float = 0.3,
                 visit_penalty_m: float = 8.0,
                 visit_decay_sec: float = 120.0,
                 failure_cooldown_sec: float = 90.0,
                 penalty_radius_m: float = 1.5,
                 start_search_m: float = 1.0):
        """Configure selection.

        ``min_frontier_m``: frontiers shorter than this are ignored; the robot
        is 1 m wide. ``min_clearance_m``: a goal must be this far from any
        occupied cell; robot radius 0.5 m plus margin. ``gain_weight``: metres
        of extra path accepted per metre of frontier length. ``visit_penalty_m``:
        path-length penalty near a goal reached recently, fading linearly over
        ``visit_decay_sec``. ``failure_cooldown_sec``: how long goals near a
        failed one are excluded. ``penalty_radius_m``: how near counts as the
        same place. ``start_search_m``: how far to look for a free cell when the
        robot's own cell is not free, as happens at start-up.
        """
        self.min_frontier_m = min_frontier_m
        self.min_clearance_m = min_clearance_m
        self.gain_weight = gain_weight
        self.visit_penalty_m = visit_penalty_m
        self.visit_decay_ns = int(visit_decay_sec * 1e9)
        self.failure_cooldown_ns = int(failure_cooldown_sec * 1e9)
        self.penalty_radius_m = penalty_radius_m
        self.start_search_m = start_search_m

        self._visits: list[tuple[float, float, int]] = []
        self._failures: list[tuple[float, float, int]] = []

        # Results of the latest select_goal call, for visualisation.
        self.frontiers: tuple[Frontier, ...] = ()
        self.chosen: Frontier | None = None
        self.frame_id = 'map'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_goal(self, map_msg, robot_xy: tuple[float, float],
                    now_ns: int) -> ExplorationGoal:
        """Pick the next exploration goal from ``map_msg`` (a ``nav_msgs/OccupancyGrid``)."""
        self.frontiers = ()
        self.chosen = None
        self._forget_old(now_ns)

        if map_msg is None:
            return self._not_ready('no map received yet')
        info = map_msg.info
        width, height, resolution = info.width, info.height, info.resolution
        if width == 0 or height == 0 or resolution <= 0.0:
            return self._not_ready('map is empty')
        self.frame_id = map_msg.header.frame_id or 'map'
        origin = (info.origin.position.x, info.origin.position.y)

        grid = np.asarray(map_msg.data, dtype=np.int16).reshape(height, width)
        unknown = grid < 0
        occupied = grid >= OCCUPIED_THRESHOLD
        free = ~unknown & ~occupied
        if not free.any():
            return self._not_ready('map has no free cells yet')

        start = self._start_cell(free, robot_xy, origin, resolution)
        if start is None:
            return self._not_ready('robot is not on known free space in the map')

        frontier_mask = free & _has_unknown_neighbour(unknown)
        if not frontier_mask.any():
            return self._exhausted('no frontier left: every free cell borders only known cells')

        distance = _path_cells(free, start).ravel()
        clearance = (ndimage.distance_transform_edt(~occupied) * resolution).ravel()

        frontiers = [self._evaluate(cells, width, origin, resolution, distance, clearance,
                                    now_ns)
                     for cells in _frontier_groups(frontier_mask)]
        self.frontiers = tuple(frontiers)

        candidates = [frontier for frontier in frontiers if not frontier.rejection]
        if not candidates:
            counts = Counter(frontier.rejection for frontier in frontiers)
            summary = ', '.join(f'{count} {_REJECTION_TEXT[key]}'
                                for key, count in sorted(counts.items()))
            return self._exhausted(f'{len(frontiers)} frontiers, none usable: {summary}')

        self.chosen = min(candidates, key=lambda frontier: frontier.cost)
        goal = self._pose(self.chosen.goal, robot_xy, map_msg.header.frame_id)
        return ExplorationGoal(
            ExplorationGoalStatus.READY, goal,
            f'frontier {self.chosen.length_m:.1f} m long, {self.chosen.path_m:.1f} m away')

    def report_outcome(self, goal: PoseStamped, status: NavigationStatus,
                       reason: str, now_ns: int) -> None:
        """Feed back how driving to a goal from ``select_goal`` went.

        ``SUCCEEDED`` marks the place visited; ``ABORTED``, ``REJECTED`` and
        ``ERROR`` mark it failed. ``CANCELED`` changes nothing: navigation_client
        reports our own preemptions that way, and they say nothing about the goal.
        ``reason`` is accepted for the record and the logs, not used in scoring.
        """
        place = (goal.pose.position.x, goal.pose.position.y, now_ns)
        if status is NavigationStatus.SUCCEEDED:
            self._visits.append(place)
        elif status in _FAILURE_STATUSES:
            self._failures.append(place)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate(self, cells, width, origin, resolution, distance, clearance,
                  now_ns) -> Frontier:
        points = tuple(_world(int(cell), width, origin, resolution) for cell in cells)
        length_m = len(cells) * resolution

        def rejected(key, goal=None, path_m=None):
            return Frontier(points, length_m, goal, path_m, None, key)

        if length_m < self.min_frontier_m:
            return rejected('too_narrow')
        reachable = cells[distance[cells] >= 0]
        if reachable.size == 0:
            return rejected('unreachable')
        clear = reachable[clearance[reachable] >= self.min_clearance_m]
        if clear.size == 0:
            return rejected('no_clearance')

        best = int(clear[np.argmin(distance[clear])])
        goal = _world(best, width, origin, resolution)
        path_m = float(distance[best]) * resolution
        if self._near_failure(goal):
            return rejected('failed_recently', goal, path_m)

        cost = path_m - self.gain_weight * length_m + self._visit_penalty(goal, now_ns)
        return Frontier(points, length_m, goal, path_m, cost, '')

    def _start_cell(self, free, robot_xy, origin, resolution):
        """The robot's cell if free, else the nearest free cell within ``start_search_m``."""
        height, width = free.shape
        row = int(math.floor((robot_xy[1] - origin[1]) / resolution))
        col = int(math.floor((robot_xy[0] - origin[0]) / resolution))
        reach = int(math.ceil(self.start_search_m / resolution))
        best = None
        for r in range(max(0, row - reach), min(height, row + reach + 1)):
            for c in range(max(0, col - reach), min(width, col + reach + 1)):
                if free[r, c]:
                    squared = (r - row) ** 2 + (c - col) ** 2
                    if squared <= reach ** 2 and (best is None or squared < best[0]):
                        best = (squared, r, c)
        return None if best is None else (best[1], best[2])

    def _forget_old(self, now_ns):
        self._visits = [v for v in self._visits if now_ns - v[2] < self.visit_decay_ns]
        self._failures = [f for f in self._failures
                          if now_ns - f[2] < self.failure_cooldown_ns]

    def _near_failure(self, goal) -> bool:
        return any(math.dist(goal, failure[:2]) <= self.penalty_radius_m
                   for failure in self._failures)

    def _visit_penalty(self, goal, now_ns) -> float:
        penalty = 0.0
        for x, y, stamp in self._visits:
            if math.dist(goal, (x, y)) <= self.penalty_radius_m:
                remaining = 1.0 - (now_ns - stamp) / self.visit_decay_ns
                penalty += self.visit_penalty_m * max(0.0, remaining)
        return penalty

    @staticmethod
    def _pose(goal_xy, robot_xy, frame_id) -> PoseStamped:
        """Goal pose facing along the direction of travel, so the camera looks ahead."""
        pose = PoseStamped()
        pose.header.frame_id = frame_id or 'map'
        pose.pose.position.x, pose.pose.position.y = goal_xy
        dx, dy = goal_xy[0] - robot_xy[0], goal_xy[1] - robot_xy[1]
        yaw = math.atan2(dy, dx) if (dx or dy) else 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose

    @staticmethod
    def _not_ready(reason) -> ExplorationGoal:
        return ExplorationGoal(ExplorationGoalStatus.NOT_READY, None, reason)

    @staticmethod
    def _exhausted(reason) -> ExplorationGoal:
        return ExplorationGoal(ExplorationGoalStatus.NO_REACHABLE_FRONTIER, None, reason)


# ----------------------------------------------------------------------
# Grid helpers
# ----------------------------------------------------------------------

def _has_unknown_neighbour(unknown):
    """Cells with an unknown 4-neighbour. Outside the map does not count as unknown."""
    padded = np.pad(unknown, 1, constant_values=False)
    return padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]


def _frontier_groups(frontier_mask):
    """Flat cell indices of each 8-connected group of frontier cells."""
    labels, count = ndimage.label(frontier_mask, structure=np.ones((3, 3), dtype=int))
    flat = labels.ravel()
    cells = np.flatnonzero(flat)
    ordered = cells[np.argsort(flat[cells], kind='stable')]
    sizes = np.bincount(flat[cells], minlength=count + 1)[1:]
    return np.split(ordered, np.cumsum(sizes)[:-1])


def _path_cells(free, start):
    """BFS distance in cells through free space from ``start``; -1 where unreachable.

    4-connected, so diagonal moves count double. That over-estimates the true
    length by up to about 40 %, evenly across frontiers, which is fine for
    ranking them.
    """
    height, width = free.shape
    passable = free.ravel().tolist()
    distance = [-1] * (height * width)
    origin = start[0] * width + start[1]
    distance[origin] = 0
    queue = deque([origin])
    while queue:
        cell = queue.popleft()
        step = distance[cell] + 1
        col = cell % width
        for neighbour, inside in ((cell - width, cell >= width),
                                  (cell + width, cell < (height - 1) * width),
                                  (cell - 1, col > 0),
                                  (cell + 1, col < width - 1)):
            if inside and passable[neighbour] and distance[neighbour] < 0:
                distance[neighbour] = step
                queue.append(neighbour)
    return np.array(distance, dtype=np.int32).reshape(height, width)


def _world(cell, width, origin, resolution):
    """Centre of a flat grid cell in map coordinates. The SLAM map origin is never rotated."""
    row, col = divmod(cell, width)
    return (origin[0] + (col + 0.5) * resolution,
            origin[1] + (row + 0.5) * resolution)


# ----------------------------------------------------------------------
# RViz
# ----------------------------------------------------------------------

def frontier_markers(explorer: FrontierExplorer) -> MarkerArray:
    """Markers for the latest ``select_goal``: candidates, rejected frontiers, the choice.

    Usable frontier cells are cyan with a yellow goal; rejected ones are grey;
    the chosen goal is a large green sphere. Planning 1 asks for the chosen
    frontier and candidate goals in RViz; the path comes from Nav2 on ``/plan``.
    """
    frame_id = explorer.frame_id
    clear_all = Marker()
    clear_all.header.frame_id = frame_id
    clear_all.action = Marker.DELETEALL
    markers = [clear_all]

    usable = [f for f in explorer.frontiers if not f.rejection]
    rejected = [f for f in explorer.frontiers if f.rejection]

    markers.append(_points_marker(frame_id, 'frontier_cells', 0, Marker.POINTS, 0.15,
                                  (0.0, 0.9, 0.9, 1.0),
                                  [p for f in usable for p in f.points]))
    markers.append(_points_marker(frame_id, 'frontier_rejected', 0, Marker.POINTS, 0.1,
                                  (0.5, 0.5, 0.5, 0.8),
                                  [p for f in rejected for p in f.points]))
    markers.append(_points_marker(frame_id, 'frontier_goals', 0, Marker.SPHERE_LIST, 0.4,
                                  (1.0, 0.85, 0.0, 1.0), [f.goal for f in usable]))
    if explorer.chosen is not None:
        markers.append(_points_marker(frame_id, 'frontier_chosen', 0, Marker.SPHERE_LIST, 0.8,
                                      (0.1, 0.9, 0.1, 1.0), [explorer.chosen.goal]))
    return MarkerArray(markers=markers)


def _points_marker(frame_id, namespace, marker_id, marker_type, size, rgba, points):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = namespace
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = marker.scale.y = marker.scale.z = size
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba
    marker.points = [Point(x=x, y=y, z=0.1) for x, y in points]
    return marker
