"""Camera coverage: where the camera has looked, and where to drive so it looks everywhere.

Owner: C.
Contract: repository-root Workplan.md, section 4.5 (the Explorer protocol).

Planning 1, second phase. Frontier exploration finishes when the *lidar* has
mapped the cave, but the lidar reaches 40 m and artefacts are localised only
within the camera's 10 m depth range. In simulation the map closed after 5.5
minutes of driving with most of the cave never in camera view, so most
artefacts could not have been found.

``CoverageExplorer.observe`` marks the known cells the camera can see from its
current pose: rays across the horizontal field of view out to
``view_range_m``, each stopping at the first occupied cell. ``select_goal``
groups free cells the camera has not seen, reachable through known free space,
into connected regions; regions smaller than ``min_region_m2`` are lidar
shadows behind pillars and are ignored. Each region's goal is its cell nearest
by path that is clear of walls (or the clear cell closest to that), facing the
region's centroid. Score: path length minus a bonus per square metre unseen.
Cells near a goal reached or failed recently are left out when picking the
nearest cell, not the whole region: the nearest unseen cell is often right
behind the robot, and dropping the region would end coverage early.

``TwoPhaseExplorer`` runs frontier exploration until it reports no reachable
frontier, then coverage, and routes each outcome back to whichever planner
produced the goal. It exposes ``frontiers``/``chosen``/``frame_id`` of the
planner in use, so ``BehaviourManager`` replanning and ``frontier_markers``
work unchanged; coverage regions are reported as ``Frontier`` records whose
``length_m`` holds the region's area in square metres.
"""

from collections import Counter
import math

from nav_msgs.msg import OccupancyGrid
import numpy as np
from scipy import ndimage

from cave_explorer.frontier_explorer import (
    OCCUPIED_THRESHOLD,
    Frontier,
    FrontierExplorer,
    _path_cells,
    _world,
)
from cave_explorer.interfaces import (
    ExplorationGoal,
    ExplorationGoalStatus,
    NavigationStatus,
)

_FAILURES = frozenset({NavigationStatus.ABORTED, NavigationStatus.REJECTED,
                       NavigationStatus.ERROR})

_REJECTION_TEXT = {
    'too_small': 'too small',
    'no_clearance': 'no point clear of walls',
    'failed_recently': 'failed recently',
    'visited_recently': 'visited recently',
}


class CoverageExplorer:
    """Remembers what the camera has seen and picks goals to see the rest."""

    def __init__(self, *,
                 view_range_m: float = 9.0,
                 fov_rad: float = 2.0944,
                 min_region_m2: float = 2.0,
                 min_clearance_m: float = 0.7,
                 gain_per_m2: float = 0.05,
                 goal_search_m: float = 3.0,
                 failure_cooldown_sec: float = 90.0,
                 visit_cooldown_sec: float = 120.0,
                 penalty_radius_m: float = 1.5,
                 start_search_m: float = 1.0):
        """``view_range_m`` is how far the camera is trusted to spot an
        artefact and the tracker to place it: 1 m inside the 10 m depth limit,
        beyond which a detection cannot be localised. Replaying a recorded run,
        the detector found every visible artefact of every class in every
        frame out to 10 m (white_sphere, the first to drop, 94% at 10-12 m).
        It was 6 m for the earlier detector. ``fov_rad`` is the camera's
        horizontal field of view (Workplan 5.1). ``min_clearance_m`` matches
        FrontierExplorer: robot radius 0.5 m plus margin. ``goal_search_m``
        bounds how far from a region's nearest cell a clear goal may be. A goal
        near a failed one is skipped for ``failure_cooldown_sec``, and near one
        reached for ``visit_cooldown_sec``: arriving should have revealed it, so
        returning would loop."""
        self.view_range_m = view_range_m
        self.fov_rad = fov_rad
        self.min_region_m2 = min_region_m2
        self.min_clearance_m = min_clearance_m
        self.gain_per_m2 = gain_per_m2
        self.goal_search_m = goal_search_m
        self.failure_cooldown_ns = int(failure_cooldown_sec * 1e9)
        self.visit_cooldown_ns = int(visit_cooldown_sec * 1e9)
        self.penalty_radius_m = penalty_radius_m
        self.start_search_m = start_search_m

        self._seen = None           # bool grid aligned with the latest map
        self._geometry = None       # (width, height, origin_x, origin_y, resolution)
        self._failures = []         # (x, y, stamp_ns)
        self._visits = []

        self.frontiers: tuple[Frontier, ...] = ()
        self.chosen: Frontier | None = None
        self.frame_id = 'map'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, map_msg, camera_xy, camera_yaw: float) -> None:
        """Mark the known cells visible from the camera's current pose."""
        if map_msg is None or map_msg.info.width == 0 or map_msg.info.height == 0:
            return
        grid, (ox, oy), res = self._align(map_msg)
        height, width = grid.shape
        occupied = grid >= OCCUPIED_THRESHOLD
        known = grid >= 0

        n_rays = int(math.ceil(self.fov_rad * self.view_range_m / res * 1.5)) + 1
        angles = camera_yaw + np.linspace(-self.fov_rad / 2, self.fov_rad / 2, n_rays)
        steps = np.arange(1, int(self.view_range_m / (res / 2)) + 1) * (res / 2)
        xs = camera_xy[0] + np.cos(angles)[:, None] * steps[None, :]
        ys = camera_xy[1] + np.sin(angles)[:, None] * steps[None, :]
        cols = np.floor((xs - ox) / res).astype(int)
        rows = np.floor((ys - oy) / res).astype(int)
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        r, c = np.clip(rows, 0, height - 1), np.clip(cols, 0, width - 1)
        hit = inside & occupied[r, c]
        # A ray sees up to and including its first occupied cell. The map is a
        # rectangle, so a ray that leaves it never comes back.
        behind_wall = (np.cumsum(hit, axis=1) - hit) > 0
        visible = inside & ~behind_wall & known[r, c]
        self._seen[r[visible], c[visible]] = True

    def select_goal(self, map_msg, robot_xy, now_ns: int) -> ExplorationGoal:
        """Drive to the most worthwhile region the camera has not seen yet."""
        self.frontiers = ()
        self.chosen = None
        self._forget_old(now_ns)
        if map_msg is None or map_msg.info.width == 0 or map_msg.info.height == 0:
            return ExplorationGoal(ExplorationGoalStatus.NOT_READY, None, 'no map yet')
        self.frame_id = map_msg.header.frame_id or 'map'
        grid, origin, res = self._align(map_msg)
        width = grid.shape[1]
        occupied = grid >= OCCUPIED_THRESHOLD
        free = (grid >= 0) & ~occupied

        start = self._start_cell(free, robot_xy, origin, res)
        if start is None:
            return ExplorationGoal(ExplorationGoalStatus.NOT_READY, None,
                                   'robot is not on known free space in the map')
        distance = _path_cells(free, start)
        reachable = distance >= 0
        clear = reachable & (ndimage.distance_transform_edt(~occupied) * res
                             >= self.min_clearance_m)

        labels, count = ndimage.label(free & reachable & ~self._seen,
                                      structure=np.ones((3, 3), dtype=bool))
        clear_cells = np.flatnonzero(clear)
        clear_cells = clear_cells[~self._near_cells(clear_cells, width, origin, res,
                                                    self._failures)]
        flat = labels.ravel()
        cells = np.flatnonzero(flat)
        cells = cells[np.argsort(flat[cells], kind='stable')]
        groups = np.split(cells, np.flatnonzero(np.diff(flat[cells])) + 1) if count else []
        regions = [self._evaluate(group, width, origin, res, distance.ravel(), clear.ravel(),
                                  clear_cells, now_ns)
                   for group in groups]
        self.frontiers = tuple(regions)

        seen_pct = 100.0 * self.seen_fraction(map_msg)
        candidates = [region for region in regions if not region.rejection]
        if not candidates:
            counts = Counter(region.rejection for region in regions)
            summary = ', '.join(f'{n} {_REJECTION_TEXT[key]}' for key, n in sorted(counts.items()))
            return ExplorationGoal(
                ExplorationGoalStatus.NO_REACHABLE_FRONTIER, None,
                f'camera has seen {seen_pct:.0f}% of free space; '
                + (f'{len(regions)} unseen regions left: {summary}' if regions
                   else 'nothing reachable left unseen'))

        self.chosen = min(candidates, key=lambda region: region.cost)
        goal = FrontierExplorer._pose(self.chosen.goal, robot_xy, self.frame_id)
        centroid = np.mean(self.chosen.points, axis=0)
        yaw = math.atan2(centroid[1] - self.chosen.goal[1], centroid[0] - self.chosen.goal[0])
        if math.dist(centroid, self.chosen.goal) > 0.5:     # else keep facing along the path
            goal.pose.orientation.z = math.sin(yaw / 2.0)
            goal.pose.orientation.w = math.cos(yaw / 2.0)
        return ExplorationGoal(
            ExplorationGoalStatus.READY, goal,
            f'coverage: {self.chosen.length_m:.0f} m2 unseen, {self.chosen.path_m:.1f} m away '
            f'({seen_pct:.0f}% seen)')

    def report_outcome(self, goal, status: NavigationStatus, reason: str, now_ns: int) -> None:
        place = (goal.pose.position.x, goal.pose.position.y, now_ns)
        if status is NavigationStatus.SUCCEEDED:
            self._visits.append(place)
        elif status in _FAILURES:
            self._failures.append(place)

    def seen_fraction(self, map_msg) -> float:
        """Fraction of the map's free cells the camera has seen."""
        grid, _, _ = self._align(map_msg)
        free = (grid >= 0) & (grid < OCCUPIED_THRESHOLD)
        return float((free & self._seen).sum()) / max(int(free.sum()), 1)

    def coverage_grid(self, map_msg) -> OccupancyGrid:
        """Free cells not yet seen as 100, everything else 0 - for RViz (costmap scheme)."""
        grid, _, _ = self._align(map_msg)
        free = (grid >= 0) & (grid < OCCUPIED_THRESHOLD)
        msg = OccupancyGrid()
        msg.header = map_msg.header
        msg.info = map_msg.info
        msg.data = np.where(free & ~self._seen, 100, 0).astype(np.int8).ravel().tolist()
        return msg

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate(self, cells, width, origin, res, distance, clear, clear_cells, now_ns):
        points = tuple(_world(int(cell), width, origin, res) for cell in cells)
        area_m2 = len(cells) * res * res

        def rejected(key, goal=None, path_m=None):
            return Frontier(points, area_m2, goal, path_m, None, key)

        if area_m2 < self.min_region_m2:
            return rejected('too_small')
        near_visit = self._near_cells(cells, width, origin, res, self._visits)
        near_failure = self._near_cells(cells, width, origin, res, self._failures)
        allowed = cells[~near_visit & ~near_failure]
        if allowed.size == 0:
            return rejected('visited_recently' if near_visit.all() else 'failed_recently')
        anchor = int(allowed[np.argmin(distance[allowed])])
        if clear[anchor]:
            best = anchor
        else:
            ar, ac = divmod(anchor, width)
            cr, cc = np.divmod(clear_cells, width)
            gap = np.hypot(cr - ar, cc - ac) * res
            near = gap <= self.goal_search_m
            if not near.any():
                return rejected('no_clearance')
            best = int(clear_cells[near][np.argmin(gap[near])])
        goal = _world(best, width, origin, res)
        path_m = float(distance[best]) * res
        return Frontier(points, area_m2, goal, path_m,
                        path_m - self.gain_per_m2 * area_m2, '')

    def _align(self, map_msg):
        """Keep the seen grid on the current map, carrying it over when SLAM resizes it."""
        info = map_msg.info
        res = info.resolution
        origin = (info.origin.position.x, info.origin.position.y)
        geometry = (info.width, info.height, origin[0], origin[1], res)
        grid = np.asarray(map_msg.data, dtype=np.int16).reshape(info.height, info.width)
        if geometry != self._geometry:
            seen = np.zeros((info.height, info.width), dtype=bool)
            if self._seen is not None and math.isclose(self._geometry[4], res):
                dc = int(round((self._geometry[2] - origin[0]) / res))
                dr = int(round((self._geometry[3] - origin[1]) / res))
                old_h, old_w = self._seen.shape
                r0, c0 = max(0, dr), max(0, dc)
                r1, c1 = min(info.height, dr + old_h), min(info.width, dc + old_w)
                if r1 > r0 and c1 > c0:
                    seen[r0:r1, c0:c1] = self._seen[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
            self._seen, self._geometry = seen, geometry
        return grid, origin, res

    def _start_cell(self, free, robot_xy, origin, res):
        """The robot's cell if free, else the nearest free cell within ``start_search_m``."""
        row = int(math.floor((robot_xy[1] - origin[1]) / res))
        col = int(math.floor((robot_xy[0] - origin[0]) / res))
        reach = int(math.ceil(self.start_search_m / res))
        r0, c0 = max(0, row - reach), max(0, col - reach)
        window = free[r0:row + reach + 1, c0:col + reach + 1]
        if not window.any():
            return None
        rr, cc = np.nonzero(window)
        squared = (rr + r0 - row) ** 2 + (cc + c0 - col) ** 2
        best = int(np.argmin(squared))
        if squared[best] > reach ** 2:
            return None
        return int(rr[best] + r0), int(cc[best] + c0)

    def _forget_old(self, now_ns):
        self._failures = [f for f in self._failures if now_ns - f[2] < self.failure_cooldown_ns]
        self._visits = [v for v in self._visits if now_ns - v[2] < self.visit_cooldown_ns]

    def _near_cells(self, cells, width, origin, res, places):
        """Which of ``cells`` lie within ``penalty_radius_m`` of any of ``places``."""
        near = np.zeros(cells.shape, dtype=bool)
        if not places or cells.size == 0:
            return near
        rows, cols = np.divmod(cells, width)
        xs = origin[0] + (cols + 0.5) * res
        ys = origin[1] + (rows + 0.5) * res
        for x, y, _ in places:
            near |= np.hypot(xs - x, ys - y) <= self.penalty_radius_m
        return near


class TwoPhaseExplorer:
    """Frontier exploration until the map is done, then camera coverage."""

    def __init__(self, frontier: FrontierExplorer, coverage: CoverageExplorer):
        self.frontier = frontier
        self.coverage = coverage
        self.active = frontier
        self._sources = []      # (goal, planner) for recent goals, newest last

    @property
    def frontiers(self):
        return self.active.frontiers

    @property
    def chosen(self):
        return self.active.chosen

    @property
    def frame_id(self):
        return self.active.frame_id

    def select_goal(self, map_msg, robot_xy, now_ns: int) -> ExplorationGoal:
        result = self.frontier.select_goal(map_msg, robot_xy, now_ns)
        self.active = self.frontier
        if result.status is ExplorationGoalStatus.NO_REACHABLE_FRONTIER:
            result = self.coverage.select_goal(map_msg, robot_xy, now_ns)
            self.active = self.coverage
        if result.status is ExplorationGoalStatus.READY:
            self._sources = self._sources[-15:] + [(result.goal, self.active)]
        return result

    def report_outcome(self, goal, status, reason, now_ns: int) -> None:
        for sent, planner in reversed(self._sources):
            if sent is goal:
                planner.report_outcome(goal, status, reason, now_ns)
                return
