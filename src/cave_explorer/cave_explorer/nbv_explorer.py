"""Information-gain exploration for the camera (Planning 1).

Only the camera finds artefacts, so the information that matters is camera
views of places that could hold one. Information grid:

* a known free cell the camera has not seen: 1 (unless in an unseen patch
  smaller than ``min_region_m2``);
* an unknown cell within ``view_range_m`` of a usable frontier: ``p_free``,
  the fraction of known cells that are free;
* everything else: 0.

A viewpoint's gain is the information in the camera's view from it (rays to
``view_range_m``, stopping at walls, best of ``n_headings`` headings). Utility
(Gonzalez-Banos and Latombe, 2002):

    U = gain_m2 * exp(-lambda_per_m * path_m)

Candidates are ray cast in order of an FFT upper bound until no bound can beat
the best exact utility. Exploration ends when no viewpoint shows ``min_gain_m2``.
"""

import math
from types import SimpleNamespace

import numpy as np
from scipy import ndimage, signal
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from cave_explorer.coverage_explorer import CoverageExplorer
from cave_explorer.frontier_explorer import (
    OCCUPIED_THRESHOLD,
    Frontier,
    FrontierExplorer,
    _has_unknown_neighbour,
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


class InformationGainExplorer:
    """Picks the viewpoint whose camera view is worth most per path length."""

    def __init__(self, coverage: CoverageExplorer, frontier: FrontierExplorer | None = None, *,
                 lambda_per_m: float = 0.1,
                 spacing_m: float = 1.5,
                 n_headings: int = 12,
                 min_gain_m2: float = 2.0,
                 batch: int = 16,
                 visit_cooldown_sec: float = 120.0,
                 failure_cooldown_sec: float = 90.0,
                 penalty_radius_m: float = 1.5,
                 settle_sec: float = 10.0):
        """``coverage`` holds what the camera has seen; ``frontier`` finds usable
        frontiers. ``lambda_per_m`` 0.1 halves a view's value every 7 m.
        ``spacing_m``: candidate lattice. Viewpoints near a reached or failed
        one cool down. Unknown cells still unknown ``settle_sec`` after a
        viewpoint with line of sight was reached are treated as unobservable."""
        self.coverage = coverage
        self.frontier = frontier or FrontierExplorer(min_clearance_m=coverage.min_clearance_m)
        self.lambda_per_m = lambda_per_m
        self.spacing_m = spacing_m
        self.n_headings = n_headings
        self.min_gain_m2 = min_gain_m2
        self.batch = batch
        self.visit_cooldown_ns = int(visit_cooldown_sec * 1e9)
        self.failure_cooldown_ns = int(failure_cooldown_sec * 1e9)
        self.penalty_radius_m = penalty_radius_m
        self.settle_ns = int(settle_sec * 1e9)

        self._visits = []           # (x, y, stamp_ns)
        self._reached = []          # every viewpoint reached, (x, y, stamp_ns)
        self._failures = []
        self._last_goal = None      # (x, y) of the previous choice
        self._ray_cache = None      # (resolution, offsets) for one cell size
        self._usable = None         # usable frontier cells of the latest plan

        self.frontiers: tuple[Frontier, ...] = ()
        self.chosen: Frontier | None = None
        self.frame_id = 'map'

    # ------------------------------------------------------------------
    # Explorer protocol
    # ------------------------------------------------------------------

    def select_goal(self, map_msg, robot_xy, now_ns: int) -> ExplorationGoal:
        ctx = self._prepare(map_msg, robot_xy, now_ns)
        if isinstance(ctx, ExplorationGoal):
            return ctx
        value, occupied, distance = ctx.value, ctx.occupied, ctx.distance
        width, height, origin, res = ctx.width, ctx.height, ctx.origin, ctx.res
        cells, previous, seen_pct = ctx.cells, ctx.previous, ctx.seen_pct

        bound_m2 = self._bound_m2(value, cells, res)
        path_m = distance.ravel()[cells] * res
        bound_u = bound_m2 * np.exp(-self.lambda_per_m * path_m)
        hopeful = bound_m2 >= self.min_gain_m2
        cells, path_m, bound_u = cells[hopeful], path_m[hopeful], bound_u[hopeful]
        order = np.argsort(-bound_u, kind='stable')

        # The previous goal is always cast, so replanning can see it still pays.
        first = np.flatnonzero(cells == previous)
        queue = list(first) + [j for j in order if j not in set(first)]
        evaluated, best = [], None
        for i in range(0, len(queue), self.batch):
            if best is not None and i > 0 and bound_u[queue[i]] <= best[0]:
                break       # nothing further down can beat it
            for j in queue[i:i + self.batch]:
                gain_m2, yaw = self._view_gain(int(cells[j]), value, occupied, width, height, res)
                if gain_m2 < self.min_gain_m2:
                    continue
                utility = gain_m2 * math.exp(-self.lambda_per_m * path_m[j])
                goal = _world(int(cells[j]), width, origin, res)
                record = Frontier((goal,), gain_m2, goal, float(path_m[j]), -utility, '')
                evaluated.append(record)
                if best is None or utility > best[0]:
                    best = (utility, record, yaw)
        self.frontiers = tuple(evaluated)
        if best is None:
            return self._done(seen_pct, f'no viewpoint shows {self.min_gain_m2:g} m2')

        _, self.chosen, yaw = best
        self._last_goal = self.chosen.goal
        goal = FrontierExplorer._pose(self.chosen.goal, robot_xy, self.frame_id)
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        return ExplorationGoal(
            ExplorationGoalStatus.READY, goal,
            f'information gain: {self.chosen.length_m:.0f} m2 in view, '
            f'{self.chosen.path_m:.1f} m away ({seen_pct:.0f}% seen)')

    def report_outcome(self, goal, status: NavigationStatus, reason: str, now_ns: int) -> None:
        place = (goal.pose.position.x, goal.pose.position.y, now_ns)
        if status is NavigationStatus.SUCCEEDED:
            self._visits.append(place)
            self._reached.append(place)
        elif status in _FAILURES:
            self._failures.append(place)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare(self, map_msg, robot_xy, now_ns):
        """Grids and candidates shared by the planners, or the goal to return
        when there is nothing to plan on."""
        self.frontiers = ()
        self.chosen = None
        self._forget_old(now_ns)
        if map_msg is None or map_msg.info.width == 0 or map_msg.info.height == 0:
            return ExplorationGoal(ExplorationGoalStatus.NOT_READY, None, 'no map yet')
        self.frame_id = map_msg.header.frame_id or 'map'
        grid, origin, res = self.coverage._align(map_msg)
        height, width = grid.shape
        occupied = grid >= OCCUPIED_THRESHOLD
        unknown = grid < 0
        free = ~unknown & ~occupied

        start = self.coverage._start_cell(free, robot_xy, origin, res)
        if start is None:
            return ExplorationGoal(ExplorationGoalStatus.NOT_READY, None,
                                   'robot is not on known free space in the map')
        distance = _path_cells(free, start)
        clear = (distance >= 0) & (ndimage.distance_transform_edt(~occupied) * res
                                   >= self.coverage.min_clearance_m)

        value = self._information(map_msg, robot_xy, now_ns, grid, free, unknown, distance, res)
        seen_pct = 100.0 * self.coverage.seen_fraction(map_msg)
        cells, previous = self._candidates(clear, origin, res, width)
        if cells.size == 0 or not value.any():
            return self._done(seen_pct, 'nothing left to see')
        return SimpleNamespace(value=value, occupied=occupied, free=free, unknown=unknown,
                               distance=distance,
                               width=width, height=height, origin=origin, res=res,
                               cells=cells, previous=previous, seen_pct=seen_pct)

    def _bound_m2(self, value, cells, res):
        """Upper bound on each candidate's gain: all information within view range."""
        radius = int(self.coverage.view_range_m / res)
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        disk = (xx ** 2 + yy ** 2 <= radius ** 2).astype(float)
        return signal.fftconvolve(value, disk, mode='same').ravel()[cells] * res * res

    def _information(self, map_msg, robot_xy, now_ns, grid, free, unknown, distance, res):
        """Expected artefact-bearing area per cell, in cells (see module docstring)."""
        unseen = free & (distance >= 0) & ~self.coverage._seen
        labels, count = ndimage.label(unseen, structure=np.ones((3, 3), dtype=bool))
        if count:
            sizes = np.bincount(labels.ravel(), minlength=count + 1) * res * res
            sizes[0] = 0.0
            unseen = sizes[labels] >= self.coverage.min_region_m2
        value = unseen.astype(float)

        known = int((grid >= 0).sum())
        p_free = float(free.sum()) / max(known, 1)
        self.frontier.select_goal(map_msg, robot_xy, now_ns)
        usable = np.zeros(grid.shape, dtype=bool)
        for f in self.frontier.frontiers:
            if not f.rejection:
                for x, y in f.points:
                    usable[int((y - map_msg.info.origin.position.y) / res),
                           int((x - map_msg.info.origin.position.x) / res)] = True
        self._usable = usable
        if usable.any():
            near = ndimage.distance_transform_edt(~usable) * res <= self.coverage.view_range_m
            value[unknown & near] = p_free
            value.ravel()[self._unobservable(map_msg, grid, now_ns, res)] = 0.0
        return value

    def _unobservable(self, map_msg, grid, now_ns, res):
        """Flat indices of unknown cells in line of sight of a viewpoint reached
        at least ``settle_sec`` ago: standing there did not map them."""
        height, width = grid.shape
        occupied = grid >= OCCUPIED_THRESHOLD
        unknown = grid.ravel() < 0
        ox, oy = map_msg.info.origin.position.x, map_msg.info.origin.position.y
        cells = []
        for x, y, stamp in self._reached:
            row, col = int((y - oy) / res), int((x - ox) / res)
            if now_ns - stamp >= self.settle_ns and 0 <= row < height and 0 <= col < width:
                flat, _ = self._cast(row * width + col, occupied, width, height, res)
                cells.append(flat[unknown[flat]])
        return np.unique(np.concatenate(cells)) if cells else np.zeros(0, dtype=int)

    def _candidates(self, clear, origin, res, width):
        """Flat indices of lattice cells clear of walls, plus the previous goal,
        without those near a recent visit or failure; and the previous goal's
        index (-1 if none)."""
        step = max(1, int(round(self.spacing_m / res)))
        lattice = np.zeros(clear.shape, dtype=bool)
        lattice[::step, ::step] = True
        previous = -1
        if self._last_goal is not None:
            row = int((self._last_goal[1] - origin[1]) / res)
            col = int((self._last_goal[0] - origin[0]) / res)
            if 0 <= row < clear.shape[0] and 0 <= col < clear.shape[1]:
                lattice[row, col] = True
                previous = row * width + col
        cells = np.flatnonzero(lattice & clear)
        places = self._visits + self._failures
        if places and cells.size:
            rows, cols = np.divmod(cells, width)
            xs = origin[0] + (cols + 0.5) * res
            ys = origin[1] + (rows + 0.5) * res
            keep = np.ones(cells.shape, dtype=bool)
            for x, y, _ in places:
                keep &= np.hypot(xs - x, ys - y) > self.penalty_radius_m
            cells = cells[keep]
        return cells, previous

    def _rays(self, res):
        """Cell offsets along 360 degrees of rays, and each ray's angle."""
        if self._ray_cache is None or self._ray_cache[0] != res:
            reach = self.coverage.view_range_m
            n_rays = int(math.ceil(2 * math.pi * reach / res)) + 1
            angles = np.linspace(-math.pi, math.pi, n_rays, endpoint=False)
            steps = np.arange(1, int(reach / (res / 2)) + 1) * (res / 2)
            dx = np.cos(angles)[:, None] * steps[None, :] / res
            dy = np.sin(angles)[:, None] * steps[None, :] / res
            self._ray_cache = (res, dx, dy, angles)
        return self._ray_cache[1:]

    def _cast(self, cell, occupied, width, height, res):
        """Flat indices of the cells seen along each ray from ``cell`` (with
        repeats), and the angle of the ray each was seen along."""
        dx, dy, angles = self._rays(res)
        row, col = divmod(cell, width)
        rows = np.floor(row + 0.5 + dy).astype(int)
        cols = np.floor(col + 0.5 + dx).astype(int)
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        r, c = np.clip(rows, 0, height - 1), np.clip(cols, 0, width - 1)
        hit = inside & occupied[r, c]
        visible = inside & ((np.cumsum(hit, axis=1) - hit) == 0)
        flat = (r * width + c)[visible]
        return flat, np.broadcast_to(angles[:, None], visible.shape)[visible]

    def _view_gain(self, cell, value, occupied, width, height, res):
        """Best (gain in m2, camera yaw) over ``n_headings`` from ``cell``."""
        flat, ray_angle = self._cast(cell, occupied, width, height, res)
        worth = value.ravel()[flat] > 0
        flat, ray_angle = flat[worth], ray_angle[worth]
        if flat.size == 0:
            return 0.0, 0.0
        best = (0.0, 0.0)
        half = self.coverage.fov_rad / 2.0
        for yaw in np.linspace(-math.pi, math.pi, self.n_headings, endpoint=False):
            off = np.abs((ray_angle - yaw + math.pi) % (2 * math.pi) - math.pi)
            cells = np.unique(flat[off <= half])
            gain = float(value.ravel()[cells].sum()) * res * res
            if gain > best[0]:
                best = (gain, float(yaw))
        return best

    def _forget_old(self, now_ns):
        self._visits = [v for v in self._visits if now_ns - v[2] < self.visit_cooldown_ns]
        self._failures = [f for f in self._failures if now_ns - f[2] < self.failure_cooldown_ns]

    @staticmethod
    def _done(seen_pct, why) -> ExplorationGoal:
        return ExplorationGoal(ExplorationGoalStatus.NO_REACHABLE_FRONTIER, None,
                               f'camera has seen {seen_pct:.0f}% of free space; {why}')


class InformationTourExplorer(InformationGainExplorer):
    """Tour of every place worth a camera view, in the shortest order (after FUEL,
    Zhou et al., RA-L 2021).

    1. Clusters: stretches of usable frontier and pieces of unseen regions, cut
       to within ``max_cluster_radius_m`` of their centre.
    2. Viewpoints: for each cluster, the nearby candidate and heading that see
       most of it, weighted by path length.
    3. Tour: shortest open route from the robot through all viewpoints
       (Held-Karp when small, else nearest neighbour + 2-opt/or-opt). The first
       viewpoint is the goal; ``tour`` holds the route for RViz.
    """

    def __init__(self, coverage: CoverageExplorer, frontier: FrontierExplorer | None = None, *,
                 max_cluster_radius_m: float | None = None, per_cluster: int = 6, **kwargs):
        """``max_cluster_radius_m`` defaults to the camera range; ``per_cluster``
        candidates nearest each centre are cast. Other arguments as for
        ``InformationGainExplorer``."""
        super().__init__(coverage, frontier, **kwargs)
        self.per_cluster = per_cluster
        self.max_cluster_radius_m = (coverage.view_range_m if max_cluster_radius_m is None
                                     else max_cluster_radius_m)
        self.tour: tuple[tuple[float, float], ...] = ()

    def select_goal(self, map_msg, robot_xy, now_ns: int) -> ExplorationGoal:
        self.tour = ()
        ctx = self._prepare(map_msg, robot_xy, now_ns)
        if isinstance(ctx, ExplorationGoal):
            return ctx
        width, height, origin, res = ctx.width, ctx.height, ctx.origin, ctx.res
        cluster, areas = self._clusters(ctx.value, ctx.unknown, res)
        k = areas.size
        cells = ctx.cells[self._bound_m2(ctx.value, ctx.cells, res) >= self.min_gain_m2]
        if k == 0 or cells.size == 0:
            return self._done(ctx.seen_pct, 'nothing left to see')
        cells = self._near_clusters(cells, cluster, k, width, ctx.previous)

        # Each cluster's viewpoint: most of it in view, discounted by path length.
        need = np.minimum(self.min_gain_m2, 0.5 * areas)
        best_u, best_gain = np.zeros(k), np.zeros(k)
        best_cell, best_yaw = np.full(k, -1), np.zeros(k)
        path_m = ctx.distance.ravel() * res
        for cell in cells:
            gains, yaws = self._cluster_gains(
                int(cell), ctx.value, cluster, k, ctx.occupied, width, height, res)
            utility = gains * math.exp(-self.lambda_per_m * path_m[cell])
            better = (gains >= need) & (utility > best_u)
            best_u[better], best_gain[better] = utility[better], gains[better]
            best_cell[better], best_yaw[better] = cell, yaws[better]
        seen = np.flatnonzero(best_cell >= 0)
        if seen.size == 0:
            return self._done(ctx.seen_pct, f'no viewpoint shows {self.min_gain_m2:g} m2')

        # Shortest open route from the robot through every cluster's viewpoint.
        viewpoints = best_cell[seen]
        start_m = path_m[viewpoints]
        between_m = shortest_path(_grid_graph(ctx.free), directed=False, unweighted=True,
                                  indices=viewpoints)[:, viewpoints] * res
        order = _open_tour(start_m, between_m)
        route = seen[order]
        length_m = float(start_m[order[0]] + between_m[order[:-1], order[1:]].sum())

        flat_cluster = cluster.ravel()
        members = np.argsort(flat_cluster, kind='stable')
        members = members[flat_cluster[members] >= 0]
        split = np.split(members, np.flatnonzero(np.diff(flat_cluster[members])) + 1)
        records = {}
        for position, c in enumerate(route):
            goal = _world(int(best_cell[c]), width, origin, res)
            points = tuple(_world(int(i), width, origin, res) for i in split[c])
            records[c] = Frontier(points, float(best_gain[c]), goal,
                                  float(path_m[best_cell[c]]), float(position), '')
        self.frontiers = tuple(records[c] for c in route)
        self.tour = (tuple(robot_xy),) + tuple(records[c].goal for c in route)

        first = route[0]
        self.chosen = records[first]
        self._last_goal = self.chosen.goal
        yaw = float(best_yaw[first])
        goal = FrontierExplorer._pose(self.chosen.goal, robot_xy, self.frame_id)
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        return ExplorationGoal(
            ExplorationGoalStatus.READY, goal,
            f'tour: 1st of {route.size} clusters, {self.chosen.length_m:.0f} m2 in view, '
            f'{self.chosen.path_m:.1f} m away, tour {length_m:.0f} m ({ctx.seen_pct:.0f}% seen)')

    def _clusters(self, value, unknown, res):
        """Cluster index per cell (-1 for none) and each cluster's information in m2."""
        width = value.shape[1]
        radius = self.max_cluster_radius_m / res
        weights = value.ravel()
        pieces = []
        if self._usable is not None and self._usable.any():
            labels, _ = ndimage.label(self._usable, structure=np.ones((3, 3), dtype=bool))
            stretches = []
            for group in _groups(labels):
                stretches += _bisect(group, width, radius)
            owner = np.full(value.size, -1)
            for i, stretch in enumerate(stretches):
                owner[stretch] = i
            _, (rows, cols) = ndimage.distance_transform_edt(~self._usable, return_indices=True)
            nearest = owner[(rows * width + cols).ravel()]
            info_unknown = np.flatnonzero(unknown.ravel() & (weights > 0))
            by_stretch = nearest[info_unknown]
            order = np.argsort(by_stretch, kind='stable')
            cells, ids = info_unknown[order], by_stretch[order]
            pieces += np.split(cells, np.flatnonzero(np.diff(ids)) + 1) if cells.size else []
        labels, _ = ndimage.label((weights > 0).reshape(value.shape) & ~unknown,
                                  structure=np.ones((3, 3), dtype=bool))
        for region in _groups(labels):
            pieces += _bisect(region, width, radius)
        cluster = np.full(value.size, -1)
        areas = []
        for piece in pieces:
            area = weights[piece].sum() * res * res
            if area >= self.min_gain_m2:
                cluster[piece] = len(areas)
                areas.append(area)
        return cluster.reshape(value.shape), np.array(areas)

    def _near_clusters(self, cells, cluster, k, width, previous):
        """The ``per_cluster`` candidates nearest each cluster's centre, and the
        previous goal: casting every candidate is too slow for the main loop."""
        flat = cluster.ravel()
        members = np.flatnonzero(flat >= 0)
        rows, cols = np.divmod(members, width)
        count = np.bincount(flat[members], minlength=k)
        centre_r = np.bincount(flat[members], weights=rows, minlength=k) / count
        centre_c = np.bincount(flat[members], weights=cols, minlength=k) / count
        cr, cc = np.divmod(cells, width)
        gap = np.hypot(cr[None, :] - centre_r[:, None], cc[None, :] - centre_c[:, None])
        n = min(self.per_cluster, cells.size)
        chosen = np.unique(np.argpartition(gap, n - 1, axis=1)[:, :n])
        keep = cells[chosen]
        if previous in cells and previous not in keep:
            keep = np.append(keep, previous)
        return keep

    def _cluster_gains(self, cell, value, cluster, k, occupied, width, height, res):
        """Per cluster, the best gain in m2 and camera yaw from ``cell``."""
        flat, _ = self._cast(cell, occupied, width, height, res)
        flat = np.unique(flat)
        ids = cluster.ravel()[flat]
        flat, ids = flat[ids >= 0], ids[ids >= 0]
        if flat.size == 0:
            return np.zeros(k), np.zeros(k)
        n = self.n_headings
        row, col = divmod(cell, width)
        rows, cols = np.divmod(flat, width)
        angle = np.arctan2(rows - row, cols - col)
        sector = np.floor((angle + math.pi) / (2 * math.pi) * n).astype(int) % n
        per_sector = np.bincount(sector * k + ids, weights=value.ravel()[flat],
                                 minlength=n * k).reshape(n, k)
        # Heading h points at the boundary between sectors h-1 and h, so a
        # field of view of ``span`` sectors covers h - span//2 ... h + span - span//2 - 1.
        span = max(1, int(round(self.coverage.fov_rad / (2 * math.pi) * n)))
        window = sum(np.roll(per_sector, -offset, axis=0)
                     for offset in range(-(span // 2), span - span // 2))
        headings = -math.pi + np.arange(n) * 2 * math.pi / n
        return window.max(axis=0) * res * res, headings[window.argmax(axis=0)]


def _groups(labels):
    """Flat cell indices of each labelled group (label 0 is background)."""
    flat = labels.ravel()
    cells = np.flatnonzero(flat)
    if cells.size == 0:
        return []
    cells = cells[np.argsort(flat[cells], kind='stable')]
    return np.split(cells, np.flatnonzero(np.diff(flat[cells])) + 1)


def _bisect(cells, width, max_radius):
    """Split flat ``cells`` along their principal axis until each piece lies
    within ``max_radius`` cells of its centre."""
    done, todo = [], [cells]
    while todo:
        piece = todo.pop()
        rows, cols = np.divmod(piece, width)
        points = np.column_stack([cols, rows]).astype(float)
        centre = points.mean(axis=0)
        offset = points - centre
        if piece.size < 2 or np.max(np.hypot(offset[:, 0], offset[:, 1])) <= max_radius:
            done.append(piece)
            continue
        _, vectors = np.linalg.eigh(offset.T @ offset)
        side = offset @ vectors[:, -1] < 0
        if side.all() or not side.any():
            done.append(piece)
            continue
        todo += [piece[side], piece[~side]]
    return done


def _grid_graph(free):
    """4-connected graph of free cells, nodes numbered by flat cell index."""
    height, width = free.shape
    index = np.arange(height * width).reshape(height, width)
    across = free[:, :-1] & free[:, 1:]
    up = free[:-1, :] & free[1:, :]
    a = np.concatenate([index[:, :-1][across], index[:-1, :][up]])
    b = np.concatenate([index[:, 1:][across], index[1:, :][up]])
    return csr_matrix((np.ones(a.size), (a, b)), shape=(height * width, height * width))


def _open_tour(start_m, between_m, exact_up_to=8):
    """Visiting order (indices) of the shortest open route from the robot
    through every point. ``start_m``: robot to each point; ``between_m``:
    point to point. Exact (Held-Karp) for up to ``exact_up_to`` points;
    beyond that nearest neighbour, then 2-opt and or-opt moves until none
    shortens it."""
    k = len(start_m)
    if k <= exact_up_to:
        return _held_karp(np.asarray(start_m, float), np.asarray(between_m, float))
    # Nodes: 0 the robot, 1..k the points, k + 1 a free end, so the open
    # route is a path between two fixed ends and every move has a local cost.
    cost = np.zeros((k + 2, k + 2))
    cost[0, 1:k + 1] = cost[1:k + 1, 0] = start_m
    cost[1:k + 1, 1:k + 1] = between_m
    cost[0, k + 1] = cost[k + 1, 0] = np.inf

    route, left = [0], set(range(1, k + 1))
    while left:
        nearest = min(left, key=lambda j: cost[route[-1], j])
        route.append(nearest)
        left.remove(nearest)
    route = np.array(route + [k + 1])
    inner = np.arange(1, k + 1)
    while True:
        # 2-opt: reverse route[i..j], 1 <= i < j <= k.
        i, j = inner[:, None], inner[None, :]
        r = route
        two = (cost[r[i - 1], r[j]] + cost[r[i], r[j + 1]]
               - cost[r[i - 1], r[i]] - cost[r[j], r[j + 1]])
        two[j <= i] = np.inf
        best_two = np.unravel_index(np.argmin(two), two.shape)
        gain, move = two[best_two], ('2opt', best_two[0] + 1, best_two[1] + 1)
        # Or-opt: move route[i..i+size-1] between route[p] and route[p+1].
        for size in (1, 2, 3):
            if size > k:
                break
            starts = np.arange(1, k - size + 2)[:, None]
            ends = starts + size - 1
            p = np.arange(0, k + 1)[None, :]
            removed = (cost[r[starts - 1], r[ends + 1]]
                       - cost[r[starts - 1], r[starts]] - cost[r[ends], r[ends + 1]])
            base = -cost[r[p], r[p + 1]]
            ahead = removed + base + cost[r[p], r[starts]] + cost[r[ends], r[p + 1]]
            behind = removed + base + cost[r[p], r[ends]] + cost[r[starts], r[p + 1]]
            invalid = (p >= starts - 1) & (p <= ends)
            for flip, delta in ((False, ahead), (True, behind)):
                delta = np.where(invalid, np.inf, delta)
                at = np.unravel_index(np.argmin(delta), delta.shape)
                if delta[at] < gain:
                    gain = delta[at]
                    move = ('or', int(starts[at[0], 0]), int(p[0, at[1]]), size, flip)
        if not gain < -1e-9:
            break
        if move[0] == '2opt':
            _, a, b = move
            route[a:b + 1] = route[a:b + 1][::-1].copy()
        else:
            _, start, after, size, flip = move
            run = route[start:start + size]
            run = run[::-1] if flip else run
            rest = np.concatenate([route[:start], route[start + size:]])
            at = after + 1 if after < start else after + 1 - size
            route = np.concatenate([rest[:at], run, rest[at:]])
    return route[1:-1] - 1


def _held_karp(start_m, between_m):
    """Exact shortest open route by dynamic programming over visited sets."""
    k = len(start_m)
    full = 1 << k
    best = np.full((full, k), np.inf)
    parent = np.full((full, k), -1)
    for j in range(k):
        best[1 << j, j] = start_m[j]
    for visited in range(1, full):
        for j in range(k):
            if not visited & (1 << j) or visited == 1 << j:
                continue
            before = visited ^ (1 << j)
            options = best[before] + between_m[:, j]
            i = int(np.argmin(options))
            best[visited, j], parent[visited, j] = options[i], i
    j, visited, order = int(np.argmin(best[full - 1])), full - 1, []
    while j >= 0:
        order.append(j)
        visited, j = visited ^ (1 << j), int(parent[visited, j])
    return np.array(order[::-1])
