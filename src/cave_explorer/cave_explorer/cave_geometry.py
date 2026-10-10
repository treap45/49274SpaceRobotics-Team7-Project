"""Online cave geometry using numpy/scipy only; coordinates are (row, column).

The ridge is a distance-gradient approximation, not a mathematical skeleton.
Unknown cells and the exterior are clearance boundaries, never confirmed walls.
"""

from dataclasses import dataclass
import math
import time

import numpy as np
from scipy import ndimage


@dataclass
class GeometryResult:
    """Geometry in grid coordinates, with all distances in metres."""

    distance_map: np.ndarray
    reachable_mask: np.ndarray
    ridge_mask: np.ndarray
    widest_position_rc: tuple | None = None
    widest_radius_m: float | None = None
    bottleneck_position_rc: tuple | None = None
    bottleneck_width_m: float | None = None
    bottleneck_wall_points_rc: tuple = ()
    saddle_candidates: tuple = ()
    processing_time_s: float = 0.0
    reason: str = ''


def grid_to_world(rc, resolution, origin=(0.0, 0.0), yaw=0.0):
    """Return the cell centre, respecting the occupancy-grid origin rotation."""
    x, y = (rc[1] + 0.5) * resolution, (rc[0] + 0.5) * resolution
    c, s = math.cos(yaw), math.sin(yaw)
    return origin[0] + c * x - s * y, origin[1] + s * x + c * y


def world_to_grid(xy, resolution, origin=(0.0, 0.0), yaw=0.0):
    """Inverse origin transform; floor also handles points below the origin."""
    x, y = xy[0] - origin[0], xy[1] - origin[1]
    c, s = math.cos(yaw), math.sin(yaw)
    return math.floor((-s * x + c * y) / resolution), math.floor((c * x + s * y) / resolution)


def analyze_geometry(grid, resolution, robot_rc, robot_radius_m=0.5,
                     ridge_threshold=0.5, search_radius_m=1.5,
                     widening_ratio=1.5, widening_distance_m=12.0):
    """Analyse a snapshot, restricting all reported features to robot reachability.

    Four-connectivity prevents corner cutting. Local ridge-minimum plateaus are
    represented once. A candidate needs opposite occupied walls, and cutting a
    disk of its clearance out of the reachable space within 12 m must leave two
    or more parts that each widen beyond ``widening_ratio`` times its clearance.
    This holds for straight, diagonal and curved passages, and rejects dead-end
    tips and gaps beside an obstacle that the robot can drive round.
    """
    started = time.perf_counter()
    grid = np.asarray(grid)
    if grid.ndim != 2 or not grid.size or resolution <= 0 or not math.isfinite(resolution):
        raise ValueError('expected nonempty 2-D grid and positive finite resolution')
    if (not math.isfinite(robot_radius_m) or robot_radius_m < 0
            or not 0 < ridge_threshold < 1 or search_radius_m <= 0
            or widening_ratio <= 1 or widening_distance_m <= 0):
        raise ValueError('invalid geometry thresholds')
    free = (grid >= 0) & (grid < 50)
    # Explicit padding avoids EDT's asymmetric implicit boundary on all-free maps.
    padded = np.pad(free, 1, constant_values=False)
    distance, nearest = ndimage.distance_transform_edt(padded, return_indices=True)
    distance = distance[1:-1, 1:-1] * resolution
    nearest = nearest[:, 1:-1, 1:-1] - 1
    reachable = np.zeros(grid.shape, dtype=bool)
    result = GeometryResult(distance, reachable, reachable.copy())

    def finish(reason=''):
        result.reason = reason
        result.processing_time_s = time.perf_counter() - started
        return result

    if robot_rc is None:
        return finish('robot TF unavailable')
    r, c = robot_rc
    if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
        return finish('robot outside map')
    labels, _ = ndimage.label(free & (distance > robot_radius_m))
    component = labels[r, c]
    if not component:
        return finish('robot cell unknown, occupied, or insufficient clearance')
    reachable[:] = labels == component
    widest = np.unravel_index(np.argmax(np.where(reachable, distance, -1)), grid.shape)
    result.widest_position_rc = tuple(int(v) for v in widest)
    result.widest_radius_m = float(distance[widest])
    # Gradients in metric coordinates: a plane distance field has magnitude 1.
    field = np.pad(distance, 1)
    gy, gx = np.gradient(field, resolution)
    magnitude = np.hypot(gy[1:-1, 1:-1], gx[1:-1, 1:-1])
    ridge = reachable & (magnitude <= ridge_threshold + 1e-9)
    result.ridge_mask = ridge
    window = 2 * max(1, math.ceil(search_radius_m / resolution)) + 1
    minimum = ndimage.minimum_filter(np.where(ridge, distance, np.inf), size=window)
    minima = ridge & (distance <= minimum + 1e-9)
    groups, count = ndimage.label(minima, structure=np.ones((3, 3)))
    candidates = []
    walls = {}
    # Iterate candidate plateaus, not every map cell. Slices bound local work.
    for index, region in enumerate(ndimage.find_objects(groups), start=1):
        cells = np.argwhere(groups[region] == index)
        cells += np.array([part.start for part in region])
        centre = cells.mean(axis=0)
        p = tuple(cells[np.argmin(np.sum((cells - centre) ** 2, axis=1))])
        boundary = nearest[:, p[0], p[1]]
        normal = boundary - p
        norm = np.linalg.norm(normal)
        if not norm:
            continue
        normal = normal / norm
        # Trace both wall normals. First non-free cell must be observed occupied.
        pair = []
        for sign in (-1, 1):
            steps = np.arange(0.5, norm * 2 + 3, 0.5)
            points = np.rint(np.asarray(p) + sign * steps[:, None] * normal).astype(int)
            inside = ((points >= 0) & (points < grid.shape)).all(axis=1)
            values = np.full(len(points), -1)
            values[inside] = grid[tuple(points[inside].T)]
            hits = np.flatnonzero((values < 0) | (values >= 50))
            if not hits.size or values[hits[0]] < 50:
                break
            pair.append(tuple(points[hits[0]]))
        if len(pair) != 2:
            continue
        # A bottleneck separates nearby reachable space: cut a disk at p and
        # require two or more of the remaining local parts to widen.
        rr, cc = np.ogrid[:grid.shape[0], :grid.shape[1]]
        near = (rr - p[0]) ** 2 + (cc - p[1]) ** 2
        cut = near <= (distance[p] / resolution + 1.5) ** 2
        local = near <= (widening_distance_m / resolution) ** 2
        parts, _ = ndimage.label(reachable & local & ~cut)
        rim = ndimage.binary_dilation(cut) & ~cut
        wider_sides = [side for side in set(np.unique(parts[rim])) - {0}
                       if np.any(distance[parts == side] > widening_ratio * distance[p])]
        if len(wider_sides) >= 2:
            candidates.append(p)
            walls[p] = tuple(pair)
    result.saddle_candidates = tuple(candidates)
    if candidates:
        best = min(candidates, key=lambda p: distance[p])
        result.bottleneck_position_rc = best
        result.bottleneck_width_m = float(2 * distance[best])
        result.bottleneck_wall_points_rc = walls[best]
    return finish()
