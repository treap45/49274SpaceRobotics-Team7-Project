"""Artefact shapes for Nav2's costmaps, from the depth pixels inside detection boxes.

Owner: C.

The lidar is mounted 1.0 m up (``scan_joint`` in mars_explorer.urdf.xacro).
green_crystals and ice_formation are about 5 m across at the base but narrow at
lidar height, so the costmap holds only a few spikes and Nav2 plans across the
base. In simulation the robot sat stuck for 3.5 min on the green_crystals near
the start.

Each detection's box is sampled on a pixel grid; its depth pixels are
unprojected (optical frame, Workplan 5) and transformed to ``map``, and points
between ``min_height_m`` and ``max_height_m`` are kept as occupied voxels. The
voxels accumulate over the run and are published as one point cloud that Nav2
marks as obstacles. Using only pixels inside boxes keeps the floor and the
cave walls - which the lidar already has - out of the cloud.

The depth cloud cannot hold parts below ``min_height_m``, yet the chassis
clears the floor by only ~6 cm (1.0 x 1.0 m box at z = 0.047 m on 0.033 m
wheels). green_crystals' shards are 0-0.15 m tall out to 2.66 m from the model
origin (mesh in worlds/models), and the robot stuck on them again with the
depth cloud running. ``keep_out_points`` therefore fills a disk of
``KEEP_OUT_RADIUS_M`` round each confirmed artefact of such a type.
"""

import math

import numpy as np

from cave_explorer.artifact_tracker import ArtifactTracker, DEPTH_MAX_M, DEPTH_MIN_M
from cave_explorer.interfaces import GREEN_CRYSTALS

# Footprint radius per artefact type whose low parts the depth cloud misses.
# ice_formation's lowest part is 0.14 m, so the depth cloud already has it.
KEEP_OUT_RADIUS_M = {GREEN_CRYSTALS: 2.7}


class ArtifactObstacles:
    """Accumulates occupied voxels of detected artefacts in the ``map`` frame."""

    def __init__(self, *,
                 voxel_m: float = 0.15,
                 pixel_step: int = 4,
                 min_height_m: float = 0.15,
                 max_height_m: float = 2.0,
                 max_range_m: float = 6.0):
        """``min_height_m`` sits above the floor (z = 0.04 m in map in simulation).
        ``max_range_m`` keeps to where depth is accurate enough to draw a shape."""
        self.voxel_m = voxel_m
        self.pixel_step = pixel_step
        self.min_height_m = min_height_m
        self.max_height_m = max_height_m
        self.max_range_m = min(max_range_m, DEPTH_MAX_M)
        self._voxels = set()

    def __len__(self):
        return len(self._voxels)

    def reset(self) -> None:
        self._voxels = set()

    def add(self, detections, depth, camera_model, transform) -> int:
        """Add the voxels seen inside every box of ``detections``; return how many were new.

        ``depth`` is the depth image captured with the detections (metres),
        ``transform`` is ``map <- camera_rgb_optical_frame`` at that time, as a
        ``TransformStamped`` or the dict ``ArtifactTracker`` uses.
        """
        depth = np.asarray(depth, dtype=np.float32)
        height, width = depth.shape[:2]
        transform = ArtifactTracker._as_transform_dict(transform)
        fx, fy, cx, cy = (ArtifactTracker._camera_value(camera_model, key)
                          for key in ('fx', 'fy', 'cx', 'cy'))
        rotation = _rotation(transform['rotation'])
        translation = np.asarray(transform['translation'], dtype=np.float64)

        before = len(self._voxels)
        for detection in detections.detections:
            x_min, y_min, x_max, y_max = detection.bbox_xyxy
            us = np.arange(max(0, int(x_min)), min(width, int(math.ceil(x_max))), self.pixel_step)
            vs = np.arange(max(0, int(y_min)), min(height, int(math.ceil(y_max))), self.pixel_step)
            if us.size == 0 or vs.size == 0:
                continue
            u, v = np.meshgrid(us, vs)
            d = depth[v, u]
            valid = np.isfinite(d) & (d >= DEPTH_MIN_M) & (d < self.max_range_m)
            if not valid.any():
                continue
            u, v, d = u[valid], v[valid], d[valid].astype(np.float64)
            optical = np.stack(((u - cx) * d / fx, (v - cy) * d / fy, d), axis=1)
            points = optical @ rotation.T + translation
            keep = (points[:, 2] >= self.min_height_m) & (points[:, 2] <= self.max_height_m)
            cells = np.floor(points[keep] / self.voxel_m).astype(np.int64)
            self._voxels.update(map(tuple, cells))
        return len(self._voxels) - before

    def points(self) -> np.ndarray:
        """Voxel centres as an (N, 3) float32 array in ``map``."""
        if not self._voxels:
            return np.zeros((0, 3), dtype=np.float32)
        cells = np.array(sorted(self._voxels), dtype=np.float64)
        return ((cells + 0.5) * self.voxel_m).astype(np.float32)


def keep_out_points(tracks, robot_xy, *,
                    spacing_m: float = 0.15,
                    height_m: float = 0.5,
                    robot_clear_m: float = 1.0) -> np.ndarray:
    """A filled disk of points round every track whose class has a ``KEEP_OUT_RADIUS_M``.

    A disk is shrunk to stay ``robot_clear_m`` from ``robot_xy`` (or dropped),
    so a robot already inside one is never marked as in collision. Returns an
    (N, 3) float32 array in ``map`` at ``height_m``, inside Nav2's obstacle band.
    """
    disks = []
    for track in tracks:
        radius = KEEP_OUT_RADIUS_M.get(track.class_name)
        if radius is None:
            continue
        cx, cy = track.position_map[0], track.position_map[1]
        if robot_xy is not None:
            radius = min(radius, math.dist(robot_xy, (cx, cy)) - robot_clear_m)
        if radius <= 0:
            continue
        steps = np.arange(-radius, radius + 1e-9, spacing_m)
        dx, dy = np.meshgrid(steps, steps)
        inside = dx ** 2 + dy ** 2 <= radius ** 2
        disks.append(np.stack((cx + dx[inside], cy + dy[inside],
                               np.full(inside.sum(), height_m)), axis=1))
    if not disks:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(disks).astype(np.float32)


def _rotation(quaternion):
    x, y, z, w = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
