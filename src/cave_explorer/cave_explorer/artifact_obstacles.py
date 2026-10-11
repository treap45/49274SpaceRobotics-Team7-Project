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
wheels), and the robot still stuck on low parts of an artefact with the depth
cloud running. Two further sources, neither using any prior knowledge of the
artefacts:

* ``keep_out_points`` fills a disk round each confirmed artefact whose radius
  is half its width as measured by ``ArtifactTracker`` (box width times depth),
  capped at ``max_radius_m``. Only for the types the detector handles reliably
  (the inspection classes): in simulation a false ice_formation 4 m from any
  real artefact put a 3 m disk over two coverage goals, and Nav2 could not
  plan to either.
* ``mark_contact`` records where the robot got stuck: a patch ahead of the
  chassis, kept for the rest of the run, so Nav2 plans round it next time.
"""

import math

import numpy as np

from cave_explorer.artifact_tracker import ArtifactTracker, DEPTH_MAX_M, DEPTH_MIN_M


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
        self._contacts = set()      # voxels where the robot got stuck

    def __len__(self):
        return len(self._voxels | self._contacts)

    def reset(self) -> None:
        self._voxels = set()
        self._contacts = set()

    def mark_contact(self, x: float, y: float, yaw: float, *,
                     ahead_from_m: float = 0.3, ahead_to_m: float = 1.0,
                     half_width_m: float = 0.6, height_m: float = 0.5) -> int:
        """Mark the patch ahead of a robot at (x, y, yaw) that could not move.

        The chassis front is 0.5 m ahead of ``base_link``; whatever stopped it is
        under or just beyond it. The patch must be clear of the robot once it
        has backed off, or Nav2 would see the robot in collision. Returns how
        many voxels were new.
        """
        step = self.voxel_m / 2.0
        ahead = np.arange(ahead_from_m, ahead_to_m + 1e-9, step)
        side = np.arange(-half_width_m, half_width_m + 1e-9, step)
        a, s_ = np.meshgrid(ahead, side)
        c, n = math.cos(yaw), math.sin(yaw)
        points = np.stack((x + a * c - s_ * n, y + a * n + s_ * c,
                           np.full(a.shape, height_m)), axis=-1).reshape(-1, 3)
        before = len(self._contacts)
        self._contacts.update(map(tuple, np.floor(points / self.voxel_m).astype(np.int64)))
        return len(self._contacts) - before

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
        """Voxel centres, from detections and contacts, as an (N, 3) float32 array in ``map``."""
        voxels = self._voxels | self._contacts
        if not voxels:
            return np.zeros((0, 3), dtype=np.float32)
        cells = np.array(sorted(voxels), dtype=np.float64)
        return ((cells + 0.5) * self.voxel_m).astype(np.float32)


def keep_out_radius_m(width_m, *, max_radius_m: float = 3.0):
    """Keep-out radius for an artefact of measured width ``width_m`` (None if unknown).

    Capped so one oversized box cannot close a passage.
    """
    if not width_m or width_m <= 0.0:
        return None
    return min(0.5 * width_m, max_radius_m)


def keep_out_points(tracks, size_of, robot_xy, classes=None, *,
                    spacing_m: float = 0.15,
                    height_m: float = 0.5,
                    robot_clear_m: float = 1.0) -> np.ndarray:
    """A filled disk of points round every track, sized by its measured width.

    ``size_of(artifact_id)`` returns the width from ``ArtifactTracker.size_m``.
    ``classes``, if given, limits the disks to those artefact types.
    A disk is shrunk to stay ``robot_clear_m`` from ``robot_xy`` (or dropped),
    so a robot already inside one is never marked as in collision. Returns an
    (N, 3) float32 array in ``map`` at ``height_m``, inside Nav2's obstacle band.
    """
    disks = []
    for track in tracks:
        if classes is not None and track.class_name not in classes:
            continue
        radius = keep_out_radius_m(size_of(track.artifact_id))
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
