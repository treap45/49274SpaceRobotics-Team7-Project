"""Artefact localisation, association and stable IDs.

Owner: B.
Contract: repository-root Workplan.md, section 4.3.

``update`` turns one DetectionBatch into map positions, using the depth image
and the camera transform taken at the batch's capture time, and associates each
position with a track of the same class within ``match_radius_m``.

Tracks stay provisional and internal until they have ``confirm_count``
observations. Only then do they get an ``artifact_id`` and appear in
``snapshot()``, so all merging happens before an id exists and an id never
changes afterwards (Workplan 4.3). A single bad-depth observation therefore
never reaches C.

Depth (Workplan 4.3 item 2, 5.1): the depth image is the same 720 x 480 sensor
as the RGB image, so a bbox indexes it directly. Valid depths in the central
half of the box are split into depth bands. The nearest supported band supplies
both depth and pixel coordinates, keeping wall depth off the object ray.
Readings outside the 0.4-10 m depth clip are invalid; a detection with too few
valid pixels is skipped with a reason rather than placed at a clipped depth.

Frames (Workplan 5): unprojection gives a point in the camera *optical* frame
(z forward, x right, y down). ``transform_at_stamp`` must therefore be
``map <- camera_rgb_optical_frame``; the image header says ``camera_link``,
which is rotated relative to it, and is rejected.
"""

from dataclasses import dataclass
import math
from typing import List, Optional

import numpy as np

from .interfaces import ArtifactTrack, DetectionBatch

DEPTH_MIN_M = 0.4     # rgbd_camera depth clip (Workplan 5.1)
DEPTH_MAX_M = 10.0

# Intrinsics of the simulated rgbd_camera, derived from the URDF (Workplan 5.1):
# 720 x 480 pixels, horizontal FOV 2.0944 rad.
SIM_CAMERA_MODEL = {
    'fx': 360.0 / math.tan(2.0944 / 2.0),
    'fy': 360.0 / math.tan(2.0944 / 2.0),
    'cx': 360.0,
    'cy': 240.0,
}


@dataclass
class _Track:
    """Internal, mutable track. ``artifact_id`` stays None until confirmed."""

    class_name: str
    position_map: tuple
    last_seen_ns: int
    observation_count: int
    artifact_id: Optional[int] = None

    def public(self) -> ArtifactTrack:
        return ArtifactTrack(
            artifact_id=self.artifact_id,
            class_name=self.class_name,
            position_map=self.position_map,
            last_seen_ns=self.last_seen_ns,
            observation_count=self.observation_count,
            confirmed=True,
        )


class ArtifactTracker:
    """Maintain persistent artefact tracks."""

    def __init__(
        self,
        confirm_count: int = 3,
        match_radius_m: float = 2.5,
        min_valid_depth_fraction: float = 0.2,
        provisional_timeout_sec: float = 60.0,
        depth_band_width_m: float = 0.3,
        min_foreground_fraction: float = 0.05,
    ):
        """``match_radius_m`` must stay below the spacing of same-class
        artefacts or two of them merge: mars_cave.sdf has three per class, at
        least 10 m apart. At 1.0 m, large artefacts seen from different sides
        split into several tracks in simulation (one green_alien became three,
        their estimates up to 1.7 m apart).
        ``provisional_timeout_sec`` drops unconfirmed tracks that were never
        seen again, which is what most bad-depth observations become."""
        if depth_band_width_m <= 0 or not math.isfinite(depth_band_width_m):
            raise ValueError('depth_band_width_m must be finite and positive')
        if not 0 < min_foreground_fraction <= 1:
            raise ValueError('min_foreground_fraction must be in (0, 1]')
        self.depth_band_width_m = depth_band_width_m
        self.min_foreground_fraction = min_foreground_fraction
        self.confirm_count = confirm_count
        self.match_radius_m = match_radius_m
        self.min_valid_depth_fraction = min_valid_depth_fraction
        self.provisional_timeout_ns = int(provisional_timeout_sec * 1e9)

        self._tracks: List[_Track] = []
        self._next_artifact_id = 1
        self._latest_stamp_ns: Optional[int] = None
        # Why each detection of the last batch was skipped, for logging.
        self.last_rejections: List[str] = []

    def reset(self) -> None:
        """Forget every track, e.g. after the simulation was reset."""
        self._tracks = []
        self._next_artifact_id = 1
        self._latest_stamp_ns = None
        self.last_rejections = []

    def snapshot(self) -> List[ArtifactTrack]:
        """Confirmed artefacts only; provisional tracks have no id yet."""
        return [track.public() for track in self._tracks
                if track.artifact_id is not None]

    def update(
        self,
        detections: DetectionBatch,
        matched_depth,
        camera_model,
        transform_at_stamp,
    ) -> List[ArtifactTrack]:
        """Update tracks from one detection batch and return ``snapshot()``.

        ``matched_depth`` is the depth image (metres, float) captured with the
        batch's RGB image. ``camera_model`` has ``fx``, ``fy``, ``cx``, ``cy``
        as keys or attributes. ``transform_at_stamp`` is a ``TransformStamped``
        (or the equivalent dict) for ``map <- camera_rgb_optical_frame`` at
        ``detections.stamp_ns``.
        """
        stamp_ns = detections.stamp_ns
        if self._latest_stamp_ns is not None and stamp_ns < self._latest_stamp_ns:
            # Time went backwards: the simulation was reset (Workplan 4.1).
            self.reset()
        self._latest_stamp_ns = stamp_ns
        self.last_rejections = []

        transform = self._as_transform_dict(transform_at_stamp)
        depth = np.asarray(matched_depth, dtype=np.float32)

        for detection in detections.detections:
            sample, reason = self._depth_in_bbox(depth, detection.bbox_xyxy)
            if sample is None:
                self.last_rejections.append(f'{detection.class_name}: {reason}')
                continue
            depth_m, u, v = sample
            point_camera = self._pixel_to_camera(u, v, depth_m, camera_model)
            position_map = self._transform_point(point_camera, transform)
            self.update_observation(detection.class_name, position_map, stamp_ns)

        self._drop_stale_provisional(stamp_ns)
        return self.snapshot()

    def update_observation(
        self,
        class_name: str,
        position_map,
        stamp_ns: int,
    ) -> Optional[ArtifactTrack]:
        """Associate one map observation; return the track once confirmed."""
        position_map = tuple(float(c) for c in position_map)
        track = self._find_matching_track(class_name, position_map)
        if track is None:
            track = _Track(class_name, position_map, stamp_ns, 1)
            self._tracks.append(track)
        else:
            # Running mean of every observation of this artefact.
            n = track.observation_count
            track.position_map = tuple(
                (old * n + new) / (n + 1)
                for old, new in zip(track.position_map, position_map))
            track.observation_count = n + 1
            track.last_seen_ns = max(track.last_seen_ns, stamp_ns)

        if track.artifact_id is None and track.observation_count >= self.confirm_count:
            track.artifact_id = self._allocate_id()
        return track.public() if track.artifact_id is not None else None

    def _allocate_id(self) -> int:
        artifact_id = self._next_artifact_id
        self._next_artifact_id += 1
        return artifact_id

    def _find_matching_track(self, class_name: str, position_map) -> Optional[_Track]:
        """Nearest track of the same class within ``match_radius_m``."""
        best_track = None
        best_distance = self.match_radius_m
        for track in self._tracks:
            if track.class_name != class_name:
                continue
            distance = math.dist(track.position_map, position_map)
            if distance <= best_distance:
                best_distance = distance
                best_track = track
        return best_track

    def _drop_stale_provisional(self, now_ns: int) -> None:
        self._tracks = [
            track for track in self._tracks
            if track.artifact_id is not None
            or now_ns - track.last_seen_ns <= self.provisional_timeout_ns]

    def _depth_in_bbox(self, depth, bbox_xyxy):
        """Nearest supported depth band in the central half of the box.

        Group sorted valid depths into bands spanning at most ``depth_band_width_m``.
        Bounding the total span prevents sparse edge readings bridging to a wall.
        Require at least four pixels and ``min_foreground_fraction`` of the
        sampling window, so isolated near readings do not win over an object.
        Return ``((depth_m, u, v), None)`` from the same band, or ``(None, reason)``.
        A supported nearer occluder cannot be distinguished from the target
        using depth alone; this is a foreground heuristic, not segmentation.
        """
        height, width = depth.shape[:2]
        x_min, y_min, x_max, y_max = bbox_xyxy
        quarter_w = (x_max - x_min) / 4.0
        quarter_h = (y_max - y_min) / 4.0
        # Right and bottom edges are exclusive (Workplan 4.1).
        u0 = max(0, int(math.floor(x_min + quarter_w)))
        u1 = min(width, int(math.ceil(x_max - quarter_w)))
        v0 = max(0, int(math.floor(y_min + quarter_h)))
        v1 = min(height, int(math.ceil(y_max - quarter_h)))
        if u1 <= u0 or v1 <= v0:
            return None, 'bounding box has no pixels inside the depth image'

        window = depth[v0:v1, u0:u1]
        rows, cols = np.nonzero(np.isfinite(window)
                                & (window >= DEPTH_MIN_M) & (window < DEPTH_MAX_M))
        valid = window[rows, cols]
        if valid.size < self.min_valid_depth_fraction * window.size:
            return None, (f'{valid.size}/{window.size} valid depth pixels '
                          f'(outside {DEPTH_MIN_M:g}-{DEPTH_MAX_M:g} m or missing)')
        order = np.argsort(valid)
        sorted_depths = valid[order]
        min_support = max(4, math.ceil(self.min_foreground_fraction * window.size))
        start = 0
        while start < order.size:
            end = int(np.searchsorted(sorted_depths,
                                      sorted_depths[start] + self.depth_band_width_m,
                                      side='right'))
            band = order[start:end]
            start = end
            if band.size >= min_support:
                # Pixel centres follow the exclusive bbox edge convention.
                return (float(np.median(valid[band])),
                        float(u0 + np.median(cols[band]) + 0.5),
                        float(v0 + np.median(rows[band]) + 0.5)), None
        return None, 'no depth band has enough foreground support'

    def _pixel_to_camera(self, u: float, v: float, depth_m: float, camera_model):
        """Depth pixel to a point in the camera optical frame."""
        fx, fy, cx, cy = (self._camera_value(camera_model, key)
                          for key in ('fx', 'fy', 'cx', 'cy'))
        return ((u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m)

    @staticmethod
    def _camera_value(camera_model, key):
        if isinstance(camera_model, dict):
            return float(camera_model[key])
        return float(getattr(camera_model, key))

    @staticmethod
    def _as_transform_dict(transform):
        """Accept a ``TransformStamped`` or an already converted dict."""
        if isinstance(transform, dict):
            return transform
        child = transform.child_frame_id
        if child and not child.endswith('optical_frame'):
            raise ValueError(
                f'transform is to {child!r}; unprojected points are in the optical '
                'frame, so use map <- camera_rgb_optical_frame (Workplan 5)')
        t = transform.transform.translation
        q = transform.transform.rotation
        return {'translation': (t.x, t.y, t.z), 'rotation': (q.x, q.y, q.z, q.w)}

    @staticmethod
    def _transform_point(point_camera, transform):
        """Rotate by the quaternion, then translate."""
        x, y, z = point_camera
        tx, ty, tz = transform['translation']
        qx, qy, qz, qw = transform['rotation']

        r00 = 1 - 2 * (qy * qy + qz * qz)
        r01 = 2 * (qx * qy - qz * qw)
        r02 = 2 * (qx * qz + qy * qw)
        r10 = 2 * (qx * qy + qz * qw)
        r11 = 1 - 2 * (qx * qx + qz * qz)
        r12 = 2 * (qy * qz - qx * qw)
        r20 = 2 * (qx * qz - qy * qw)
        r21 = 2 * (qy * qz + qx * qw)
        r22 = 1 - 2 * (qx * qx + qy * qy)

        return (r00 * x + r01 * y + r02 * z + tx,
                r10 * x + r11 * y + r12 * z + ty,
                r20 * x + r21 * y + r22 * z + tz)
