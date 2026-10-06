
"""Artifact tracking and stable ID management."""

from typing import List,Optional

from .interfaces import ArtifactTrack,DetectionBatch


class ArtifactTracker:
    """Maintain persistent artefact tracks."""
    def _transform_stamped_to_dict(self, transform_stamped):
        """Convert ROS TransformStamped to internal transform dictionary."""

        t = transform_stamped.transform.translation
        q = transform_stamped.transform.rotation

        return {
            "translation": (
                t.x,
                t.y,
                t.z
            ),
            "rotation": (
                q.x,
                q.y,
                q.z,
                q.w
            )
        }
    def __init__(self):
        self._tracks: List[ArtifactTrack] = []
        self._next_artifact_id = 1

    def snapshot(self) -> List[ArtifactTrack]:
        """Return the current list of tracked artefacts."""
        return list(self._tracks)

    def _allocate_id(self) -> int:
        """Allocate a new stable artefact ID."""
        artifact_id = self._next_artifact_id
        self._next_artifact_id += 1
        return artifact_id
    def _create_track(
        self,
        class_name: str,
        position_map,
        stamp_ns: int
    ) -> ArtifactTrack:
        """Create and store a new artefact track."""

        track = ArtifactTrack(
            artifact_id=self._allocate_id(),
            class_name=class_name,
            position_map=position_map,
            last_seen_ns=stamp_ns,
            observation_count=1,
            confirmed=False
        )

        self._tracks.append(track)

        return track
    def _find_matching_track(
        self,
        class_name: str,
        position_map,
        max_distance: float = 1.0
    ) -> Optional[ArtifactTrack]:
        """Find the nearest existing track of the same class."""

        best_track = None
        best_distance = max_distance

        for track in self._tracks:
            if track.class_name != class_name:
                continue

            dx = track.position_map[0] - position_map[0]
            dy = track.position_map[1] - position_map[1]
            dz = track.position_map[2] - position_map[2]

            distance = (dx * dx + dy * dy + dz * dz) ** 0.5

            if distance <= best_distance:
                best_distance = distance
                best_track = track

        return best_track
    def _update_existing_track(
        self,
        track: ArtifactTrack,
        position_map,
        stamp_ns: int
    ) -> ArtifactTrack:
        """Update an existing track with a new observation."""

        old_count = track.observation_count
        new_count = old_count + 1

        old_x, old_y, old_z = track.position_map
        new_x, new_y, new_z = position_map

        track.position_map = (
            (old_x * old_count + new_x) / new_count,
            (old_y * old_count + new_y) / new_count,
            (old_z * old_count + new_z) / new_count
        )

        track.last_seen_ns = stamp_ns
        track.observation_count = new_count
        track.confirmed = new_count >= 3
        return track
    def update_observation(
        self,
        class_name: str,
        position_map,
        stamp_ns: int
    ) -> ArtifactTrack:
        """Associate an observation with an existing track or create a new one."""

        track = self._find_matching_track(
            class_name,
            position_map
        )

        if track is None:
            return self._create_track(
                class_name,
                position_map,
                stamp_ns
            )

        return self._update_existing_track(
            track,
            position_map,
            stamp_ns
        )
    def _pixel_to_camera(
        self,
        u: float,
        v: float,
        depth_m: float,
        camera_model
    ):
        """Convert a depth pixel into a 3D point in the camera optical frame."""

        fx = camera_model["fx"]
        fy = camera_model["fy"]
        cx = camera_model["cx"]
        cy = camera_model["cy"]

        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        z = depth_m

        return (x, y, z)
    def _bbox_center(self, bbox_xyxy):
        """Return the center pixel of an xyxy bounding box."""

        x_min, y_min, x_max, y_max = bbox_xyxy

        u = (x_min + x_max) / 2.0
        v = (y_min + y_max) / 2.0

        return (u, v)
    def _detection_to_camera_point(
        self,
        bbox_xyxy,
        depth_m: float,
        camera_model
    ):
        """Convert a detection bounding box and depth into a 3D camera point."""
        if not self._valid_depth(depth_m):
            return None
        u, v = self._bbox_center(bbox_xyxy)

        return self._pixel_to_camera(
            u,
            v,
            depth_m,
            camera_model
        )

    def _valid_depth(self, depth_m: float) -> bool:
        """Check whether a depth value is usable."""

        return (
            depth_m > 0.0
            and depth_m <= 10.0
            and depth_m == depth_m
            and depth_m != float("inf")
            and depth_m != float("-inf")
        )
    def _transform_point(self, point_camera, transform):
        """Transform a 3D point using translation and quaternion rotation."""

        x, y, z = point_camera

        tx = transform["translation"][0]
        ty = transform["translation"][1]
        tz = transform["translation"][2]

        qx = transform["rotation"][0]
        qy = transform["rotation"][1]
        qz = transform["rotation"][2]
        qw = transform["rotation"][3]

        # Quaternion rotation matrix
        r00 = 1 - 2 * (qy * qy + qz * qz)
        r01 = 2 * (qx * qy - qz * qw)
        r02 = 2 * (qx * qz + qy * qw)

        r10 = 2 * (qx * qy + qz * qw)
        r11 = 1 - 2 * (qx * qx + qz * qz)
        r12 = 2 * (qy * qz - qx * qw)

        r20 = 2 * (qx * qz - qy * qw)
        r21 = 2 * (qy * qz + qx * qw)
        r22 = 1 - 2 * (qx * qx + qy * qy)

        map_x = r00 * x + r01 * y + r02 * z + tx
        map_y = r10 * x + r11 * y + r12 * z + ty
        map_z = r20 * x + r21 * y + r22 * z + tz

        return (map_x, map_y, map_z)
    def _detection_to_map_point(
        self,
        bbox_xyxy,
        depth_m: float,
        camera_model,
        transform
    ):
        """Convert a detection into a 3D point in the map frame."""

        point_camera = self._detection_to_camera_point(
            bbox_xyxy,
            depth_m,
            camera_model
        )

        if point_camera is None:
            return None

        return self._transform_point(
            point_camera,
            transform
        )
    def update(
        self,
        detections: DetectionBatch,
        matched_depth,
        camera_model,
        transform_at_stamp
    ) -> List[ArtifactTrack]:
        """Update artifact tracks from one detection batch."""

        updated_tracks = []

        for detection, depth_m in zip(
            detections.detections,
            matched_depth
        ):
            position_map = self._detection_to_map_point(
                detection.bbox_xyxy,
                depth_m,
                camera_model,
                transform_at_stamp
            )

            if position_map is None:
                continue

            track = self.update_observation(
                detection.class_name,
                position_map,
                detections.stamp_ns
            )

            updated_tracks.append(track)

        return updated_tracks
