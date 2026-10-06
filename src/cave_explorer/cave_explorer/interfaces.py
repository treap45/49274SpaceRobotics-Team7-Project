"""Shared data contracts for the cave explorer project."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Detection2D:
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]


@dataclass
class DetectionBatch:
    stamp_ns: int
    source_frame_id: str
    image_width: int
    image_height: int
    detections: List[Detection2D]


@dataclass
class ArtifactTrack:
    artifact_id: int
    class_name: str
    position_map: Tuple[float, float, float]
    last_seen_ns: int
    observation_count: int
    confirmed: bool
