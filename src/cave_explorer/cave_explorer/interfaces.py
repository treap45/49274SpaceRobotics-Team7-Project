"""Shared data types between the perception, tracking and planning modules.

Conventions:
  * Distances in metres, angles in radians.
  * Map positions are in the ``map`` frame; navigation goals are ``PoseStamped``.
  * ``stamp_ns`` is the ROS (simulation) time of the observation in integer
    nanoseconds; caches are cleared if time goes backwards.
  * Bounding boxes are ``(x_min, y_min, x_max, y_max)`` in image pixels; right
    and bottom edges are exclusive.
  * Failure is a status value, never a sentinel coordinate.

Payloads are frozen dataclasses holding tuples, so a snapshot cannot be
changed by the module that receives it.
"""

from dataclasses import dataclass
from enum import Enum

from geometry_msgs.msg import PoseStamped

# --------------------------------------------------------------------------
# Label table
# --------------------------------------------------------------------------
# Same-class artefacts differ only by position, so association must not
# merge them on class alone.

GREEN_ALIEN = 'green_alien'
GREEN_CRYSTALS = 'green_crystals'
WHITE_SPHERE = 'white_sphere'
ICE_FORMATION = 'ice_formation'
BLUE_MUSHROOMS = 'blue_mushrooms'

ARTIFACT_CLASSES = frozenset({
    GREEN_ALIEN,
    GREEN_CRYSTALS,
    WHITE_SPHERE,
    ICE_FORMATION,
    BLUE_MUSHROOMS,
})


# --------------------------------------------------------------------------
# Detector -> tracker: 2D detections
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection2D:
    """One detected artefact in one image (no depth or map position)."""

    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def __post_init__(self):
        if self.class_name not in ARTIFACT_CLASSES:
            raise ValueError(
                f'unknown class_name {self.class_name!r}; '
                f'expected one of {sorted(ARTIFACT_CLASSES)}')
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f'confidence {self.confidence} outside [0.0, 1.0]')
        x_min, y_min, x_max, y_max = self.bbox_xyxy
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(
                f'empty or inverted bbox {self.bbox_xyxy}; '
                'expected (x_min, y_min, x_max, y_max) with exclusive edges')

    @property
    def centre_xy(self) -> tuple[float, float]:
        """Pixel centre of the box, for bearing estimation."""
        x_min, y_min, x_max, y_max = self.bbox_xyxy
        return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)


@dataclass(frozen=True)
class DetectionBatch:
    """Every detection from one image; ``stamp_ns`` and ``source_frame_id`` are
    the image's capture time and frame, used to look up depth and TF."""

    stamp_ns: int
    source_frame_id: str
    image_width: int
    image_height: int
    detections: tuple[Detection2D, ...] = ()

    def __post_init__(self):
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError(
                f'image size {self.image_width}x{self.image_height} '
                'must be positive')
        if not self.source_frame_id:
            raise ValueError('source_frame_id must name the camera frame')
        for detection in self.detections:
            x_min, y_min, x_max, y_max = detection.bbox_xyxy
            if x_min < 0 or y_min < 0:
                raise ValueError(
                    f'bbox {detection.bbox_xyxy} extends past the image origin')
            if x_max > self.image_width or y_max > self.image_height:
                raise ValueError(
                    f'bbox {detection.bbox_xyxy} exceeds image bounds '
                    f'{self.image_width}x{self.image_height}; '
                    'undo any model input scaling or padding first')

    def __len__(self) -> int:
        return len(self.detections)


# --------------------------------------------------------------------------
# Tracker -> behaviour manager: artefact map snapshot
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ArtifactTrack:
    """One confirmed artefact at its fused map position.

    ``artifact_id`` never changes; if two tracks are merged, the newer id is
    retired. ``last_seen_ns`` tells whether it is in view now.
    """

    artifact_id: int
    class_name: str
    position_map: tuple[float, float, float]
    last_seen_ns: int
    observation_count: int
    confirmed: bool

    def __post_init__(self):
        if self.class_name not in ARTIFACT_CLASSES:
            raise ValueError(f'unknown class_name {self.class_name!r}')
        if self.observation_count < 1:
            raise ValueError(
                'a track exists because it was observed at least once')

    def age_ns(self, now_ns: int) -> int:
        """Nanoseconds since this artefact was last seen."""
        return now_ns - self.last_seen_ns


# --------------------------------------------------------------------------
# Inspection planner: observation goals and inspection assessment
# --------------------------------------------------------------------------

class InspectionGoalStatus(Enum):
    """Outcome of ``InspectionPlanner.make_goal``."""

    READY = 'READY'
    NOT_READY = 'NOT_READY'
    NO_SAFE_GOAL = 'NO_SAFE_GOAL'


class InspectionEvaluation(Enum):
    """Outcome of ``InspectionPlanner.evaluate``.

    ``SATISFIED`` reflects recent observation plus the configured distance and
    heading conditions. A Nav2 success result alone is not an inspection.
    """

    PENDING = 'PENDING'
    SATISFIED = 'SATISFIED'
    LOST = 'LOST'


@dataclass(frozen=True)
class InspectionGoal:
    """Where the robot should stand to inspect an artefact (checked for local
    clearance; Nav2 still decides whether it is reachable)."""

    status: InspectionGoalStatus
    goal: PoseStamped | None
    reason: str

    def __post_init__(self):
        if self.status is InspectionGoalStatus.READY and self.goal is None:
            raise ValueError('READY requires a goal pose')
        if self.status is not InspectionGoalStatus.READY and self.goal is not None:
            raise ValueError(f'{self.status.value} must not carry a goal pose')
        if self.status is not InspectionGoalStatus.READY and not self.reason:
            raise ValueError(f'{self.status.value} must explain why')


@dataclass(frozen=True)
class InspectionAssessment:
    """Whether the current close-range inspection has succeeded yet."""

    status: InspectionEvaluation
    reason: str


# --------------------------------------------------------------------------
# Exploration and navigation
# --------------------------------------------------------------------------

class ExplorationGoalStatus(Enum):
    """Outcome of ``FrontierExplorer.select_goal``.

    ``NOT_READY`` means there is no usable map yet.
    ``NO_REACHABLE_FRONTIER`` means the map exists but nothing reachable is
    left to expand - which is not by itself proof the cave is fully explored.
    """

    READY = 'READY'
    NOT_READY = 'NOT_READY'
    NO_REACHABLE_FRONTIER = 'NO_REACHABLE_FRONTIER'


@dataclass(frozen=True)
class ExplorationGoal:
    """The next exploration target, or why there isn't one."""

    status: ExplorationGoalStatus
    goal: PoseStamped | None
    reason: str

    def __post_init__(self):
        if self.status is ExplorationGoalStatus.READY and self.goal is None:
            raise ValueError('READY requires a goal pose')
        if self.status is not ExplorationGoalStatus.READY and self.goal is not None:
            raise ValueError(f'{self.status.value} must not carry a goal pose')
        if self.status is not ExplorationGoalStatus.READY and not self.reason:
            raise ValueError(f'{self.status.value} must explain why')


class NavigationStatus(Enum):
    """Lifecycle of one ``NavigateToPose`` request; only ``SUCCEEDED`` means arrival."""

    PENDING = 'PENDING'
    ACTIVE = 'ACTIVE'
    SUCCEEDED = 'SUCCEEDED'
    REJECTED = 'REJECTED'
    ABORTED = 'ABORTED'
    CANCELED = 'CANCELED'
    ERROR = 'ERROR'

    @property
    def is_terminal(self) -> bool:
        """True once this request can produce no further transitions."""
        return self in _TERMINAL_NAVIGATION_STATUSES

    @property
    def succeeded(self) -> bool:
        return self is NavigationStatus.SUCCEEDED


_TERMINAL_NAVIGATION_STATUSES = frozenset({
    NavigationStatus.SUCCEEDED,
    NavigationStatus.REJECTED,
    NavigationStatus.ABORTED,
    NavigationStatus.CANCELED,
    NavigationStatus.ERROR,
})


@dataclass(frozen=True)
class NavigationProgress:
    """What ``NavigationClient.poll`` reports for one request.

    ``distance_remaining`` comes from Nav2 feedback (None until the first one);
    ``reason`` explains a non-success status. An unknown id gives ``ERROR``.
    """

    status: NavigationStatus
    distance_remaining: float | None = None
    reason: str = ''

    def __post_init__(self):
        if self.distance_remaining is not None and self.distance_remaining < 0.0:
            raise ValueError(
                f'distance_remaining {self.distance_remaining} cannot be negative')

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def succeeded(self) -> bool:
        return self.status.succeeded


# --------------------------------------------------------------------------
# Inspection records
# --------------------------------------------------------------------------

class InspectionState(Enum):
    """Inspection progress for one ``artifact_id``:
    ``UNVISITED -> IN_PROGRESS -> VISITED`` on success, or ``DEFERRED`` on failure.
    """

    UNVISITED = 'UNVISITED'
    IN_PROGRESS = 'IN_PROGRESS'
    VISITED = 'VISITED'
    DEFERRED = 'DEFERRED'


@dataclass(frozen=True)
class InspectionRecord:
    """Inspection record for one artefact."""

    artifact_id: int
    state: InspectionState
    attempts: int
    last_failure_reason: str = ''

    def __post_init__(self):
        if self.attempts < 0:
            raise ValueError('attempts cannot be negative')
        if self.state is InspectionState.UNVISITED and self.attempts:
            raise ValueError('UNVISITED means no attempt has been made')
        if self.state is InspectionState.DEFERRED and not self.last_failure_reason:
            raise ValueError('DEFERRED must record why the attempt failed')
