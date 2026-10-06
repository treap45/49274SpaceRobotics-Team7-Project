"""Shared data contracts between the perception, tracking and planning modules.

Owner: A/B/C (joint review).
Contract: repository-root Workplan.md, section 4.

Conventions (Workplan 4.1):
  * Distances are metres and angles are radians.
  * Map positions are in the ``map`` frame; navigation goals are ``PoseStamped``.
  * ``stamp_ns`` is the ROS simulation time of the original observation as an
    integer nanosecond count. Every timeout compares against the same clock.
    Clear cached observations, tracks and action records if time goes backwards.
  * Bounding boxes are ``(x_min, y_min, x_max, y_max)`` in the original RGB
    image resolution. The right and bottom edges are exclusive.
  * Failure is reported through a status value, never through a sentinel
    coordinate: ``(0.0, 0.0)`` is a valid position in the cave.

Every payload here is a frozen dataclass holding tuples rather than lists, so a
snapshot handed to another module cannot be mutated behind the producer's back.
This is the one place these definitions tighten Workplan.md section 4.2, which
sketches ``detections`` as a ``list``; the read-only-snapshot rule in section
4.1 is the reason. Raise it in review if it causes friction.

Thresholds, clustering radii and inspection parameters do not belong here -
they are configuration, per Workplan.md section 4.1.
"""

from dataclasses import dataclass
from enum import Enum

from geometry_msgs.msg import PoseStamped

# --------------------------------------------------------------------------
# Label table (Workplan 4.1)
# --------------------------------------------------------------------------
# These are the five artefact types actually placed in worlds/mars_cave.sdf,
# six instances of each. Same-class artefacts are only distinguishable by
# position, so association must not merge them on class alone (Workplan 4.3).

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
# A -> B: two-dimensional detections (Workplan 4.2)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection2D:
    """One detected artefact in one image.

    Produced by A. Carries no depth and no map position - those are B's result.
    """

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
    """Every detection from a single image, with that image's metadata.

    ``stamp_ns`` and ``source_frame_id`` are the capture time and frame of the
    input image, not of the moment inference finished. B must look up depth and
    TF at ``stamp_ns``, so A preserves them even when inference lags.
    """

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
# B -> C: artefact map snapshot (Workplan 4.3)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ArtifactTrack:
    """One artefact B believes exists, at its fused map position.

    ``artifact_id`` is assigned once the track is confirmed and never changes
    afterwards. B keeps provisional tracks internal, so all merging and
    de-duplication happens before an id exists and C never sees one move - there
    is deliberately no old-id-to-new-id remapping mechanism (Workplan 4.3).

    ``last_seen_ns`` answers "is it in view now?", which is not the same
    question as "have we ever seen it?".
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
# B -> C: observation goals and inspection assessment (Workplan 4.4)
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
    """Where the robot should stand to inspect an artefact.

    ``goal`` is the observation pose for the robot, not the artefact's own
    position. B checks local clearance and heading; C confirms reachability
    through the planner, because geometric safety does not imply Nav2 can
    get there.
    """

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
# C internal: exploration and navigation (Workplan 4.5)
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
    """Lifecycle of one ``NavigateToPose`` request.

    Only ``SUCCEEDED`` means the robot arrived. The template code treats every
    terminal callback as arrival; do not carry that over.
    """

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

    ``distance_remaining`` is carried through from Nav2 feedback. It is ``None``
    until the first feedback message arrives, and on a request that never became
    active. C accumulates it for the travelled-distance metric; B uses it to
    monitor an approach.

    ``reason`` explains a non-success terminal status, and is what C passes on
    to ``FrontierExplorer.report_outcome``. Polling an unknown request id is an
    ``ERROR`` with a reason, not an exception.
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
# C owns inspection state; B owns observations (Workplan 4.6)
# --------------------------------------------------------------------------

class InspectionState(Enum):
    """Inspection progress for one ``artifact_id``.

    ``UNVISITED -> IN_PROGRESS -> VISITED`` on success, or ``-> DEFERRED`` on
    failure, which returns the robot to exploration. ``VISITED`` is set only on
    a successful inspection.

    ``DEFERRED`` is terminal for the run: it is not re-enabled later. That keeps
    a still-visible artefact from retriggering an attempt on the next frame, and
    matches the task sheet, which allows abandoning the attempt outright. A
    cooldown-and-retry policy is future work, not current behaviour.
    """

    UNVISITED = 'UNVISITED'
    IN_PROGRESS = 'IN_PROGRESS'
    VISITED = 'VISITED'
    DEFERRED = 'DEFERRED'


@dataclass(frozen=True)
class InspectionRecord:
    """C's record for one artefact, snapshotted read-only to B for markers."""

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
