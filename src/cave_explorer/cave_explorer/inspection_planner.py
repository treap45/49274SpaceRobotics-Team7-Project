"""Observation poses and inspection success for one artefact.

Owner: B.
Contract: repository-root Workplan.md, section 4.4.

``make_goal`` places the robot ``stand_off_m`` from the artefact, facing it;
types with a keep-out disk (``artifact_obstacles.KEEP_OUT_RADIUS_M``) are
viewed from outside it, ``keep_out_margin_m`` beyond its edge.
Candidates go round the artefact, starting on the side facing the robot, then
at a larger stand-off. A candidate is kept only if its cell is known free, it
has ``clearance_m`` to every occupied cell (robot radius 0.5 m plus margin) and
the camera has a line of sight to the artefact. Nav2 plans on an inflated
costmap rather than this grid, so a pose that passes here can still be
rejected; ``exclude`` lists poses already tried so the next call returns a
different one (Workplan 4.4).

``evaluate`` follows the task sheet's Planning 2: reach a close-range
viewpoint "keeping the artefact in view". It is SATISFIED once the robot has
been within the distance and heading tolerances for ``hold_time_sec`` of ROS
time, having seen the artefact from within ``sighting_range_m`` during the last
``sighting_window_sec``. It does not demand a detection at the viewpoint
itself: in simulation the detector stopped seeing artefacts inside about 2 m,
so that rule failed every inspection. It is LOST when the robot has been at
close range for ``sighting_window_sec`` without seeing the artefact; the time
counts from when it came within range, or from the last sighting if later, and
restarts with every ``make_goal``. Counted from the last sighting alone, an
artefact queued behind another, or viewed from 3.7 m, was LOST on arrival.
Far away and unseen is still PENDING: the camera usually loses the artefact
while the robot drives round to it.
"""

import math

from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy import ndimage

from .artifact_obstacles import KEEP_OUT_RADIUS_M
from .interfaces import (
    InspectionAssessment,
    InspectionEvaluation,
    InspectionGoal,
    InspectionGoalStatus,
)

OCCUPIED_THRESHOLD = 50     # same classification as frontier_explorer
UNKNOWN = -1


class InspectionPlanner:
    """Generate safe inspection poses and evaluate inspection success."""

    def __init__(
        self,
        stand_off_m: float = 1.2,
        distance_tolerance_m: float = 1.0,
        heading_tolerance_deg: float = 30.0,
        hold_time_sec: float = 2.0,
        sighting_range_m: float = 4.0,
        sighting_window_sec: float = 15.0,
        clearance_m: float = 0.6,
        exclude_radius_m: float = 0.3,
        bearing_offsets_deg: tuple = (0, 45, -45, 90, -90, 135, -135, 180),
        extra_stand_off_m: tuple = (0.0, 0.5),
        keep_out_margin_m: float = 1.0,
    ):
        """``sighting_window_sec`` covers the last few metres of the approach
        at the robot's ~0.5 m/s plus ``hold_time_sec``.

        ``distance_tolerance_m`` matches Nav2's ``xy_goal_tolerance`` (1.0 m in
        nav2_params.yaml): Nav2 reports arrival anywhere within that of the
        viewpoint, and in simulation stopped 2.0 m from an artefact whose
        viewpoint was at 1.2 m. A tighter window is never satisfied there.

        ``keep_out_margin_m`` covers the chassis' 0.71 m half-diagonal plus 0.3 m.
        """
        self.stand_off_m = stand_off_m
        self.distance_tolerance_m = distance_tolerance_m
        self.heading_tolerance = math.radians(heading_tolerance_deg)
        self.hold_time_ns = int(hold_time_sec * 1e9)
        self.sighting_range_m = sighting_range_m
        self.sighting_window_ns = int(sighting_window_sec * 1e9)
        self.clearance_m = clearance_m
        self.exclude_radius_m = exclude_radius_m
        self.bearing_offsets = tuple(math.radians(d) for d in bearing_offsets_deg)
        self.extra_stand_off_m = tuple(extra_stand_off_m)
        self.keep_out_margin_m = keep_out_margin_m

        # artifact_id -> ROS time (ns) since which all conditions have held.
        self._holding_since = {}
        # artifact_id -> last_seen_ns of the latest sighting that arrived while
        # the robot was within sighting_range_m, and the last_seen_ns from the
        # previous call, so an old far sighting is not credited as close.
        self._close_sighting = {}
        self._previous_seen = {}
        # artifact_id -> ROS time (ns) the robot came within view range for the
        # current viewpoint; the LOST clock runs from here.
        self._in_range_since = {}

    # ------------------------------------------------------------------
    # Observation pose
    # ------------------------------------------------------------------

    def make_goal(self, track, robot_pose, map_msg, *, exclude: tuple = ()) -> InspectionGoal:
        """Observation pose for ``track``, avoiding every pose in ``exclude``."""
        if map_msg is None or robot_pose is None:
            return InspectionGoal(InspectionGoalStatus.NOT_READY, None,
                                  'no map or robot pose yet')
        self._in_range_since.pop(track.artifact_id, None)    # new viewpoint, new chance
        grid = _Grid(map_msg)
        ax, ay = track.position_map[0], track.position_map[1]
        if not grid.contains(ax, ay):
            return InspectionGoal(InspectionGoalStatus.NOT_READY, None,
                                  'artefact is outside the current map')

        clearance = grid.clearance_m()
        excluded = [(pose.pose.position.x, pose.pose.position.y) for pose in exclude]
        toward_robot = math.atan2(robot_pose.y - ay, robot_pose.x - ax)
        rejected = {'excluded': 0, 'not free': 0, 'too close to a wall': 0,
                    'no line of sight': 0}

        base = self.stand_off_for(track.class_name)
        for stand_off in (base + extra for extra in self.extra_stand_off_m):
            for offset in self.bearing_offsets:
                bearing = toward_robot + offset
                gx = ax + stand_off * math.cos(bearing)
                gy = ay + stand_off * math.sin(bearing)
                if any(math.dist((gx, gy), p) < self.exclude_radius_m for p in excluded):
                    rejected['excluded'] += 1
                elif not grid.is_free(gx, gy):
                    rejected['not free'] += 1
                elif clearance[grid.cell(gx, gy)] < self.clearance_m:
                    rejected['too close to a wall'] += 1
                elif not grid.line_of_sight(gx, gy, ax, ay):
                    rejected['no line of sight'] += 1
                else:
                    return InspectionGoal(
                        InspectionGoalStatus.READY,
                        _pose_facing(gx, gy, ax, ay, map_msg.header.frame_id or 'map'),
                        f'{stand_off:.1f} m from artefact {track.artifact_id}')

        summary = ', '.join(f'{n} {why}' for why, n in rejected.items() if n)
        return InspectionGoal(InspectionGoalStatus.NO_SAFE_GOAL, None,
                              f'no safe viewpoint ({summary})')

    def stand_off_for(self, class_name: str) -> float:
        """Viewing distance for an artefact type: outside its keep-out disk, if it has one."""
        keep_out = KEEP_OUT_RADIUS_M.get(class_name)
        if keep_out is None:
            return self.stand_off_m
        return max(self.stand_off_m, keep_out + self.keep_out_margin_m)

    # ------------------------------------------------------------------
    # Inspection success
    # ------------------------------------------------------------------

    def evaluate(self, track, robot_pose, now_ns: int) -> InspectionAssessment:
        """Whether the inspection of ``track`` has succeeded yet."""
        artifact_id = track.artifact_id
        stand_off = self.stand_off_for(track.class_name)
        # A far viewpoint still needs its sightings credited as close.
        sighting_range = max(self.sighting_range_m, stand_off + self.distance_tolerance_m)
        dx = track.position_map[0] - robot_pose.x
        dy = track.position_map[1] - robot_pose.y
        distance = math.hypot(dx, dy)
        heading_error = _wrap(math.atan2(dy, dx) - robot_pose.theta)
        unseen_ns = now_ns - track.last_seen_ns
        previous = self._previous_seen.get(artifact_id)
        self._previous_seen[artifact_id] = track.last_seen_ns
        if (previous is not None and track.last_seen_ns != previous
                and distance <= sighting_range):
            self._close_sighting[artifact_id] = track.last_seen_ns
        close_sighting = self._close_sighting.get(artifact_id)
        if close_sighting is not None and close_sighting > now_ns:    # time went backwards
            close_sighting = None
            self._close_sighting.pop(artifact_id)

        in_view_range = distance <= stand_off + self.distance_tolerance_m
        if not in_view_range:
            self._in_range_since.pop(artifact_id, None)
        else:
            in_range_since = self._in_range_since.setdefault(artifact_id, now_ns)
            if in_range_since > now_ns:     # time went backwards
                in_range_since = self._in_range_since[artifact_id] = now_ns
            unseen_in_range_ns = now_ns - max(track.last_seen_ns, in_range_since)
            if unseen_in_range_ns > self.sighting_window_ns:
                self._holding_since.pop(artifact_id, None)
                return InspectionAssessment(
                    InspectionEvaluation.LOST,
                    f'not seen for {unseen_in_range_ns / 1e9:.1f} s in range, '
                    f'{unseen_ns / 1e9:.1f} s in all, at {distance:.2f} m')

        distance_ok = abs(distance - stand_off) <= self.distance_tolerance_m
        heading_ok = abs(heading_error) <= self.heading_tolerance
        seen_ok = (close_sighting is not None
                   and now_ns - close_sighting <= self.sighting_window_ns)
        if not (distance_ok and heading_ok and seen_ok):
            self._holding_since.pop(artifact_id, None)
            return InspectionAssessment(
                InspectionEvaluation.PENDING,
                f'distance={distance:.2f} m, '
                f'heading_error={math.degrees(heading_error):.1f} deg, '
                f'last seen {unseen_ns / 1e9:.1f} s ago'
                + ('' if seen_ok else f', not seen within {sighting_range:g} m lately'))

        since = self._holding_since.setdefault(artifact_id, now_ns)
        if now_ns < since:      # time went backwards (Workplan 4.1)
            since = self._holding_since[artifact_id] = now_ns
        held_ns = now_ns - since
        if held_ns >= self.hold_time_ns:
            self._holding_since.pop(artifact_id, None)
            return InspectionAssessment(
                InspectionEvaluation.SATISFIED,
                f'conditions held for {held_ns / 1e9:.2f} s at {distance:.2f} m')
        return InspectionAssessment(
            InspectionEvaluation.PENDING,
            f'conditions met; holding {held_ns / 1e9:.2f}/{self.hold_time_ns / 1e9:.2f} s')


class _Grid:
    """Read-only view of an OccupancyGrid in world coordinates."""

    def __init__(self, map_msg):
        info = map_msg.info
        self.resolution = info.resolution
        self.origin_x = info.origin.position.x
        self.origin_y = info.origin.position.y
        self.width = info.width
        self.height = info.height
        self.data = np.asarray(map_msg.data, dtype=np.int16).reshape(self.height, self.width)

    def cell(self, x, y):
        return (int(math.floor((y - self.origin_y) / self.resolution)),
                int(math.floor((x - self.origin_x) / self.resolution)))

    def contains(self, x, y):
        row, col = self.cell(x, y)
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free(self, x, y):
        if not self.contains(x, y):
            return False
        value = self.data[self.cell(x, y)]
        return value != UNKNOWN and value < OCCUPIED_THRESHOLD

    def clearance_m(self):
        """Distance in metres from every cell to the nearest occupied cell."""
        occupied = self.data >= OCCUPIED_THRESHOLD
        return ndimage.distance_transform_edt(~occupied) * self.resolution

    def line_of_sight(self, x0, y0, x1, y1, stop_short_m: float = 0.4):
        """No occupied cell between the two points.

        Stops ``stop_short_m`` before the artefact, which may itself be
        occupied in the map.
        """
        length = math.hypot(x1 - x0, y1 - y0) - stop_short_m
        steps = int(max(length, 0.0) / (self.resolution / 2.0))
        for i in range(steps + 1):
            s = i * (self.resolution / 2.0) / math.hypot(x1 - x0, y1 - y0)
            x, y = x0 + s * (x1 - x0), y0 + s * (y1 - y0)
            if self.contains(x, y) and self.data[self.cell(x, y)] >= OCCUPIED_THRESHOLD:
                return False
        return True


def _pose_facing(x, y, target_x, target_y, frame_id) -> PoseStamped:
    heading = math.atan2(target_y - y, target_x - x)
    goal = PoseStamped()
    goal.header.frame_id = frame_id
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.orientation.z = math.sin(heading / 2.0)
    goal.pose.orientation.w = math.cos(heading / 2.0)
    return goal


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))
