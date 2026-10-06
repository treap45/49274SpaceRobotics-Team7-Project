import math
import time
from geometry_msgs.msg import PoseStamped

from .interfaces import (
    InspectionGoal,
    InspectionAssessment,
    InspectionGoalStatus,
    InspectionEvaluation,
)


class InspectionPlanner:
    """Generate safe inspection poses and evaluate inspection success."""

    def __init__(
        self,
        stand_off_distance: float = 1.0,
        distance_tolerance: float = 0.4,
        heading_tolerance_deg: float = 30.0,
    ):
        self.stand_off_distance = stand_off_distance
        self.distance_tolerance = distance_tolerance
        self.heading_tolerance = math.radians(heading_tolerance_deg)
        self.hold_time = 2.0
        self._condition_since = None

    def make_goal(self, robot_pose, artifact_position):
        """
        Create an observation pose near the artifact.

        The robot stops at a safe stand-off distance and faces the artifact.
        """

        rx = robot_pose.x
        ry = robot_pose.y

        ax = artifact_position.x
        ay = artifact_position.y

        dx = ax - rx
        dy = ay - ry

        distance = math.hypot(dx, dy)

        if distance <= 1e-6:
            return InspectionGoal(
                status=InspectionGoalStatus.NO_SAFE_GOAL,
                goal=None,
                reason="Robot position is too close to artifact centre",
            )

        ux = dx / distance
        uy = dy / distance

        goal_x = ax - ux * self.stand_off_distance
        goal_y = ay - uy * self.stand_off_distance

        goal_heading = math.atan2(
            ay - goal_y,
            ax - goal_x,
        )
        goal = PoseStamped()

        goal.header.frame_id = "map"

        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.position.z = 0.0

        goal.pose.orientation.z = math.sin(goal_heading / 2.0)
        goal.pose.orientation.w = math.cos(goal_heading / 2.0)

        return InspectionGoal(
        status=InspectionGoalStatus.READY,
        goal=goal,
        reason="Inspection goal generated",
    )
    def evaluate(self, robot_pose, artifact_position):
        """
        Evaluate whether the robot is in a suitable inspection pose.
        """

        dx = artifact_position.x - robot_pose.x
        dy = artifact_position.y - robot_pose.y

        distance = math.hypot(dx, dy)

        target_heading = math.atan2(dy, dx)

        heading_error = math.atan2(
            math.sin(target_heading - robot_pose.theta),
            math.cos(target_heading - robot_pose.theta),
        )

        distance_ok = abs(distance - self.stand_off_distance) <= self.distance_tolerance
        heading_ok = abs(heading_error) <= self.heading_tolerance

        if distance_ok and heading_ok:
            now = time.monotonic()

            if self._condition_since is None:
                self._condition_since = now

            held_for = now - self._condition_since

            if held_for >= self.hold_time:
                return InspectionAssessment(
                    status=InspectionEvaluation.SATISFIED,
                    reason=f"Inspection conditions held for {held_for:.2f} s",
                )

            return InspectionAssessment(
                status=InspectionEvaluation.PENDING,
                reason=f"Conditions satisfied; holding {held_for:.2f}/{self.hold_time:.2f} s",
            )

        self._condition_since = None

        return InspectionAssessment(
            status=InspectionEvaluation.PENDING,
            reason=(
                f"distance={distance:.2f} m, "
                f"heading_error={math.degrees(heading_error):.1f} deg"
            ),
        )
