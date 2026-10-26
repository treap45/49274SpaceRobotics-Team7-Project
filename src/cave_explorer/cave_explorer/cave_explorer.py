#!/usr/bin/env python3
"""Main node: ROS I/O and scheduling for the exploration and inspection pipeline.

Data flow, one module call per arrow::

    camera/image ──▶ Detector ──▶ ArtifactTracker ──▶ confirmed tracks
    camera/depth/image ─┘ matched by stamp      ▲             │
    TF map <- camera_rgb_optical_frame at stamp ┘             ▼
    /map, TF map <- base_link ────────────────▶ BehaviourManager
                                                  ├ exploration (exploration_mode)
                                                  ├ InspectionPlanner
                                                  └ NavigationClient ──▶ Nav2

Everything runs on one single-threaded executor. Goals are shown on
``exploration_goal``, not ``goal_pose``, which Nav2 would treat as a new goal.
"""

from collections import deque
import math

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from cave_explorer.artifact_obstacles import ArtifactObstacles, keep_out_points
from cave_explorer.artifact_tracker import ArtifactTracker, SIM_CAMERA_MODEL
from cave_explorer.behaviour_manager import BehaviourConfig, BehaviourManager
from cave_explorer.coverage_explorer import CoverageExplorer, TwoPhaseExplorer
from cave_explorer.frontier_explorer import FrontierExplorer, frontier_markers
from cave_explorer.inspection_planner import InspectionPlanner
from cave_explorer.interfaces import ARTIFACT_CLASSES, InspectionState
from cave_explorer.navigation_client import NavigationClient
from cave_explorer.nbv_explorer import InformationGainExplorer, InformationTourExplorer

MAP_FRAME = 'map'
ROBOT_FRAME = 'base_link'
# Unprojected depth points are in the optical frame, not camera_link.
CAMERA_OPTICAL_FRAME = 'camera_rgb_optical_frame'
# After getting stuck: reverse 0.75 m, clear of the contact mark ahead.
BACKOFF_SPEED_M_S = 0.25
BACKOFF_SEC = 3.0
# No /plan for this long means Nav2 could not plan at all (it replans at 1 Hz).
PLAN_FRESH_SEC = 15.0
# The camera is 0.5 m ahead of base_link.
CAMERA_FORWARD_M = 0.5
EXPLORATION_MODES = ('two_phase', 'interleaved', 'information_gain', 'info_tour')

# A detection batch waits this long for its depth image and camera TF.
PENDING_TIMEOUT_NS = 1_000_000_000

STATE_COLOURS = {
    None: (0.6, 0.6, 0.6),                          # not an inspection class
    InspectionState.UNVISITED: (1.0, 0.85, 0.0),
    InspectionState.IN_PROGRESS: (1.0, 0.5, 0.0),
    InspectionState.VISITED: (0.1, 0.9, 0.1),
    InspectionState.DEFERRED: (0.9, 0.1, 0.1),
}


class _GoalPublishingNavigator:
    """NavigationClient that also shows every goal it sends in RViz."""

    def __init__(self, client: NavigationClient, publisher):
        self._client = client
        self._publisher = publisher

    def send(self, goal, timeout_sec=None):
        self._publisher.publish(goal)
        return self._client.send(goal, timeout_sec)

    def poll(self, request_id):
        return self._client.poll(request_id)

    def cancel(self, request_id):
        self._client.cancel(request_id)


class CaveExplorer(Node):
    def __init__(self):
        super().__init__('cave_explorer_node')

        # Planning 2: the types the detector handles most reliably.
        self.declare_parameter('inspect_classes', ['green_crystals', 'white_sphere'])
        self.declare_parameter('detection_confidence', 0.5)
        self.declare_parameter('detector_model', '')
        # Channel order the YOLO model expects: the default model takes BGR.
        self.declare_parameter('detector_input_bgr', True)
        self.declare_parameter('depth_sync_tolerance_sec', 0.05)
        # two_phase / interleaved: TwoPhaseExplorer; information_gain:
        # InformationGainExplorer; info_tour: InformationTourExplorer.
        self.declare_parameter('exploration_mode', 'info_tour')

        inspect_classes = frozenset(self.get_parameter('inspect_classes').value)
        self.inspect_classes_ = inspect_classes
        unknown = inspect_classes - ARTIFACT_CLASSES
        if unknown:
            raise ValueError(f'inspect_classes has unknown labels {sorted(unknown)}')
        self.detector_input_bgr_ = self.get_parameter('detector_input_bgr').value
        self.depth_sync_tolerance_ns_ = int(
            self.get_parameter('depth_sync_tolerance_sec').value * 1e9)

        self.cv_bridge_ = CvBridge()
        self.tf_buffer_ = Buffer()
        self.tf_listener_ = TransformListener(self.tf_buffer_, self)

        # Perception and localisation
        self.detector_ = self._load_detector()
        self.tracker_ = ArtifactTracker()
        # Artefact shapes for Nav2: the lidar, 1.0 m up, misses their wide bases.
        self.obstacles_ = ArtifactObstacles()
        self.depth_images_ = deque(maxlen=10)       # (stamp_ns, depth image)
        self.pending_batch_ = None

        # Planning
        self.goal_pub_ = self.create_publisher(PoseStamped, 'exploration_goal', 1)
        self.navigation_ = NavigationClient(self)
        mode = self.get_parameter('exploration_mode').value
        if mode not in EXPLORATION_MODES:
            raise ValueError(f'exploration_mode {mode!r} is not one of {EXPLORATION_MODES}')
        interleave = mode == 'interleaved'
        # Interleaved: cap both gains at what one camera view covers.
        view = CoverageExplorer()
        sector_m2 = 0.5 * view.fov_rad * view.view_range_m ** 2
        chord_m = 2.0 * view.view_range_m * math.sin(view.fov_rad / 2.0)
        self.coverage_ = CoverageExplorer(max_gain_m2=sector_m2 if interleave else math.inf)
        self.explorer_ = TwoPhaseExplorer(
            FrontierExplorer(max_gain_length_m=chord_m if interleave else math.inf),
            self.coverage_, interleave=interleave)
        if mode == 'information_gain':
            self.explorer_ = InformationGainExplorer(self.coverage_)
        elif mode == 'info_tour':
            self.explorer_ = InformationTourExplorer(self.coverage_)
        self.manager_ = BehaviourManager(
            _GoalPublishingNavigator(self.navigation_, self.goal_pub_),
            self.explorer_,
            InspectionPlanner(size_of=self.tracker_.size_m),
            BehaviourConfig(inspect_classes=inspect_classes),
            log=self.get_logger().info,
            on_stuck=self.on_stuck)
        self.map_ = None
        self.last_status_ = None
        self.last_frontiers_ = None

        self.image_detections_pub_ = self.create_publisher(Image, 'detections_image', 1)
        self.marker_pub_ = self.create_publisher(MarkerArray, 'marker_array_artifacts', 10)
        self.frontier_pub_ = self.create_publisher(MarkerArray, 'frontier_markers', 1)
        self.coverage_pub_ = self.create_publisher(OccupancyGrid, 'camera_coverage', 1)
        self.obstacle_pub_ = self.create_publisher(PointCloud2, 'artifact_obstacles', 1)
        self.obstacle_timer_ = self.create_timer(1.0, self.publish_artifact_obstacles)
        # Back off after getting stuck (Nav2's goal is already cancelled).
        self.cmd_vel_pub_ = self.create_publisher(Twist, 'cmd_vel', 1)
        self.backoff_until_ns_ = None
        self.backoff_timer_ = self.create_timer(0.1, self.back_off)
        self.ticks_ = 0

        self.map_sub_ = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 1)
        self.last_plan_ns_ = None
        self.plan_sub_ = self.create_subscription(Path, 'plan', self.plan_callback, 1)
        self.image_sub_ = self.create_subscription(Image, 'camera/image', self.image_callback, 1)
        self.depth_sub_ = self.create_subscription(
            Image, 'camera/depth/image', self.depth_callback, 5)

        self.main_loop_timer_ = self.create_timer(0.2, self.main_loop)
        self.get_logger().info(f'Inspecting classes: {sorted(inspect_classes)}')

    def _load_detector(self):
        """A missing model or ultralytics leaves the node exploring without detection."""
        try:
            from cave_explorer.perception import Detector
            model = self.get_parameter('detector_model').value or None
            detector = Detector(model, self.get_parameter('detection_confidence').value)
        except Exception as ex:     # noqa: BLE001 - any failure here is reported the same way
            self.get_logger().error(f'Detector unavailable, exploring only: {ex}')
            return None
        self.get_logger().info(f'Detector loaded: {detector.model_path}')
        return detector

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def get_pose_2d(self):
        """Robot pose in ``map``, or None before SLAM publishes the transform."""
        try:
            t = self.tf_buffer_.lookup_transform(MAP_FRAME, ROBOT_FRAME, Time())
        except TransformException:
            return None
        q = t.transform.rotation
        pose = Pose2D()
        pose.x = t.transform.translation.x
        pose.y = t.transform.translation.y
        pose.theta = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return pose

    def map_callback(self, map_msg: OccupancyGrid):
        self.map_ = map_msg

    def depth_callback(self, depth_msg: Image):
        stamp_ns = Time.from_msg(depth_msg.header.stamp).nanoseconds
        depth = self.cv_bridge_.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
        self.depth_images_.append((stamp_ns, depth))
        self._localise_pending()

    def image_callback(self, image_msg: Image):
        """Detect artefacts, publish the annotated image, then localise them."""
        if self.detector_ is None:
            return
        stamp_ns = Time.from_msg(image_msg.header.stamp).nanoseconds
        bgr = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        model_input = bgr if self.detector_input_bgr_ else bgr[:, :, ::-1].copy()
        batch = self.detector_.detect(model_input, stamp_ns, image_msg.header.frame_id)

        annotated = self.detector_.annotate(bgr, batch)
        out = self.cv_bridge_.cv2_to_imgmsg(annotated, encoding='bgr8')
        out.header = image_msg.header       # keep the capture time and frame
        self.image_detections_pub_.publish(out)

        if batch.detections:
            self.pending_batch_ = batch
            self._localise_pending()

    def _localise_pending(self):
        """Hand the latest batch to the tracker once its depth and TF have arrived."""
        batch = self.pending_batch_
        if batch is None:
            return
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - batch.stamp_ns > PENDING_TIMEOUT_NS:
            self.pending_batch_ = None
            self.get_logger().warn('Detections dropped: no matching depth image or TF',
                                   throttle_duration_sec=5.0)
            return

        depth = self._depth_at(batch.stamp_ns)
        if depth is None:
            return
        try:
            transform = self.tf_buffer_.lookup_transform(
                MAP_FRAME, CAMERA_OPTICAL_FRAME, Time(nanoseconds=batch.stamp_ns))
        except TransformException:
            return

        self.pending_batch_ = None
        self.tracker_.update(batch, depth, SIM_CAMERA_MODEL, transform)
        for retired_id, kept_id in self.tracker_.last_merges:
            self.get_logger().info(f'artefact {retired_id} merged into artefact {kept_id}')
            self.manager_.merge_records(retired_id, kept_id)
        self.obstacles_.add(batch, depth, SIM_CAMERA_MODEL, transform)
        for reason in self.tracker_.last_rejections:
            self.get_logger().info(f'Detection not localised: {reason}',
                                   throttle_duration_sec=2.0)

    def _depth_at(self, stamp_ns):
        """The buffered depth image closest to ``stamp_ns``, if close enough."""
        best = min(self.depth_images_, key=lambda item: abs(item[0] - stamp_ns), default=None)
        if best is None or abs(best[0] - stamp_ns) > self.depth_sync_tolerance_ns_:
            return None
        return best[1]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def main_loop(self):
        self._localise_pending()

        robot_pose = self.get_pose_2d()
        if robot_pose is None:
            self.get_logger().warn('Waiting for transform... Have you launched a SLAM node?',
                                   throttle_duration_sec=5.0)
            return
        if not self.navigation_.server_ready():
            self.get_logger().warn('Waiting for navigate_to_pose action...',
                                   throttle_duration_sec=5.0)
            return
        if self.map_ is None:
            self.get_logger().warn('Waiting for map...', throttle_duration_sec=5.0)
            return

        # Where the camera has looked, for the coverage phase of exploration.
        self.coverage_.observe(
            self.map_,
            (robot_pose.x + CAMERA_FORWARD_M * math.cos(robot_pose.theta),
             robot_pose.y + CAMERA_FORWARD_M * math.sin(robot_pose.theta)),
            robot_pose.theta)
        self.ticks_ += 1
        if self.ticks_ % 5 == 0:
            self.coverage_pub_.publish(self.coverage_.coverage_grid(self.map_))

        tracks = self.tracker_.snapshot()
        self.manager_.step(self.get_clock().now().nanoseconds, robot_pose, self.map_, tracks)

        if self.manager_.status != self.last_status_:
            self.last_status_ = self.manager_.status
            self.get_logger().info(f'[{self.manager_.mode.name}] {self.last_status_}')
        if self.explorer_.frontiers is not self.last_frontiers_:
            self.last_frontiers_ = self.explorer_.frontiers
            self.frontier_pub_.publish(frontier_markers(self.explorer_))
        self.publish_artifact_markers(tracks)

    def plan_callback(self, _path: Path):
        self.last_plan_ns_ = self.get_clock().now().nanoseconds

    def on_stuck(self, robot_pose) -> float:
        """Mark where the robot stuck and back it off; return seconds to wait.

        Only if Nav2 still had a recent /plan (the robot was held, not unroutable).
        """
        now_ns = self.get_clock().now().nanoseconds
        if self.last_plan_ns_ is None or now_ns - self.last_plan_ns_ > PLAN_FRESH_SEC * 1e9:
            self.get_logger().warn(
                f'No progress at ({robot_pose.x:.1f}, {robot_pose.y:.1f}) and no path from Nav2: '
                'goal unreachable, not a collision')
            return 0.0
        added = self.obstacles_.mark_contact(robot_pose.x, robot_pose.y, robot_pose.theta)
        self.get_logger().warn(
            f'Stuck at ({robot_pose.x:.1f}, {robot_pose.y:.1f}): marked {added} obstacle '
            f'cells ahead, backing off {BACKOFF_SPEED_M_S * BACKOFF_SEC:.2f} m')
        self.publish_artifact_obstacles()
        self.backoff_until_ns_ = (self.get_clock().now().nanoseconds
                                  + int(BACKOFF_SEC * 1e9))
        return BACKOFF_SEC + 1.0    # and a second for the costmap to take the mark

    def back_off(self):
        if self.backoff_until_ns_ is None:
            return
        twist = Twist()
        if self.get_clock().now().nanoseconds < self.backoff_until_ns_:
            twist.linear.x = -BACKOFF_SPEED_M_S
        else:
            self.backoff_until_ns_ = None       # one zero command to stop
        self.cmd_vel_pub_.publish(twist)

    def publish_artifact_obstacles(self):
        """Everything ArtifactObstacles has accumulated, plus the keep-out disks, for Nav2."""
        pose = self.get_pose_2d()
        disks = keep_out_points(self.tracker_.snapshot(), self.tracker_.size_m,
                                None if pose is None else (pose.x, pose.y),
                                classes=self.inspect_classes_)
        header = Header(frame_id=MAP_FRAME, stamp=self.get_clock().now().to_msg())
        self.obstacle_pub_.publish(point_cloud2.create_cloud_xyz32(
            header, np.concatenate((self.obstacles_.points(), disks))))

    def publish_artifact_markers(self, tracks):
        """One sphere and label per confirmed artefact, coloured by inspection state."""
        states = {record.artifact_id: record.state for record in self.manager_.records}
        clear_all = Marker()
        clear_all.header.frame_id = MAP_FRAME
        clear_all.action = Marker.DELETEALL
        markers = [clear_all]
        for track in tracks:
            state = states.get(track.artifact_id)
            label = track.class_name if state is None else f'{track.class_name} {state.name}'
            for kind, marker_id in ((Marker.SPHERE, 2 * track.artifact_id),
                                    (Marker.TEXT_VIEW_FACING, 2 * track.artifact_id + 1)):
                marker = Marker()
                marker.header.frame_id = MAP_FRAME
                marker.ns = 'artifacts'
                marker.id = marker_id
                marker.type = kind
                marker.action = Marker.ADD
                marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = \
                    track.position_map
                marker.pose.orientation.w = 1.0
                marker.color.r, marker.color.g, marker.color.b = STATE_COLOURS[state]
                marker.color.a = 1.0
                if kind == Marker.SPHERE:
                    marker.scale.x = marker.scale.y = marker.scale.z = 0.5
                else:
                    marker.pose.position.z += 0.8
                    marker.scale.z = 0.4
                    marker.text = f'{track.artifact_id}: {label}'
                markers.append(marker)
        self.marker_pub_.publish(MarkerArray(markers=markers))


def main():
    rclpy.init()
    cave_explorer = CaveExplorer()
    try:
        rclpy.spin(cave_explorer)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        cave_explorer.destroy_node()
        rclpy.try_shutdown()
