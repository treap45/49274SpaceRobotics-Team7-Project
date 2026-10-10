"""Independent, timer-driven ROS wrapper for cave geometry analysis."""

import math
import time

from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from cave_explorer.cave_geometry import analyze_geometry, grid_to_world, world_to_grid


def map_transform(info):
    """Extract planar origin and yaw from an OccupancyGrid's metadata."""
    q = info.origin.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
    return (info.origin.position.x, info.origin.position.y), yaw


def geometry_markers(result, info, stamp):
    """Build stable markers; DELETEALL also clears stale geometry on failure."""
    origin, yaw = map_transform(info)
    markers = MarkerArray()
    clear = Marker()
    clear.header.frame_id, clear.header.stamp = 'map', stamp
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)

    def marker(namespace, kind, colour):
        item = Marker()
        item.header.frame_id, item.header.stamp = 'map', stamp
        item.ns, item.id, item.type = namespace, 0, kind
        item.action = Marker.ADD
        item.pose.orientation.w = 1.0
        item.color.r, item.color.g, item.color.b, item.color.a = colour
        markers.markers.append(item)
        return item

    def point(rc, z=0.1):
        x, y = grid_to_world(rc, info.resolution, origin, yaw)
        return Point(x=float(x), y=float(y), z=z)

    if result.widest_position_rc is not None:
        area = marker('widest', Marker.CYLINDER, (0.0, 1.0, 0.0, 0.3))
        area.pose.position = point(result.widest_position_rc)
        area.scale.x = area.scale.y = 2 * result.widest_radius_m
        area.scale.z = 0.08
        label = marker('widest_label', Marker.TEXT_VIEW_FACING, (0.0, 1.0, 0.0, 1.0))
        label.pose.position = point(result.widest_position_rc, 0.8)
        label.scale.z = 0.5
        label.text = f'Widest radius: {result.widest_radius_m:.2f} m'
    ridge = marker('ridge', Marker.POINTS, (0.1, 0.3, 1.0, 1.0))
    ridge.scale.x = ridge.scale.y = max(0.05, info.resolution * 0.5)
    cells = np.argwhere(result.ridge_mask)
    stride = max(1, math.ceil(len(cells) / 5000))
    ridge.points = [point(rc, 0.2) for rc in cells[::stride]]
    saddles = marker('saddles', Marker.SPHERE_LIST, (1.0, 0.5, 0.0, 0.9))
    saddles.scale.x = saddles.scale.y = saddles.scale.z = 0.3
    saddles.points = [point(rc, 0.3) for rc in result.saddle_candidates[:1000]]
    if result.bottleneck_position_rc is not None:
        line = marker('bottleneck', Marker.LINE_LIST, (1.0, 0.0, 0.0, 1.0))
        line.scale.x = 0.12
        line.points = [point(rc, 0.4) for rc in result.bottleneck_wall_points_rc]
        label = marker('bottleneck_label', Marker.TEXT_VIEW_FACING, (1.0, 0.0, 0.0, 1.0))
        label.pose.position = point(result.bottleneck_position_rc, 1.0)
        label.scale.z = 0.5
        label.text = f'Bottleneck width: {result.bottleneck_width_m:.2f} m'
    return markers


class CaveGeometryNode(Node):
    """Observe map/TF without sending any commands to the autonomy pipeline."""

    def __init__(self):
        super().__init__('cave_geometry')
        self.declare_parameter('update_period_s', 2.0)
        self.declare_parameter('robot_radius_m', 0.5)
        self.declare_parameter('publish_distance_map', True)
        period = self.get_parameter('update_period_s').value
        radius = self.get_parameter('robot_radius_m').value
        if not math.isfinite(period) or period <= 0 or not math.isfinite(radius) or radius < 0:
            raise ValueError('update period must be positive and robot radius nonnegative')
        self.map_msg = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # SLAM publishes a transient-local map; retain the last snapshot when
        # this observer starts after mapping has begun.
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self._map, qos)
        self.marker_pub = self.create_publisher(MarkerArray, '/cave_geometry/markers', 1)
        self.distance_pub = self.create_publisher(OccupancyGrid, '/cave_geometry/distance_map', 1)
        self.timer = self.create_timer(period, self._update)
        self.get_logger().info('Cave geometry ready; waiting for /map and map <- base_link TF')

    def _map(self, message):
        self.map_msg = message

    def _clear(self):
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.action = Marker.DELETEALL
        self.marker_pub.publish(MarkerArray(markers=[clear]))

    def _update(self):
        started = time.perf_counter()
        message = self.map_msg
        if message is None:
            return
        try:
            if message.header.frame_id.lstrip('/') != 'map':
                raise ValueError('expected occupancy grid in map frame')
            info = message.info
            origin, yaw = map_transform(info)
            robot_rc = None
            try:
                transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
                translation = transform.transform.translation
                robot_rc = world_to_grid((translation.x, translation.y), info.resolution, origin, yaw)
            except TransformException:
                pass
            grid = np.asarray(message.data, dtype=np.int16).reshape(info.height, info.width)
            result = analyze_geometry(grid, info.resolution, robot_rc,
                                      self.get_parameter('robot_radius_m').value)
            stamp = self.get_clock().now().to_msg()
            self.marker_pub.publish(geometry_markers(result, info, stamp))
            if self.get_parameter('publish_distance_map').value:
                output = OccupancyGrid()
                output.header.frame_id, output.header.stamp = 'map', stamp
                output.info = info
                scale = max(float(result.distance_map.max()), info.resolution)
                values = np.where(result.reachable_mask,
                                  np.rint(result.distance_map / scale * 100), -1)
                output.data = values.astype(np.int8).ravel().tolist()
                self.distance_pub.publish(output)
            if result.widest_position_rc is None:
                summary = f'widest unavailable: {result.reason}'
            else:
                xy = grid_to_world(result.widest_position_rc, info.resolution, origin, yaw)
                summary = f'widest centre=({xy[0]:.2f}, {xy[1]:.2f}) radius={result.widest_radius_m:.2f} m'
            if result.bottleneck_position_rc is None:
                summary += '; no valid bottleneck currently detected'
            else:
                xy = grid_to_world(result.bottleneck_position_rc, info.resolution, origin, yaw)
                summary += f'; bottleneck centre=({xy[0]:.2f}, {xy[1]:.2f}) width={result.bottleneck_width_m:.2f} m'
            self.get_logger().info(f'{summary}; processing={time.perf_counter() - started:.4f} s')
        except Exception as error:
            # A bad snapshot cannot terminate this observer or affect autonomy.
            self._clear()
            self.get_logger().error(f'Geometry update skipped: {error}')


def main(args=None):
    """Run the independent observer."""
    rclpy.init(args=args)
    node = CaveGeometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
