"""Independent ROS observer that builds the online roadmap (Advanced 4).

Grows the baseline ``trail`` roadmap and the improved ``observed`` roadmap side
by side from ``/map`` and the robot pose, and publishes both to RViz. It only
reads ``/map``, TF and ``/clicked_point``, so it never changes how the robot
explores.

Click a point with RViz's *Publish Point* tool to plan from the robot to it
over both roadmaps; the paths are drawn and the observed one is published as
``/online_roadmap/path``.
"""

import math
import pickle
import time

from geometry_msgs.msg import Point, PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from cave_explorer.cave_geometry_node import map_transform
from cave_explorer.online_roadmap import (MapSnapshot, RoadmapBuilder, failure_reason, path_length,
                                          plan_path, shortcut)

NODE_COLOURS = {'trail': (1.0, 0.55, 0.0), 'observed': (0.0, 0.75, 1.0)}
EDGE_COLOURS = {'trail': (1.0, 0.85, 0.3), 'observed': (0.2, 0.9, 0.3)}
PATH_COLOURS = {'trail': (1.0, 0.3, 0.0), 'observed': (1.0, 0.0, 1.0)}
# Draw the two roadmaps at different heights so both stay visible together.
HEIGHTS = {'trail': 0.15, 'observed': 0.35}


def roadmap_markers(builders, paths, stamp):
    """Nodes by kind, edges, any requested paths and a label, per roadmap."""
    markers = MarkerArray()
    clear = Marker()
    clear.header.frame_id, clear.header.stamp = 'map', stamp
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)

    def marker(namespace, kind, colour, alpha=1.0):
        item = Marker()
        item.header.frame_id, item.header.stamp = 'map', stamp
        item.ns, item.id, item.type = namespace, 0, kind
        item.action = Marker.ADD
        item.pose.orientation.w = 1.0
        item.color.r, item.color.g, item.color.b = colour
        item.color.a = alpha
        markers.markers.append(item)
        return item

    for row, builder in enumerate(builders):
        mode, roadmap, z = builder.mode, builder.roadmap, HEIGHTS[builder.mode]

        def point(xy, height=z):
            return Point(x=float(xy[0]), y=float(xy[1]), z=height)

        edges = marker(f'{mode}/edges', Marker.LINE_LIST, EDGE_COLOURS[mode], 0.7)
        edges.scale.x = 0.05
        for a, b in roadmap.edge_list():
            edges.points += [point(roadmap.xy(a)), point(roadmap.xy(b))]
        for kind in ('trail', 'observed'):
            nodes = marker(f'{mode}/{kind}_nodes', Marker.SPHERE_LIST, NODE_COLOURS[kind])
            nodes.scale.x = nodes.scale.y = nodes.scale.z = 0.35
            nodes.points = [point((n.x, n.y)) for n in roadmap.nodes.values() if n.kind == kind]
        path = paths.get(mode)
        if path:
            line = marker(f'{mode}/path', Marker.LINE_STRIP, PATH_COLOURS[mode])
            line.scale.x = 0.2
            line.points = [point(xy, z + 0.1) for xy in path]
        if roadmap.nodes:
            first = next(iter(roadmap.nodes.values()))
            label = marker(f'{mode}/label', Marker.TEXT_VIEW_FACING, NODE_COLOURS[mode])
            label.pose.position = point((first.x, first.y), 2.0 + 0.8 * row)
            label.scale.z = 0.6
            components = len(set(roadmap.components().values()))
            label.text = (f'{mode}: {len(roadmap.nodes)} nodes, {len(roadmap.edge_list())} edges, '
                          f'{components} component{"s" if components != 1 else ""}')
    return markers


class OnlineRoadmapNode(Node):
    """Observe map/TF and maintain both roadmaps."""

    def __init__(self):
        super().__init__('online_roadmap')
        self.declare_parameter('update_period_s', 0.2)
        self.declare_parameter('robot_radius_m', 0.5)
        self.declare_parameter('trail_spacing_m', 2.0)
        self.declare_parameter('attach_radius_m', 8.0)
        self.declare_parameter('marker_period_s', 1.0)
        self.declare_parameter('summary_period_s', 30.0)
        self.declare_parameter('record_path', '')
        if not (self._param('update_period_s') > 0 and self._param('robot_radius_m') >= 0):
            raise ValueError('update period must be positive and robot radius nonnegative')
        self.builders = [
            RoadmapBuilder(mode, trail_spacing_m=self._param('trail_spacing_m'),
                           attach_radius_m=self._param('attach_radius_m'))
            for mode in ('trail', 'observed')]
        self.map_msg = None
        self.snapshot = None
        self._snapshot_source = None
        self.map_version = 0
        self.robot_xy = None
        self.paths = {}
        self.last_markers_s = -math.inf
        self.last_summary_s = -math.inf
        self.record_path = self._param('record_path')
        self.record = {'maps': {}, 'ticks': []}
        self.last_saved = time.monotonic()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self._map, qos)
        self.create_subscription(PointStamped, '/clicked_point', self._goal, 1)
        self.marker_pub = self.create_publisher(MarkerArray, '/online_roadmap/markers', 1)
        self.path_pub = self.create_publisher(Path, '/online_roadmap/path', 1)
        self.timer = self.create_timer(self._param('update_period_s'), self._update)
        self.get_logger().info('Online roadmap ready: trail and observed roadmaps from /map; '
                               'click a point (Publish Point) to plan over them')

    def _param(self, name):
        return self.get_parameter(name).value

    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _map(self, message):
        self.map_msg = message

    def _robot_xy(self):
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        except TransformException:
            return None
        translation = transform.transform.translation
        return np.array([translation.x, translation.y])

    def _update(self):
        message = self.map_msg
        if message is None:
            return
        try:
            if message is not self._snapshot_source:
                if message.header.frame_id.lstrip('/') != 'map':
                    raise ValueError('expected occupancy grid in map frame')
                info = message.info
                origin, yaw = map_transform(info)
                grid = np.asarray(message.data, dtype=np.int16).reshape(info.height, info.width)
                self.snapshot = MapSnapshot(grid, info.resolution, origin, yaw,
                                            self._param('robot_radius_m'))
                self._snapshot_source = message
                self.map_version += 1
                if self.record_path:
                    self.record['maps'][self.map_version] = (grid.astype(np.int8), info.resolution,
                                                             origin, yaw)
            self.robot_xy = self._robot_xy()
            now = self._now_s()
            if self.record_path and self.robot_xy is not None:
                self.record['ticks'].append((now, self.map_version, self.robot_xy.copy()))
            for builder in self.builders:
                builder.update(self.snapshot, self.robot_xy, now)
            if now - self.last_markers_s >= self._param('marker_period_s'):
                self.last_markers_s = now
                self.marker_pub.publish(roadmap_markers(self.builders, self.paths,
                                                        self.get_clock().now().to_msg()))
            if now - self.last_summary_s >= self._param('summary_period_s'):
                self.last_summary_s = now
                self.get_logger().info('; '.join(self._summary(b) for b in self.builders))
            if self.record_path and time.monotonic() - self.last_saved > 30.0:
                self.save_record()
        except Exception as error:
            # A bad snapshot cannot terminate this observer or affect autonomy.
            self.get_logger().error(f'Roadmap update skipped: {error}')

    @staticmethod
    def _summary(builder):
        roadmap, totals = builder.roadmap, builder.totals
        components = len(set(roadmap.components().values()))
        mean_ms = 1000 * totals.seconds / max(1, totals.updates)
        return (f'{builder.mode}: {len(roadmap.nodes)} nodes, {len(roadmap.edge_list())} edges, '
                f'{components} components, removed {totals.removed_nodes} nodes/'
                f'{totals.removed_edges} edges, update {mean_ms:.2f} ms mean '
                f'{1000 * totals.max_seconds:.1f} ms max')

    def _goal(self, message):
        """Plan from the robot to a clicked point over both roadmaps."""
        if self.snapshot is None or self.robot_xy is None:
            self.get_logger().warn('No map or robot pose yet; cannot plan')
            return
        goal = (message.point.x, message.point.y)
        self.paths = {}
        reports = []
        for builder in self.builders:
            path = plan_path(builder.roadmap, self.snapshot, self.robot_xy, goal,
                             builder.attach_radius_m)
            if path is None:
                reason = failure_reason(builder.roadmap, self.snapshot, self.robot_xy, goal,
                                        builder.attach_radius_m)
                reports.append(f'{builder.mode}: no path ({reason})')
                continue
            self.paths[builder.mode] = path
            reports.append(f'{builder.mode}: {path_length(path):.1f} m over {len(path) - 2} nodes '
                           f'({path_length(shortcut(path, self.snapshot)):.1f} m shortcut)')
        self.get_logger().info(f'Roadmap path to ({goal[0]:.1f}, {goal[1]:.1f}): ' + '; '.join(reports))
        stamp = self.get_clock().now().to_msg()
        output = Path()
        output.header.frame_id, output.header.stamp = 'map', stamp
        for x, y in self.paths.get('observed', []):
            pose = PoseStamped()
            pose.header = output.header
            pose.pose.position.x, pose.pose.position.y = float(x), float(y)
            pose.pose.orientation.w = 1.0
            output.poses.append(pose)
        self.path_pub.publish(output)
        self.marker_pub.publish(roadmap_markers(self.builders, self.paths, stamp))

    def save_record(self):
        """Write the inputs seen so far, for ``roadmap_replay``."""
        self.last_saved = time.monotonic()
        with open(self.record_path, 'wb') as handle:
            pickle.dump(self.record, handle)


def main(args=None):
    """Run the independent observer."""
    rclpy.init(args=args)
    node = OnlineRoadmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.record_path:
            node.save_record()
        node.get_logger().info('final: ' + '; '.join(node._summary(b) for b in node.builders))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
