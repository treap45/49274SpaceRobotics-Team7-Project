"""Save the latest artefact markers (track positions + labels) to a JSON file."""
import json, os, sys, rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
OUT = sys.argv[1]
class M(Node):
    def __init__(self):
        super().__init__('marker_logger')
        self.create_subscription(MarkerArray, 'marker_array_artifacts', self.cb, 10)
    def cb(self, msg):
        pos, lab = {}, {}
        for m in msg.markers:
            if m.type == Marker.SPHERE: pos[m.id // 2] = (m.pose.position.x, m.pose.position.y)
            elif m.type == Marker.TEXT_VIEW_FACING: lab[m.id // 2] = m.text
        d = [dict(id=k, x=pos[k][0], y=pos[k][1], label=lab.get(k, '')) for k in pos]
        with open(OUT + '.tmp', 'w') as f: json.dump(dict(t=self.get_clock().now().nanoseconds/1e9, tracks=d), f)
        os.replace(OUT + '.tmp', OUT)
rclpy.init(); n = M()
try: rclpy.spin(n)
except BaseException: pass
