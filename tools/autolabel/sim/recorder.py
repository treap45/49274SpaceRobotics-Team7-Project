"""Record /map every 10 s and the robot pose every 1 s."""
import math, os, pickle, sys
import numpy as np, rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
OUT = sys.argv[1]
class Recorder(Node):
    def __init__(self):
        super().__init__('tuning_recorder')
        self.buf = Buffer(); TransformListener(self.buf, self)
        self.map = None; self.poses = []; self.maps = []
        self.create_subscription(OccupancyGrid, 'map', lambda m: setattr(self, 'map', m), 1)
        self.create_timer(1.0, self.on_pose); self.create_timer(10.0, self.on_save)
    def now(self): return self.get_clock().now().nanoseconds / 1e9
    def on_pose(self):
        try: t = self.buf.lookup_transform('map', 'base_link', Time())
        except Exception: return
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.poses.append((self.now(), t.transform.translation.x, t.transform.translation.y, yaw))
    def on_save(self):
        if self.map is None: return
        i = self.map.info
        self.maps.append(dict(t=self.now(), grid=np.asarray(self.map.data, dtype=np.int8).reshape(i.height, i.width),
                              res=i.resolution, origin=(i.origin.position.x, i.origin.position.y)))
        with open(OUT + '.tmp', 'wb') as f: pickle.dump(dict(maps=self.maps, poses=self.poses), f)
        os.replace(OUT + '.tmp', OUT)
rclpy.init(); node = Recorder()
try: rclpy.spin(node)
except BaseException: pass
