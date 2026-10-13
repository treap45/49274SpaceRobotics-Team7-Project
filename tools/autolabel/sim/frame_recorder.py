"""Record every camera frame with its depth image and map <- optical-frame transform.

For offline replay of the detector and ArtifactTracker: <out>/frames/<stamp_ns>.jpg
(BGR), <stamp_ns>.npz (depth float16, transform). Frames whose depth or TF never
arrives within 2 s are skipped.

usage: frame_recorder.py <out_dir> --ros-args -p use_sim_time:=true
"""
import os, sys
import cv2, numpy as np, rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

OUT = os.path.join(sys.argv[1], 'frames')
OPTICAL = 'camera_rgb_optical_frame'


class Recorder(Node):
    def __init__(self):
        super().__init__('frame_recorder')
        os.makedirs(OUT, exist_ok=True)
        self.bridge = CvBridge(); self.buf = Buffer(); TransformListener(self.buf, self)
        self.rgb, self.depth = {}, {}
        self.create_subscription(Image, '/camera/image', self.on_rgb, 10)
        self.create_subscription(Image, '/camera/depth/image', self.on_depth, 10)
        self.create_timer(0.2, self.flush)
        self.saved = 0

    def on_rgb(self, m):
        self.rgb[Time.from_msg(m.header.stamp).nanoseconds] = self.bridge.imgmsg_to_cv2(m, 'bgr8')

    def on_depth(self, m):
        self.depth[Time.from_msg(m.header.stamp).nanoseconds] = self.bridge.imgmsg_to_cv2(m, '32FC1')

    def flush(self):
        now = self.get_clock().now().nanoseconds
        for k in sorted(set(self.rgb) & set(self.depth)):
            try:
                t = self.buf.lookup_transform('map', OPTICAL, Time(nanoseconds=k))
            except TransformException:
                if now - k > 2e9:
                    self.rgb.pop(k); self.depth.pop(k)
                continue
            tr, q = t.transform.translation, t.transform.rotation
            cv2.imwrite(os.path.join(OUT, f'{k}.jpg'), self.rgb.pop(k), [cv2.IMWRITE_JPEG_QUALITY, 95])
            np.savez_compressed(os.path.join(OUT, f'{k}.npz'), depth=self.depth.pop(k).astype(np.float16),
                                translation=[tr.x, tr.y, tr.z], rotation=[q.x, q.y, q.z, q.w])
            self.saved += 1
        for d in (self.rgb, self.depth):
            for k in [k for k in d if now - k > 3e9]:
                d.pop(k)


rclpy.init(); n = Recorder()
try: rclpy.spin(n)
except BaseException: pass
