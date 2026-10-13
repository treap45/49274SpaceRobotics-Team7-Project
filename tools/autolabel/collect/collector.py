"""Save selected camera frames with simulator bounding boxes as YOLO labels.

Pairs /camera/image with /camera/boxes (Gazebo bounding-box camera) and
/camera/labels_map (semantic segmentation camera, pixel value = label); all
three share the pose, intrinsics and update rate, so the stamps are identical.
A box is labelled only if at least MIN_PIXELS of its pixels show the object:
the bounding-box camera also boxes slivers nobody could recognise, such as the
edge of an artefact's base poking through the ground. Every box's visible
pixel count and fill (pixels / box area) go to the meta file, so the manual
review can start with the least filled boxes. A frame is kept only
when the robot has moved or turned enough since the last kept frame of its
kind, so the dataset is not hundreds of near-identical views (task sheet,
Perception 1). Frames without artefacts are kept as negatives, less often.

usage: collector.py <out_dir> <run_name> --ros-args -p use_sim_time:=true
"""
import json, math, os, sys
import cv2, numpy as np, rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from vision_msgs.msg import Detection2DArray

CLASSES = ['white_sphere', 'blue_mushrooms', 'green_alien', 'ice_formation', 'green_crystals']
POSITIVE_STEP = (0.5, math.radians(15))     # metres, radians since the last kept positive
NEGATIVE_STEP = (2.0, math.radians(45))
MIN_PIXELS = 150                            # fewer visible pixels: kept in meta only

OUT, RUN = sys.argv[1], sys.argv[2]


class Collector(Node):
    def __init__(self):
        super().__init__('label_collector')
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(OUT, sub), exist_ok=True)
        self.meta = open(os.path.join(OUT, f'{RUN}_meta.jsonl'), 'a')
        self.buf = Buffer(); TransformListener(self.buf, self)
        self.images, self.boxes, self.maps = {}, {}, {}
        self.last = {True: None, False: None}
        self.count = {True: 0, False: 0}
        self.create_subscription(Image, '/camera/image', self.on_image, 10)
        self.create_subscription(Detection2DArray, '/camera/boxes', self.on_boxes, 10)
        self.create_subscription(Image, '/camera/labels_map', self.on_map, 10)

    @staticmethod
    def key(msg):
        return (msg.header.stamp.sec, msg.header.stamp.nanosec)

    def on_image(self, msg):
        self.images[self.key(msg)] = msg; self.match(self.key(msg))

    def on_boxes(self, msg):
        self.boxes[self.key(msg)] = msg; self.match(self.key(msg))

    def on_map(self, msg):
        self.maps[self.key(msg)] = msg; self.match(self.key(msg))

    def match(self, k):
        stores = (self.images, self.boxes, self.maps)
        if not all(k in d for d in stores):
            for d in stores:                        # drop anything older than 5 frames
                for old in sorted(d)[:-5]:
                    del d[old]
            return
        self.process(*(d.pop(k) for d in stores))

    def pose(self, stamp):
        try:
            t = self.buf.lookup_transform('odom', 'base_link', Time())
        except Exception:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return (t.transform.translation.x, t.transform.translation.y, yaw)

    def process(self, image, boxes, labels_map):
        pose = self.pose(image.header.stamp)
        if pose is None:
            return
        w, h = image.width, image.height
        seg = np.frombuffer(labels_map.data, np.uint8).reshape(h, w, -1)[:, :, 0]
        raw, label = [], []
        for det in boxes.detections:
            cls = int(det.results[0].hypothesis.class_id) - 1
            b = det.bbox
            cx, cy, bw, bh = b.center.position.x, b.center.position.y, b.size_x, b.size_y
            x0, y0 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
            x1, y1 = min(w, int(math.ceil(cx + bw / 2))), min(h, int(math.ceil(cy + bh / 2)))
            pixels = int(np.count_nonzero(seg[y0:y1, x0:x1] == cls + 1))
            fill = pixels / max(1.0, bw * bh)
            raw.append([cls, cx, cy, bw, bh, pixels, round(fill, 3)])
            if 0 <= cls < len(CLASSES) and pixels >= MIN_PIXELS:
                label.append(f'{cls} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}')
        positive = bool(label)
        last = self.last[positive]
        step = POSITIVE_STEP if positive else NEGATIVE_STEP
        if last is not None and math.dist(pose[:2], last[:2]) < step[0] \
                and abs(math.remainder(pose[2] - last[2], math.tau)) < step[1]:
            return
        self.last[positive] = pose
        self.count[positive] += 1
        name = f'{RUN}_{sum(self.count.values()):05d}'
        frame = np.frombuffer(image.data, np.uint8).reshape(h, w, -1)
        if image.encoding == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUT, 'images', name + '.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(OUT, 'labels', name + '.txt'), 'w') as f:
            f.write('\n'.join(label) + ('\n' if label else ''))
        stamp = image.header.stamp.sec + image.header.stamp.nanosec * 1e-9
        self.meta.write(json.dumps(dict(name=name, stamp=stamp, pose=pose, boxes=raw)) + '\n')
        self.meta.flush()
        if sum(self.count.values()) % 25 == 0:
            self.get_logger().info(f'kept {self.count[True]} positive, {self.count[False]} negative')


rclpy.init(); node = Collector()
try: rclpy.spin(node)
except BaseException: pass
