"""Replay recorded frames through the detector and ArtifactTracker variants.

Ground truth (artefact poses in mars_cave.sdf) is used only to score the
result, never by the methods themselves.

usage: replay.py <run_dir> <model.pt> [variant ...]
  first run caches detections in <run_dir>/detections.json
"""
import glob, json, math, os, re, sys
import cv2, numpy as np
PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src', 'cave_explorer')
sys.path.insert(0, PKG)
from cave_explorer.artifact_tracker import ArtifactTracker, SIM_CAMERA_MODEL
from cave_explorer.interfaces import Detection2D, DetectionBatch

run, model = sys.argv[1], sys.argv[2]
variants = sys.argv[3:] or ['surface']
frames = sorted(glob.glob(os.path.join(run, 'frames', '*.npz')), key=lambda p: int(os.path.basename(p)[:-4]))

cache = os.path.join(run, 'detections.json')
if not os.path.exists(cache):
    from cave_explorer.perception import Detector
    det = Detector(model, 0.5); out = {}
    for p in frames:
        k = int(os.path.basename(p)[:-4])
        b = det.detect(cv2.imread(p[:-4] + '.jpg'), k, 'camera_link')     # BGR, as the model needs
        out[k] = [(d.class_name, d.confidence, list(d.bbox_xyxy)) for d in b.detections]
    json.dump(out, open(cache, 'w'))
dets = {int(k): v for k, v in json.load(open(cache)).items()}

sdf = open(os.path.join(PKG, 'worlds', 'mars_cave.sdf')).read()
CLS = {'green_alien': 'green_alien', 'green_crystals': 'green_crystals', 'white_sphere': 'white_sphere',
       'ice_castles': 'ice_formation', 'mushrooms_blue': 'blue_mushrooms'}
def _rot(r, p, y):
    cx, sx, cy, sy, cz, sz = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))
def _visual_centre(pose, body):
    # Centre of the mesh's bounding box in the world: the ice mesh sits 2.75 m from its model origin.
    uri = re.search(r'<uri>([^<]+)</uri>', body).group(1).replace('package://cave_explorer/', '')
    scale = np.array(list(map(float, re.search(r'<scale>([^<]+)</scale>', body).group(1).split())))
    v = np.array([list(map(float, l.split()[1:4])) for l in open(os.path.join(PKG, uri)) if l.startswith('v ')])
    w = (_rot(*pose[3:]) @ (v * scale).T).T + pose[:3]
    return (w.min(0) + w.max(0))[:2] / 2
REF = os.environ.get('GT_REF', 'visual')
GT = []
for m in re.finditer(r'<model name="(artifact_\d+)">.*?<pose\s*>([^<]*)</pose>(.*?)</model>', sdf, re.S):
    pose = list(map(float, m.group(2).split()))
    x, y = _visual_centre(pose, m.group(3)) if REF == 'visual' else pose[:2]
    GT.append((next(v for k, v in CLS.items() if k in m.group(3)), float(x), float(y)))
ORDER = ['white_sphere', 'blue_mushrooms', 'green_alien', 'ice_formation', 'green_crystals']


class Variant(ArtifactTracker):
    """ArtifactTracker whose observation point comes from a named method."""

    def __init__(self, method, **kw):
        super().__init__(**kw); self.method = method; self.obs = []

    def update(self, detections, depth, camera_model, transform):
        transform = self._as_transform_dict(transform)
        for d in detections.detections:
            sample, _ = self._depth_in_bbox(depth, d.bbox_xyxy)
            if sample is None:
                continue
            depth_m, u, v = sample
            x, y, z = self._pixel_to_camera(u, v, depth_m, camera_model)
            width = (d.bbox_xyxy[2] - d.bbox_xyxy[0]) * depth_m / camera_model['fx']
            push = 0.0
            if self.method == 'pushback':            # surface point -> centre: half a width further along the ray
                push = width / 2
            elif self.method == 'quarter':
                push = width / 4
            elif self.method == 'band':              # half the visible width of the nearest depth band
                push = self._band_width(depth, d.bbox_xyxy, depth_m, camera_model) / 2
            if push:
                n = math.hypot(x, z)
                x, z = x + x / n * push, z + z / n * push
            p = self._transform_point((x, y, z), transform)
            self.obs.append((d.class_name, p[0], p[1], depth_m))
            self.update_observation(d.class_name, p, detections.stamp_ns, width)
        self._drop_stale_provisional(detections.stamp_ns)
        if os.environ.get('MERGE'):
            self._merge_converged()

    def _merge_converged(self):
        # Same-class tracks whose positions have converged within the gate: keep the older one.
        merged = True
        while merged:
            merged = False
            for a in self._tracks:
                for b in self._tracks:
                    if a is b or a.class_name != b.class_name or a.artifact_id is None or b.artifact_id is None:
                        continue
                    if b.artifact_id <= a.artifact_id:
                        continue
                    if math.dist(a.position_map, b.position_map) <= self._gate_m(a, b.width()):
                        n, m = a.observation_count, b.observation_count
                        a.position_map = tuple((x * n + y * m) / (n + m) for x, y in zip(a.position_map, b.position_map))
                        a.observation_count = n + m
                        a.last_seen_ns = max(a.last_seen_ns, b.last_seen_ns)
                        a.widths = (a.widths + b.widths)[-self.width_history:]
                        self._tracks.remove(b); self.retired = getattr(self, 'retired', []) + [(b.artifact_id, a.artifact_id)]
                        merged = True
                        break
                if merged:
                    break


def _band_width(self, depth, bbox, depth_m, cam):
    # Lateral extent (m) of the pixels whose depth is within one band of the selected depth, across the whole box.
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    win = depth[max(0, y0):y1, max(0, x0):x1]
    cols = np.nonzero(np.abs(win - depth_m) <= self.depth_band_width_m)[1]
    if cols.size < 4:
        return 0.0
    lo, hi = np.percentile(cols, [5, 95])
    return (hi - lo) * depth_m / cam['fx']
Variant._band_width = _band_width


def score(tracks):
    used, found = set(), 0
    for c, x, y in GT:
        best = min(((math.hypot(t.position_map[0] - x, t.position_map[1] - y), i) for i, t in enumerate(tracks)
                    if t.class_name == c and i not in used), default=None)
        if best and best[0] < 3.0:
            used.add(best[1]); found += 1
    dup = sum(1 for i, t in enumerate(tracks) if i not in used and any(
        g[0] == t.class_name and math.hypot(g[1] - t.position_map[0], g[2] - t.position_map[1]) < 5 for g in GT))
    return found, dup, len(tracks) - found - dup


class Repo(ArtifactTracker):
    """The tracker as implemented in the repository, recording each observation."""

    def __init__(self, **kw):
        super().__init__(**kw); self.obs = []

    def update_observation(self, class_name, position_map, stamp_ns, width_m=0.0):
        self.obs.append((class_name, position_map[0], position_map[1], 0.0))
        return super().update_observation(class_name, position_map, stamp_ns, width_m)


for method in variants:
    tr = Repo() if method == 'repo' else Repo(surface_to_centre=False) if method == 'repo-surface' else Variant(method)
    for p in frames:
        k = int(os.path.basename(p)[:-4])
        if not dets.get(k):
            continue
        z = np.load(p)
        batch = DetectionBatch(k, 'camera_link', 720, 480,
                               tuple(Detection2D(c, conf, tuple(b)) for c, conf, b in dets[k]))
        tr.update(batch, z['depth'].astype(np.float32),
                  SIM_CAMERA_MODEL, {'translation': tuple(z['translation']), 'rotation': tuple(z['rotation'])})
    errs = {c: [] for c in ORDER}
    for c, x, y, d in tr.obs:
        e = min((math.hypot(g[1] - x, g[2] - y) for g in GT if g[0] == c), default=99)
        if e < 6:
            errs[c].append(e)
    f, dup, false = score(tr.snapshot())
    if os.environ.get('SHOW_TRACKS'):
        for t in tr.snapshot():
            g = min((math.hypot(g[1] - t.position_map[0], g[2] - t.position_map[1]), i) for i, g in enumerate(GT) if g[0] == t.class_name)
            inner = next(x for x in tr._tracks if x.artifact_id == t.artifact_id)
            print(f'   {t.artifact_id:3d} {t.class_name:15s} ({t.position_map[0]:6.1f},{t.position_map[1]:6.1f}) '
                  f'nearest GT #{g[1]} {g[0]:.2f} m  obs {t.observation_count:3d}  width {inner.width():.2f}')
    print(f'{method:9s} found {f:2d}/15  duplicate {dup:2d}  false {false:2d}  | median observation error (m): ' +
          '  '.join(f'{c.split("_")[-1][:6]} {np.median(v):.2f}' if v else f'{c} -' for c, v in errs.items()))
