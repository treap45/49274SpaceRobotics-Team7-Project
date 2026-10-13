"""Score recorded tracks against artefacts in mars_cave.sdf (visual centres; GT_REF=origin for model origins)."""
import json, math, os, re, sys
import numpy as np
PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src', 'cave_explorer')
sdf = open(os.path.join(PKG, 'worlds', 'mars_cave.sdf')).read()
CLS = {'green_alien': 'green_alien', 'green_crystals': 'green_crystals', 'white_sphere': 'white_sphere',
       'ice_castles': 'ice_formation', 'mushrooms_blue': 'blue_mushrooms'}
gt = []
def visual_centre(pose, body):
    # Centre of the mesh's bounding box: the ice mesh sits 2.75 m from its model origin.
    r, p, y = pose[3:]
    cx, sx, cy, sy, cz, sz = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    R = (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
         @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))
    uri = re.search(r'<uri>([^<]+)</uri>', body).group(1).replace('package://cave_explorer/', '')
    scale = np.array(list(map(float, re.search(r'<scale>([^<]+)</scale>', body).group(1).split())))
    v = np.array([list(map(float, l.split()[1:4])) for l in open(os.path.join(PKG, uri)) if l.startswith('v ')])
    w = (R @ (v * scale).T).T + pose[:3]
    return (w.min(0) + w.max(0))[:2] / 2
for m in re.finditer(r'<model name="(artifact_\d+)">.*?<pose\s*>([^<]*)</pose>(.*?)</model>', sdf, re.S):
    pose = list(map(float, m.group(2).split()))
    x, y = pose[:2] if os.environ.get('GT_REF') == 'origin' else map(float, visual_centre(pose, m.group(3)))
    c = next(v for k, v in CLS.items() if k in m.group(3))
    gt.append((m.group(1), c, x, y))
tracks = json.load(open(sys.argv[1]))['tracks']
used = set(); rows = []
for name, c, x, y in gt:
    best = min(((math.hypot(t['x'] - x, t['y'] - y), t) for t in tracks
                if t['label'].split()[1] == c and t['id'] not in used), default=None, key=lambda p: p[0])
    if best and best[0] < 3.0:
        used.add(best[1]['id']); rows.append((name, c, x, y, f"{best[0]:.2f} m", best[1]['label']))
    else:
        rows.append((name, c, x, y, 'MISSED', ''))
for r in rows: print('%-12s %-15s (%5.1f,%5.1f)  %-8s %s' % r)
print('found %d/15' % sum(r[4] != 'MISSED' for r in rows))
for t in tracks:
    if t['id'] not in used: print('unmatched track', t)
