"""Thin collected runs into list files; no frame is deleted.

Keeps a frame only when the robot has moved STEP_M or turned STEP_DEG since the
last kept frame of its kind (positive / negative), within each run; the same
rule for every class. Writes <out>/<name>.txt with image paths and prints the
per-class box counts (all / near: box height >= 120 px).

usage: thin.py <data_dir> <out_dir> <name> <step_m> <step_deg> <run> [run ...]
"""
import collections, json, math, os, sys
CLASSES = ['white_sphere', 'blue_mushrooms', 'green_alien', 'ice_formation', 'green_crystals']
MIN_PIXELS, NEG_STEP = 150, (4.0, 90.0)
data, out, name, step_m, step_deg = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
kept = []
for run in sys.argv[6:]:
    last = {True: None, False: None}
    for line in open(os.path.join(data, f'{run}_meta.jsonl')):
        f = json.loads(line)
        positive = any(b[5] >= MIN_PIXELS for b in f['boxes'])
        sm, sd = (step_m, step_deg) if positive else NEG_STEP
        prev = last[positive]
        if prev is not None and math.dist(f['pose'][:2], prev[:2]) < sm \
                and abs(math.degrees(math.remainder(f['pose'][2] - prev[2], math.tau))) < sd:
            continue
        last[positive] = f['pose']
        kept.append(f)
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, name + '.txt'), 'w') as fh:
    fh.writelines(os.path.join(data, 'images', f['name'] + '.jpg') + '\n' for f in kept)
boxes = collections.Counter(b[0] for f in kept for b in f['boxes'] if b[5] >= MIN_PIXELS)
near = collections.Counter(b[0] for f in kept for b in f['boxes'] if b[5] >= MIN_PIXELS and b[4] >= 120)
neg = sum(1 for f in kept if not any(b[5] >= MIN_PIXELS for b in f['boxes']))
print(f'{name}: {len(kept)} frames ({neg} negative); ' +
      ', '.join(f'{c} {boxes[i]} (near {near[i]})' for i, c in enumerate(CLASSES)))
