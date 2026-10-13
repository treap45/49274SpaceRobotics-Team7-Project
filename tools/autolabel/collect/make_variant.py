"""Move every artefact of a labelled world to a random free place, for training data only.

Free space comes from a SLAM map the robot built (recorder pickle), not from the
world's walls. The same rules hold for every class: at least CLEARANCE_M from
any occupied or unknown cell, MIN_SPACING_M between artefacts and START_CLEAR_M
from the robot's start; random yaw; each class keeps its original height (the
meshes' origins differ). Writes the world and a JSON of the new positions.

usage: make_variant.py <labelled.sdf> <record.pkl> <seed> <out.sdf>
"""
import json, math, pickle, re, sys
import numpy as np
from scipy import ndimage

CLEARANCE_M, MIN_SPACING_M, START_CLEAR_M = 3.0, 8.0, 5.0
src, record, seed, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
rng = np.random.default_rng(seed)
m = pickle.load(open(record, 'rb'))['maps'][-1]
grid, res, (ox, oy) = m['grid'], m['res'], m['origin']
clearance = ndimage.distance_transform_edt(grid == 0) * res     # unknown counts as blocked
rows, cols = np.nonzero(clearance >= CLEARANCE_M)
candidates = np.column_stack([ox + (cols + 0.5) * res, oy + (rows + 0.5) * res])
candidates = candidates[np.hypot(*candidates.T) >= START_CLEAR_M]

world = open(src).read()
models = list(re.finditer(r'(<model name="(artifact_\d+)">\s*<pose\s*>)([^<]*)(</pose>)', world))
placed, moved = [], {}
for match in rng.permutation(len(models)):
    mm = models[match]
    for _ in range(2000):
        x, y = candidates[rng.integers(len(candidates))]
        if all(math.dist((x, y), p) >= MIN_SPACING_M for p in placed):
            break
    else:
        sys.exit('no room left; lower MIN_SPACING_M')
    placed.append((x, y))
    pose = mm.group(3).split()
    pose[0], pose[1], pose[5] = f'{x:.2f}', f'{y:.2f}', f'{rng.uniform(-math.pi, math.pi):.4f}'
    moved[mm.group(2)] = (mm.start(3), mm.end(3), ' '.join(pose))
for name, (a, b, pose) in sorted(moved.items(), key=lambda kv: -kv[1][0]):
    world = world[:a] + pose + world[b:]
open(out, 'w').write(world)
json.dump({k: v[2] for k, v in moved.items()}, open(out.replace('.sdf', '.json'), 'w'), indent=1)
print(f'{len(moved)} artefacts moved, {len(candidates)} candidate cells')
