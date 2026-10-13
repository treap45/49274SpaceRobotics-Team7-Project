"""Final YOLO split lists from the thinned lists in splits/.

Leaves out (1) frames marked in any review JSON and (2) frames with a labelled
box lying mostly (> half its area) inside a larger box of the same class: the
segmentation camera is semantic, so the visible-pixel count of such a box also
counts the other object's pixels and the MIN_PIXELS rule cannot be trusted
(found in review: a far green_alien hidden behind a near one). The rule is the
same for every class. No image is deleted.

usage: finalise_splits.py <dataset_dir> <out_dir> [review.json ...]
"""
import glob, json, os, sys
CLASSES = ['white_sphere', 'blue_mushrooms', 'green_alien', 'ice_formation', 'green_crystals']
MIN_PIXELS = 150
data, out = sys.argv[1], os.path.abspath(sys.argv[2])
reviewed_bad = set()
for path in sys.argv[3:]:
    reviewed_bad |= set(json.load(open(path))['problems'])
meta = {}
for p in glob.glob(os.path.join(data, '*', '*_meta.jsonl')):
    for line in open(p):
        m = json.loads(line); meta[m['name']] = m

def overlap(a, b):
    w = min(a[1] + a[3] / 2, b[1] + b[3] / 2) - max(a[1] - a[3] / 2, b[1] - b[3] / 2)
    h = min(a[2] + a[4] / 2, b[2] + b[4] / 2) - max(a[2] - a[4] / 2, b[2] - b[4] / 2)
    return max(0, w) * max(0, h)

def same_class_overlap(name):
    boxes = [b for b in meta[name]['boxes'] if b[5] >= MIN_PIXELS]
    return any(o[0] == b[0] and o[3] * o[4] > b[3] * b[4] and overlap(b, o) > 0.5 * b[3] * b[4]
               for b in boxes for o in boxes if o is not b)

os.makedirs(out, exist_ok=True)
for split in ('train', 'val', 'test'):
    paths = [l.strip() for l in open(os.path.join(data, 'splits', f'{split}.txt')) if l.strip()]
    names = [os.path.basename(p)[:-4] for p in paths]
    dropped_review = [n for n in names if n in reviewed_bad]
    dropped_overlap = [n for n in names if n not in reviewed_bad and same_class_overlap(n)]
    keep = [p for p, n in zip(paths, names) if n not in reviewed_bad and n not in dropped_overlap]
    open(os.path.join(out, f'{split}.txt'), 'w').write('\n'.join(keep) + '\n')
    boxes = [0] * len(CLASSES); negatives = 0
    for p in keep:
        lab = [b for b in meta[os.path.basename(p)[:-4]]['boxes'] if b[5] >= MIN_PIXELS]
        negatives += not lab
        for b in lab: boxes[b[0]] += 1
    print(f'{split}: {len(keep)} of {len(paths)} kept (review -{len(dropped_review)}, '
          f'same-class overlap -{len(dropped_overlap)}); {negatives} negative; boxes',
          dict(zip(CLASSES, boxes)))
with open(os.path.join(out, 'data.yaml'), 'w') as f:
    f.write(f'path: {out}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n')
    f.writelines(f'  {i}: {c}\n' for i, c in enumerate(CLASSES))
