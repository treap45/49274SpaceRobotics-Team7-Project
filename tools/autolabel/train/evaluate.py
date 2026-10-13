"""Compare detectors on the test split at the node's operating confidence.

For each model: mAP50 / mAP50-95 from ultralytics val, then at CONF (0.5, as
cave_explorer passes it) per-class precision / recall with IoU >= 0.5 matching,
and a confusion table: a prediction that overlaps a ground-truth box of another
class (IoU >= 0.5) counts as that confusion; one overlapping nothing is
'background'. Ground truth of the test split is the reviewed auto-labels.

usage: evaluate.py <data.yaml> <model.pt> [model.pt ...]
"""
import os, sys
import numpy as np
from ultralytics import YOLO
CONF, IOU = 0.5, 0.5
yaml_path, models = sys.argv[1], sys.argv[2:]
root = os.path.dirname(yaml_path)
test = [l.strip() for l in open(os.path.join(root, 'test.txt')) if l.strip()]

def gt(path, w=720, h=480):
    out = []
    for l in open(path.replace('/images/', '/labels/')[:-4] + '.txt'):
        if l.strip():
            c, cx, cy, bw, bh = map(float, l.split())
            out.append((int(c), (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)

for model_path in models:
    model = YOLO(model_path)
    names = [model.names[i] for i in range(len(model.names))]
    n = len(names)
    m = model.val(data=yaml_path, split='test', verbose=False, plots=False)
    print(f'\n=== {model_path}\nmAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}  (test, conf sweep)')
    tp = np.zeros(n); fp = np.zeros(n); fn = np.zeros(n)
    confusion = np.zeros((n, n + 1), int)         # [true class, predicted class | background]
    for k in range(0, len(test), 32):
        for path, res in zip(test[k:k + 32], model.predict(test[k:k + 32], conf=CONF, verbose=False)):
            truth = gt(path); used = set()
            preds = sorted(zip(res.boxes.cls.int().tolist(), res.boxes.xyxy.tolist(), res.boxes.conf.tolist()),
                           key=lambda p: -p[2])
            for c, box, _ in preds:
                same = [(iou(box, t[1:]), j) for j, t in enumerate(truth) if t[0] == c and j not in used]
                best = max(same, default=(0, None))
                if best[0] >= IOU:
                    tp[c] += 1; used.add(best[1]); continue
                fp[c] += 1
                other = max(((iou(box, t[1:]), t[0]) for t in truth if t[0] != c), default=(0, None))
                if other[0] >= IOU:
                    confusion[other[1], c] += 1
                else:
                    confusion[c, n] += 1     # predicted c on background (row = predicted class here)
            for j, t in enumerate(truth):
                if j not in used:
                    fn[t[0]] += 1
    print(f'at conf {CONF}: class          P      R     TP   FP   FN')
    for i, name in enumerate(names):
        p = tp[i] / max(1, tp[i] + fp[i]); r = tp[i] / max(1, tp[i] + fn[i])
        print(f'  {name:15s} {p:6.2f} {r:6.2f} {int(tp[i]):5d} {int(fp[i]):4d} {int(fn[i]):4d}')
    print('confusions (true -> predicted): ' + ', '.join(
        f'{names[i]}->{names[j]} {confusion[i, j]}' for i in range(n) for j in range(n) if confusion[i, j]) or 'none')
    print('false on background (predicted class: count): ' + ', '.join(
        f'{names[i]} {confusion[i, n]}' for i in range(n) if confusion[i, n]) or 'none')
