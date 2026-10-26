"""Score detectors on the test split for the report and save the numbers as JSON.

For each model: per-class AP50 (all-point interpolated, from predictions at
conf 0.001), per-class precision / recall at the node's operating confidence
0.5 with same-class IoU >= 0.5 matching, and a confusion matrix at conf 0.5
(rows: true class + background, columns: predicted class + missed; boxes are
paired by IoU >= 0.5 whatever their class).

Images are read with OpenCV (BGR, as the node gets them from cv_bridge). A
model given with ':swap' sees them channel-swapped, which is how the node feeds
a model when detector_input_bgr is false (the earlier V4 model was trained so).

usage: detector_report.py <dataset_dir> <out.json> <name>=<model.pt>[:swap] ...
"""
import json, os, sys
import cv2
import numpy as np
from ultralytics import YOLO

OP_CONF, IOU, LOW_CONF = 0.5, 0.5, 0.001
root, out_path, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
img_dir = os.path.join(root, 'images', 'test')
names_files = sorted(os.listdir(img_dir))


def truth(name, w, h):
    out = []
    for line in open(os.path.join(root, 'labels', 'test', name[:-4] + '.txt')):
        if line.strip():
            c, cx, cy, bw, bh = map(float, line.split())
            out.append((int(c), np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])))
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def average_precision(scored, n_gt):
    if n_gt == 0 or not scored:
        return 0.0
    scored = sorted(scored, key=lambda s: -s[0])
    hits = np.array([s[1] for s in scored], float)
    tp = np.cumsum(hits); fp = np.cumsum(1 - hits)
    recall = tp / n_gt; precision = tp / (tp + fp)
    r = np.concatenate(([0.0], recall, [1.0])); p = np.concatenate(([1.0], precision, [0.0]))
    p = np.flip(np.maximum.accumulate(np.flip(p)))
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


results = {}
for spec in specs:
    label, rest = spec.split('=', 1)
    swap = rest.endswith(':swap')
    model = YOLO(rest[:-5] if swap else rest)
    classes = [model.names[i] for i in range(len(model.names))]
    n = len(classes)
    scored = [[] for _ in range(n)]                # (conf, is_tp) per predicted class
    n_gt = np.zeros(n, int)
    tp = np.zeros(n, int); fp = np.zeros(n, int); fn = np.zeros(n, int)
    confusion = np.zeros((n + 1, n + 1), int)      # [true | background, predicted | missed]
    for k in range(0, len(names_files), 32):
        batch = names_files[k:k + 32]
        images = [cv2.imread(os.path.join(img_dir, f)) for f in batch]
        if swap:
            images = [im[:, :, ::-1].copy() for im in images]
        for f, im, res in zip(batch, images, model.predict(images, conf=LOW_CONF, verbose=False)):
            gt = truth(f, im.shape[1], im.shape[0])
            for c, _ in gt:
                n_gt[c] += 1
            preds = sorted(zip(res.boxes.cls.int().tolist(), res.boxes.xyxy.cpu().numpy(),
                               res.boxes.conf.tolist()), key=lambda p: -p[2])
            # AP: every prediction, same-class matching in confidence order.
            used = set()
            for c, box, conf in preds:
                cand = [(iou(box, g[1]), j) for j, g in enumerate(gt) if g[0] == c and j not in used]
                best = max(cand, default=(0.0, None))
                hit = best[0] >= IOU
                if hit:
                    used.add(best[1])
                scored[c].append((conf, hit))
            # Operating point.
            op = [p for p in preds if p[2] >= OP_CONF]
            used = set()
            for c, box, _ in op:
                cand = [(iou(box, g[1]), j) for j, g in enumerate(gt) if g[0] == c and j not in used]
                best = max(cand, default=(0.0, None))
                if best[0] >= IOU:
                    tp[c] += 1; used.add(best[1])
                else:
                    fp[c] += 1
            for j, g in enumerate(gt):
                if j not in used:
                    fn[g[0]] += 1
            # Confusion: pair boxes by IoU regardless of class, largest overlap first.
            pairs = sorted(((iou(p[1], g[1]), i, j) for i, p in enumerate(op) for j, g in enumerate(gt)),
                           reverse=True)
            pi, gj = set(), set()
            for v, i, j in pairs:
                if v < IOU:
                    break
                if i in pi or j in gj:
                    continue
                pi.add(i); gj.add(j); confusion[gt[j][0], op[i][0]] += 1
            for j, g in enumerate(gt):
                if j not in gj:
                    confusion[g[0], n] += 1
            for i, p in enumerate(op):
                if i not in pi:
                    confusion[n, p[0]] += 1
    ap = [average_precision(scored[c], n_gt[c]) for c in range(n)]
    results[label] = {
        'model': rest, 'swap': swap, 'classes': classes, 'images': len(names_files),
        'gt': n_gt.tolist(), 'ap50': ap, 'map50': float(np.mean(ap)),
        'tp': tp.tolist(), 'fp': fp.tolist(), 'fn': fn.tolist(),
        'precision': [float(tp[c] / max(1, tp[c] + fp[c])) for c in range(n)],
        'recall': [float(tp[c] / max(1, tp[c] + fn[c])) for c in range(n)],
        'confusion': confusion.tolist(),
    }
    print(f'{label}: mAP50 {np.mean(ap):.3f}  ' + '  '.join(
        f'{classes[c]} AP {ap[c]:.2f} P {results[label]["precision"][c]:.2f} R {results[label]["recall"][c]:.2f}'
        for c in range(n)))
json.dump(results, open(out_path, 'w'), indent=1)
