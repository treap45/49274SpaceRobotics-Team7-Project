"""Contact sheets of labelled frames: YOLO boxes drawn with class names.

usage: preview.py <data_dir> <out_prefix> [names...]   (default: a spread over all frames)
Boxes too small to label are drawn thin grey; labelled ones show their fill ratio.
"""
import json, glob, os, sys
import cv2, numpy as np
CLASSES = ['white_sphere', 'blue_mushrooms', 'green_alien', 'ice_formation', 'green_crystals']
COLOURS = [(255, 255, 255), (255, 120, 0), (0, 255, 0), (255, 255, 0), (0, 200, 255)]
data, prefix = sys.argv[1], sys.argv[2]
meta = {}
for path in glob.glob(os.path.join(data, '*_meta.jsonl')):
    for line in open(path):
        m = json.loads(line); meta[m['name']] = m
names = sys.argv[3:] or sorted(meta)[::max(1, len(meta) // 24)][:24]
tiles = []
for name in names:
    img = cv2.imread(os.path.join(data, 'images', name + '.jpg'))
    h, w = img.shape[:2]
    for c, cx, cy, bw, bh, *extra in meta[name]['boxes']:
        c = int(c); pixels, fill = extra if extra else (None, None)
        labelled = pixels is None or pixels >= 150
        x0, y0, x1, y1 = int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)
        colour = COLOURS[c] if labelled else (128, 128, 128)
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, 2 if labelled else 1)
        if labelled:
            text = CLASSES[c] + ('' if fill is None else f' {fill:.2f}')
            cv2.putText(img, text, (x0, max(15, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    cv2.putText(img, name, (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    tiles.append(cv2.resize(img, (w // 2, h // 2)))
for k in range(0, len(tiles), 12):
    page = tiles[k:k + 12]
    page += [np.zeros_like(tiles[0])] * (12 - len(page))
    rows = [np.hstack(page[r * 3:(r + 1) * 3]) for r in range(4)]
    out = f'{prefix}_{k // 12}.jpg'; cv2.imwrite(out, np.vstack(rows)); print(out)
