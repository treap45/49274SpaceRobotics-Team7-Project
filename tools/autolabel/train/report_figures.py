"""Report figures for Perception 1 and 2.

  dataset_examples.png    one labelled frame per class plus a negative
  detector_pr.png         per-class AP50 / precision / recall, autolabel vs V4
  detector_confusion.png  confusion matrices at conf 0.5, autolabel vs V4

usage: report_figures.py <dataset_dir> <detector_test_results.json> <out_dir>
"""
import json, os, sys
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

root, results_path, out = sys.argv[1:4]
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']   # categorical slots 1-5
BLUES = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#104281']
CLASSES = ['white_sphere', 'blue_mushrooms', 'green_alien', 'ice_formation', 'green_crystals']
SHORT = ['sphere', 'mushrooms', 'alien', 'ice', 'crystals']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': INK,
                     'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})

# Perception 1: one frame per class (single-object frames, box 110-260 px tall) and a negative.
EXAMPLES = [('train', 'variant_2_00359'), ('train', 'variant_2_00333'), ('val', 'variant_4_00613'),
            ('train', 'variant_3_00316'), ('train', 'variant_1_00401'), ('train', 'variant_1_00030')]
SOURCE = {'train': 'train, random layout', 'val': 'val, random layout', 'test': 'test, original layout'}
fig, axes = plt.subplots(2, 3, figsize=(12, 5.6))
for ax, (split, name) in zip(axes.flat, EXAMPLES):
    im = cv2.cvtColor(cv2.imread(os.path.join(root, 'images', split, name + '.jpg')), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    ax.imshow(im)
    labels = [l.split() for l in open(os.path.join(root, 'labels', split, name + '.txt')) if l.strip()]
    for c, cx, cy, bw, bh in labels:
        c = int(c); cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        ax.add_patch(plt.Rectangle((cx - bw / 2, cy - bh / 2), bw, bh, fill=False, lw=2.5, ec=SERIES[c]))
        ax.text(cx - bw / 2, cy - bh / 2 - 6, CLASSES[c], fontsize=9, color=INK, va='bottom',
                bbox=dict(boxstyle='square,pad=0.2', fc='white', ec=SERIES[c], lw=1.5))
    title = CLASSES[int(labels[0][0])] if labels else 'negative (no artefact)'
    variant = name.split('_')[1] if name.startswith('variant') else ''
    ax.set_title(f'{title}  ({SOURCE[split]}{" " + variant if variant else ""})', fontsize=10, color=INK)
    ax.set_axis_off()
fig.tight_layout()
fig.savefig(os.path.join(out, 'dataset_examples.png'), dpi=200)
plt.close(fig)

# Perception 2: test-split comparison. V4 is shown at the channel order it scored best with.
res = json.load(open(results_path))
v4 = max((k for k in res if k.startswith('V4')), key=lambda k: res[k]['map50'])
runs = [('autolabel (ours)', res['autolabel'], SERIES[0]), ('V4', res[v4], SERIES[1])]
metrics = [('ap50', 'AP50'), ('precision', 'Precision at conf 0.5'), ('recall', 'Recall at conf 0.5')]
x = np.arange(len(CLASSES)); width = 0.36
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
for ax, (key, label) in zip(axes, metrics):
    for i, (name, r, color) in enumerate(runs):
        ax.bar(x + (i - 0.5) * width, r[key], width, color=color, ec='white', lw=2, label=name, zorder=3)
        for xi, v in zip(x, r[key]):
            if v < 0.005:                      # mark zero bars so they do not read as missing
                ax.text(xi + (i - 0.5) * width, 0.01, '0', ha='center', va='bottom', fontsize=8, color=INK2)
    means = '   '.join(f'{name.split()[0]} {np.mean(r[key]):.2f}' for name, r, _ in runs)
    ax.set_title(f'{label}\nmean: {means}', fontsize=10, color=INK)
    ax.set_xticks(x, SHORT, fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
    ax.tick_params(length=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
handles, names = axes[0].get_legend_handles_labels()
fig.legend(handles, names, loc='upper center', ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(out, 'detector_pr.png'), dpi=200)
plt.close(fig)

cmap = LinearSegmentedColormap.from_list('blue', BLUES)
rows = CLASSES + ['background']; cols = CLASSES + ['missed']
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, (name, r, _) in zip(axes, runs):
    m = np.array(r['confusion'], float)
    shade = m / np.maximum(1, m.sum(axis=1, keepdims=True))       # row share drives colour
    shade[-1] = m[-1] / max(1, m[:-1].sum(axis=0).max())
    shade = np.ma.masked_array(shade, mask=np.zeros_like(shade, bool)); shade.mask[-1, -1] = True
    ax.imshow(shade, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if i == len(rows) - 1 and j == len(cols) - 1:
                continue
            v = int(m[i, j])
            ax.text(j, i, str(v) if v else '·', ha='center', va='center', fontsize=10,
                    color='white' if shade[i, j] > 0.55 else (INK if v else MUTED))
    ax.set_xticks(range(len(cols)), [c.replace('_', '\n') for c in cols], fontsize=9)
    ax.set_yticks(range(len(rows)), [c.replace('_', ' ') for c in rows], fontsize=9)
    ax.set_xlabel('predicted'); ax.set_ylabel('true')
    ax.set_title(f'{name}   (mAP50 {r["map50"]:.2f})', color=INK, fontsize=11)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True); ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which='minor', color='white', lw=2); ax.tick_params(which='minor', length=0)
fig.tight_layout()
fig.savefig(os.path.join(out, 'detector_confusion.png'), dpi=200)
plt.close(fig)
print('V4 shown:', v4)
