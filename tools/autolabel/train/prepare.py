"""Write YOLO split lists and data.yaml.

train = layouts listed in TRAIN, val = VAL (early stopping / model choice),
test = the original layout, never trained on. Frames marked in a review.json
are left out.

usage: prepare.py <dataset_dir> <out_dir> [review.json ...]
"""
import glob, json, os, sys
CLASSES = ['white_sphere', 'blue_mushrooms', 'green_alien', 'ice_formation', 'green_crystals']
TRAIN, VAL = ('variant_1', 'variant_2', 'variant_3'), ('variant_4',)
data, out = sys.argv[1], sys.argv[2]
bad = set()
for path in sys.argv[3:]:
    bad |= set(json.load(open(path))['problems'])
def frames(split, runs):
    names = sorted(os.path.basename(p)[:-4] for run in runs
                   for p in glob.glob(os.path.join(data, split, 'images', run + '_*.jpg')))
    return [os.path.join(data, split, 'images', n + '.jpg') for n in names if n not in bad]
os.makedirs(out, exist_ok=True)
splits = {'train': frames('train', TRAIN), 'val': frames('train', VAL), 'test': frames('test', ('original',))}
for name, paths in splits.items():
    open(os.path.join(out, f'{name}.txt'), 'w').write('\n'.join(paths) + '\n')
    boxes = [0] * len(CLASSES); negatives = 0
    for p in paths:
        lines = open(p.replace('/images/', '/labels/')[:-4] + '.txt').read().split('\n')
        lines = [l for l in lines if l.strip()]
        negatives += not lines
        for l in lines:
            boxes[int(l.split()[0])] += 1
    print(f'{name}: {len(paths)} images ({negatives} negative), boxes',
          dict(zip(CLASSES, boxes)))
with open(os.path.join(out, 'data.yaml'), 'w') as f:
    f.write(f'path: {out}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n')
    f.writelines(f'  {i}: {c}\n' for i, c in enumerate(CLASSES))
print('left out after review:', len(bad))
