"""Build a local review page: a stratified random sample of auto-labelled frames.

Reviewing every risky frame is not feasible (small far-away boxes are in about
a third of all frames), so the page holds a fixed-seed random sample from three
strata, and the review estimates each stratum's error rate:

  A  a labelled box with few visible pixels (MIN_PIXELS .. SMALL_PIXELS)
  B  otherwise, a labelled box with low fill (visible pixels / box area)
  C  everything else, negatives included

Each card shows the frame with its boxes and a 3x crop of every box in A or B.
Open <out>/review.html in a browser (file://), mark only the bad frames, then
press Export: it downloads review.json with the sample, strata and problems.

With --all every listed frame is queued instead (the test set: its labels are
the reference both detectors are scored against, so every frame is checked).

usage: make_review.py <data_dir> <out_dir> <list.txt> [--all]
"""
import glob, html, json, os, random, sys
import cv2
import numpy as np

CLASSES = ['white_sphere', 'blue_mushrooms', 'green_alien', 'ice_formation', 'green_crystals']
COLOURS = [(255, 255, 255), (255, 120, 0), (0, 255, 0), (255, 255, 0), (0, 200, 255)]
MIN_PIXELS, SMALL_PIXELS, LOW_FILL = 150, 600, 0.30
SAMPLE = {'A': 100, 'B': 50, 'C': 50}
STRATUM_TEXT = {'A': 'A: small box', 'B': 'B: low fill', 'C': 'C: other'}

data, out = sys.argv[1], sys.argv[2]
listed = {os.path.basename(l.strip())[:-4] for l in open(sys.argv[3]) if l.strip()}
review_all = '--all' in sys.argv[4:]
metas = sorted(glob.glob(os.path.join(data, '*_meta.jsonl')))
frames = [f for f in (json.loads(l) for p in metas for l in open(p)) if f['name'] in listed]
os.makedirs(os.path.join(out, 'img'), exist_ok=True)


def risky(box):
    c, cx, cy, bw, bh, px, fill = box
    if px < MIN_PIXELS:
        return None
    if px < SMALL_PIXELS:
        return 'A'
    if fill < LOW_FILL:
        return 'B'
    return None


def stratum(frame):
    kinds = {risky(b) for b in frame['boxes']}
    return 'A' if 'A' in kinds else 'B' if 'B' in kinds else 'C'


strata = {k: [f for f in frames if stratum(f) == k] for k in SAMPLE}
rng = random.Random(0)
queue = []
for k, n in SAMPLE.items():
    chosen = strata[k] if review_all else rng.sample(strata[k], min(n, len(strata[k])))
    queue.extend((k, f) for f in chosen)

cards = []
for k, f in queue:
    img = cv2.imread(os.path.join(data, 'images', f['name'] + '.jpg'))
    clean = img.copy()
    crops = []
    for box in f['boxes']:
        c, cx, cy, bw, bh, px, fill = box
        x0, y0, x1, y1 = int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)
        labelled = px >= MIN_PIXELS
        colour = COLOURS[c] if labelled else (128, 128, 128)
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, 2 if labelled else 1)
        if labelled:     # keep the name inside the image for boxes at the edge
            tw = cv2.getTextSize(CLASSES[c], 0, 0.6, 2)[0][0]
            cv2.putText(img, CLASSES[c], (min(max(0, x0), img.shape[1] - tw), max(15, y0 - 4)),
                        0, 0.6, colour, 2)
        if risky(box):
            pad = 10
            crop = clean[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad].copy()
            scale = min(3.0, 160.0 / max(1, crop.shape[0]))
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            off = int(pad * scale) if y0 - pad >= 0 else int(y0 * scale)
            offx = int(pad * scale) if x0 - pad >= 0 else int(x0 * scale)
            cv2.rectangle(crop, (offx, off), (offx + int(bw * scale), off + int(bh * scale)), COLOURS[c], 1)
            cv2.putText(crop, f'{CLASSES[c]} {px}px', (2, 14), 0, 0.45, COLOURS[c], 1)
            crops.append(crop)
    if crops:
        height = max(cr.shape[0] for cr in crops)
        strip = np.hstack([cv2.copyMakeBorder(cr, 0, height - cr.shape[0], 0, 6, cv2.BORDER_CONSTANT)
                           for cr in crops])
        if strip.shape[1] > img.shape[1]:
            strip = cv2.resize(strip, (img.shape[1], int(strip.shape[0] * img.shape[1] / strip.shape[1])))
        strip = cv2.copyMakeBorder(strip, 4, 0, 0, img.shape[1] - strip.shape[1], cv2.BORDER_CONSTANT)
        img = np.vstack([img, strip])
    cv2.imwrite(os.path.join(out, 'img', f['name'] + '.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    n = html.escape(f['name'])
    cards.append(f'<div class="card" data-name="{n}" data-stratum="{k}"><img src="img/{n}.jpg" loading="lazy">'
                 f'<div class="meta"><b>{n}</b> · {STRATUM_TEXT[k]}</div><div class="btns">'
                 + ''.join(f'<button data-v="{v}">{t}</button>' for v, t in
                           [('wrong_box', '框不准'), ('wrong_class', '类别错'), ('missing', '漏框'),
                            ('unrecognisable', '认不出')])
                 + '</div></div>')

counts = ', '.join(f'{STRATUM_TEXT[k]} {sum(1 for q in queue if q[0] == k)}/{len(strata[k])}' for k in SAMPLE)
page = '''<!doctype html><html><head><meta charset="utf-8"><title>Label review</title><style>
body{font-family:sans-serif;margin:16px;background:#111;color:#eee}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:12px}
.card{background:#222;padding:6px;border:3px solid #222}.card.bad{border-color:#e33}
.card img{width:100%;display:block}.meta{font-size:13px;margin:4px 0}
button{margin-right:4px;background:#333;color:#eee;border:1px solid #555;padding:4px 8px;cursor:pointer}
button.on{background:#e33;border-color:#e33}#bar{position:sticky;top:0;background:#111;padding:8px 0;z-index:1}
</style></head><body><div id="bar"><b>QCOUNT frames</b> (sampled: COUNTS) ·
默认全部算"对"，只点有问题的；"认不出"= 框里的东西人也认不出是什么 · 已标记 <span id="n">0</span>
 <button id="export">Export review.json</button></div><div class="grid">CARDS</div><script>
const KEY='label-review-'+location.pathname;let s={};try{s=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(s))}catch(e){}
 document.getElementById('n').textContent=Object.keys(s).filter(k=>s[k].length).length}
document.querySelectorAll('.card').forEach(c=>{const n=c.dataset.name;
 c.querySelectorAll('button').forEach(b=>{const v=b.dataset.v;
  const sync=()=>{const on=(s[n]||[]).includes(v);b.classList.toggle('on',on);c.classList.toggle('bad',(s[n]||[]).length>0)};
  sync();b.onclick=()=>{s[n]=s[n]||[];s[n]=s[n].includes(v)?s[n].filter(x=>x!==v):s[n].concat(v);save();sync()}})});
save();document.getElementById('export').onclick=()=>{
 const cards=[...document.querySelectorAll('.card')];
 const out={reviewed:cards.map(c=>c.dataset.name),strata:Object.fromEntries(cards.map(c=>[c.dataset.name,c.dataset.stratum])),
  problems:Object.fromEntries(Object.entries(s).filter(([k,v])=>v.length&&cards.some(c=>c.dataset.name===k)))};
 const blob=new Blob([JSON.stringify(out,null,1)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='review.json';a.click()};
</script></body></html>'''
open(os.path.join(out, 'review.html'), 'w').write(
    page.replace('QCOUNT', str(len(queue))).replace('COUNTS', counts).replace('CARDS', '\n'.join(cards)))
print(f'{len(queue)} frames sampled ({counts}) -> {os.path.join(out, "review.html")}')
