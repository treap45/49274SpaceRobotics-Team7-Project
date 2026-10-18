"""Exploration efficiency of recorded runs: camera coverage over time and distance driven.

usage: coverage_metrics.py <run_dir> [...]

Replays each run's 1 Hz poses through CoverageExplorer.observe on the map
recorded nearest before them, and measures coverage against the run's final
map. The node observes at 5 Hz, so this slightly undercounts while turning;
it is for comparing runs, not for absolute numbers.
"""
import math, os, pickle, re, sys
from types import SimpleNamespace
import numpy as np
PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'src', 'cave_explorer')
sys.path.insert(0, PKG)
from cave_explorer.coverage_explorer import CoverageExplorer
CAMERA_FORWARD_M = 0.5
LEVELS = (0.80, 0.90, 0.95, 0.99)


def as_msg(m):
    h, w = m['grid'].shape
    pos = SimpleNamespace(x=m['origin'][0], y=m['origin'][1])
    info = SimpleNamespace(width=w, height=h, resolution=m['res'], origin=SimpleNamespace(position=pos))
    return SimpleNamespace(info=info, data=m['grid'].ravel(), header=SimpleNamespace(frame_id='map'))


def log_seconds(run):
    """Seconds of the autonomy log from first goal to the coverage phase and to EXHAUSTED."""
    stamps = {}
    for line in open(os.path.join(run, 'autonomy.log'), errors='ignore'):
        m = re.search(r'\[(\d+\.\d+)\]', line)
        if not m:
            continue
        t = float(m.group(1))
        if '[EXPLORING]' in line:
            stamps.setdefault('start', t)
        if 'coverage:' in line:
            stamps.setdefault('coverage', t)
        if 'EXHAUSTED' in line:
            stamps.setdefault('end', t)
    start = stamps.get('start')
    return {k: stamps[k] - start for k in ('coverage', 'end') if k in stamps and start}


def metrics(run):
    rec = pickle.load(open(os.path.join(run, 'record.pkl'), 'rb'))
    maps, poses = rec['maps'], rec['poses']
    final = as_msg(maps[-1])
    cov = CoverageExplorer()
    t0 = poses[0][0]
    curve, dist, mi, prev = [], 0.0, 0, None
    for t, x, y, yaw in poses:
        while mi + 1 < len(maps) and maps[mi + 1]['t'] <= t:
            mi += 1
        if maps[mi]['t'] > t:
            continue
        cov.observe(as_msg(maps[mi]), (x + CAMERA_FORWARD_M * math.cos(yaw), y + CAMERA_FORWARD_M * math.sin(yaw)), yaw)
        if prev:
            dist += math.dist(prev, (x, y))
        prev = (x, y)
        curve.append((t - t0, dist, cov.seen_fraction(final)))
    out = {'duration_s': curve[-1][0], 'distance_m': dist, 'final_seen': curve[-1][2]}
    for level in LEVELS:
        hit = next((c for c in curve if c[2] >= level), None)
        out[f't{int(level * 100)}'] = hit[0] if hit else None
        out[f'd{int(level * 100)}'] = hit[1] if hit else None
    out.update({f'log_{k}_s': v for k, v in log_seconds(run).items()})
    return out


def fmt(v):
    return '   -' if v is None else f'{v:6.0f}'


if __name__ == '__main__':
    cols = ['duration_s', 'distance_m', 'log_coverage_s', 'log_end_s'] + \
        [f't{int(l * 100)}' for l in LEVELS] + [f'd{int(l * 100)}' for l in LEVELS]
    print('%-28s' % 'run' + ''.join('%11s' % c for c in cols) + '  final')
    for run in sys.argv[1:]:
        m = metrics(run)
        print('%-28s' % os.path.basename(run.rstrip('/')) + ''.join('%11s' % fmt(m.get(c)) for c in cols)
              + f'  {m["final_seen"]:.3f}')
