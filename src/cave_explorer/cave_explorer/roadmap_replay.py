"""Replay a recorded run through the roadmap builders and measure them offline.

Records come from ``online_roadmap_node`` (``record_path``): every map version
plus the robot pose at each update, so both strategies can be rebuilt and
evaluated without another simulation::

    python3 -m cave_explorer.roadmap_replay run.pkl --plot roadmap.png --timeline timeline.png

Measures, on the known traversable space connected to the robot:

* attachable: share of traversable cells with a valid straight segment to
  some node within the attach radius (can the roadmap be entered there?);
* success: share of random start/goal pairs the roadmap connects;
* stretch: roadmap path length over the shortest grid path (16-connected,
  so the grid path itself is within about 3 % of the true shortest path).
"""

import argparse
import math
import pickle

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from cave_explorer.online_roadmap import (MapSnapshot, RoadmapBuilder, attach, path_length,
                                          plan_path, shortcut)

MOVES = [(0, 1), (1, 0), (1, 1), (1, -1), (1, 2), (2, 1), (2, -1), (1, -2)]


def load(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def reachable_cells(snapshot, robot_xy):
    """Traversable cells connected to the robot (or to the largest region)."""
    labels, count = ndimage.label(snapshot.traversable)
    if not count:
        return np.zeros(snapshot.shape, dtype=bool)
    row, col = snapshot.cells(np.asarray(robot_xy, dtype=float))
    if 0 <= row < snapshot.shape[0] and 0 <= col < snapshot.shape[1] and labels[row, col]:
        return labels == labels[row, col]
    return labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1


def grid_graph(snapshot, mask):
    """Sparse 16-connected graph over ``mask`` cells; knight moves need both cells they cut."""
    index = -np.ones(snapshot.shape, dtype=int)
    cells = np.argwhere(mask)
    index[mask] = np.arange(len(cells))
    rows, cols, weights = [], [], []
    height, width = snapshot.shape
    for dr, dc in MOVES:
        r0, c0 = cells[:, 0], cells[:, 1]
        r1, c1 = r0 + dr, c0 + dc
        ok = (r1 >= 0) & (r1 < height) & (c1 >= 0) & (c1 < width)
        ok[ok] &= mask[r1[ok], c1[ok]]
        # Cells a diagonal or knight move cuts through must be traversable too.
        sr, sc = int(np.sign(dr)), int(np.sign(dc))
        if abs(dr) == 1 and abs(dc) == 1:
            between = [(0, dc), (dr, 0)]
        elif abs(dc) == 2:
            between = [(0, sc), (dr, sc)]
        elif abs(dr) == 2:
            between = [(sr, 0), (sr, dc)]
        else:
            between = []
        for br, bc in between:
            rr, cc = r0 + br, c0 + bc
            inside = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            ok &= inside
            ok[ok] &= mask[rr[ok], cc[ok]]
        rows.append(index[r0[ok], c0[ok]])
        cols.append(index[r1[ok], c1[ok]])
        weights.append(np.full(ok.sum(), math.hypot(dr, dc) * snapshot.resolution))
    n = len(cells)
    graph = coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n, n)).tocsr()
    return graph, index, cells


def evaluate(builder, snapshot, robot_xy, pairs=150, attach_samples=600, seed=0, radius_m=None):
    """Attachable share, path success and stretch for one builder on one snapshot."""
    roadmap = builder.roadmap
    radius = builder.attach_radius_m if radius_m is None else radius_m
    mask = reachable_cells(snapshot, robot_xy)
    cells = np.argwhere(mask)
    rng = np.random.default_rng(seed)
    result = {'nodes': len(roadmap.nodes), 'edges': len(roadmap.edge_list())}
    labels = roadmap.components()
    result['components'] = len(set(labels.values()))
    if not len(cells) or not roadmap.nodes:
        result.update(attachable=0.0, success=0.0, stretch=float('nan'))
        return result
    samples = snapshot.centres(*cells[rng.integers(len(cells), size=attach_samples)].T)
    result['attachable'] = float(np.mean([bool(attach(roadmap, snapshot, p, radius, limit=1))
                                          for p in samples]))
    graph, index, _ = grid_graph(snapshot, mask)
    starts = cells[rng.integers(len(cells), size=pairs)]
    goals = cells[rng.integers(len(cells), size=pairs)]
    distances = dijkstra(graph, directed=False, indices=index[starts[:, 0], starts[:, 1]])
    successes, stretches, smoothed = 0, [], []
    for k in range(pairs):
        grid_length = distances[k, index[goals[k, 0], goals[k, 1]]]
        if not np.isfinite(grid_length) or grid_length < 3.0:
            continue
        start = snapshot.centres(*starts[k])
        goal = snapshot.centres(*goals[k])
        path = plan_path(roadmap, snapshot, start, goal, radius, relaxed_start=False)
        if path is not None:
            successes += 1
            stretches.append(path_length(path) / grid_length)
            smoothed.append(path_length(shortcut(path, snapshot)) / grid_length)
        else:
            stretches.append(np.nan)
    tested = len(stretches)
    result['success'] = successes / tested if tested else 0.0
    result['stretch'] = float(np.nanmedian(stretches)) if successes else float('nan')
    result['smoothed_stretch'] = float(np.median(smoothed)) if successes else float('nan')
    result['tested_pairs'] = tested
    return result


def replay(record, builders, eval_every_s=None, **evaluate_args):
    """Run builders over a record; return the final snapshot and a timeline."""
    snapshot, version, timeline = None, None, []
    start = record['ticks'][0][0]
    next_eval = start + eval_every_s if eval_every_s else math.inf
    robot_xy = None
    for tick in record['ticks']:
        stamp, map_version, robot_xy = tick[0], tick[1], tick[2]
        if map_version != version:
            grid, resolution, origin, yaw = record['maps'][map_version]
            snapshot, version = MapSnapshot(grid, resolution, origin, yaw), map_version
        for builder in builders:
            builder.update(snapshot, robot_xy, stamp)
        if stamp >= next_eval:
            next_eval += eval_every_s
            row = {'t': stamp - start}
            for builder in builders:
                row[builder.mode] = evaluate(builder, snapshot, robot_xy, **evaluate_args)
            timeline.append(row)
    return snapshot, robot_xy, timeline


def plot_roadmaps(snapshot, builders, trajectory, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    corners = snapshot.centres(np.array([0, snapshot.shape[0] - 1]), np.array([0, snapshot.shape[1] - 1]))
    extent = (corners[0, 0], corners[1, 0], corners[0, 1], corners[1, 1])
    image = np.where(snapshot.free, 1.0, 0.25)
    image = np.where(snapshot.free & ~snapshot.traversable, 0.85, image)
    figure, axes = plt.subplots(1, len(builders), figsize=(7.5 * len(builders), 6.5))
    for axis, builder in zip(np.atleast_1d(axes), builders):
        roadmap = builder.roadmap
        axis.imshow(image, cmap='gray', vmin=0, vmax=1, origin='lower', extent=extent)
        axis.plot(trajectory[:, 0], trajectory[:, 1], color='tab:red', linewidth=0.7, alpha=0.6,
                  label='robot path')
        for a, b in roadmap.edge_list():
            (x0, y0), (x1, y1) = roadmap.xy(a), roadmap.xy(b)
            axis.plot([x0, x1], [y0, y1], color='tab:green', linewidth=0.5, alpha=0.7)
        for kind, colour in (('trail', 'tab:orange'), ('observed', 'tab:blue')):
            xy = np.array([(n.x, n.y) for n in roadmap.nodes.values() if n.kind == kind]).reshape(-1, 2)
            axis.scatter(xy[:, 0], xy[:, 1], s=10, color=colour, zorder=3, label=f'{kind} nodes')
        axis.set_title(f'{builder.mode}: {len(roadmap.nodes)} nodes, {len(roadmap.edge_list())} edges')
        axis.set_aspect('equal')
        axis.set_xlabel('x (m)')
        axis.set_ylabel('y (m)')
        axis.legend(loc='lower right', fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def plot_timeline(timeline, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    modes = [key for key in timeline[0] if key != 't']
    t = [row['t'] / 60 for row in timeline]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for mode in modes:
        axes[0].plot(t, [row[mode]['attachable'] for row in timeline], label=mode)
        axes[1].plot(t, [row[mode]['success'] for row in timeline], label=mode)
        axes[2].plot(t, [row[mode]['nodes'] for row in timeline], label=mode)
    for axis, title in zip(axes, ('Attachable share of known space', 'Path success rate',
                                  'Roadmap nodes')):
        axis.set_title(title)
        axis.set_xlabel('time (min)')
        axis.legend()
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.05)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('record')
    parser.add_argument('--plot', help='PNG of the final roadmaps')
    parser.add_argument('--timeline', help='PNG of measures over time')
    parser.add_argument('--every', type=float, default=60.0, help='timeline interval (s)')
    args = parser.parse_args()
    record = load(args.record)
    builders = [RoadmapBuilder('trail'), RoadmapBuilder('observed')]
    snapshot, robot_xy, timeline = replay(record, builders, args.every if args.timeline else None)
    for builder in builders:
        totals = builder.totals
        print(builder.mode, evaluate(builder, snapshot, robot_xy),
              f'mean update {1000 * totals.seconds / totals.updates:.2f} ms, '
              f'max {1000 * totals.max_seconds:.1f} ms, removed nodes {totals.removed_nodes}, '
              f'removed edges {totals.removed_edges}')
    if args.plot:
        trajectory = np.array([tick[2] for tick in record['ticks']])
        plot_roadmaps(snapshot, builders, trajectory, args.plot)
    if args.timeline and timeline:
        plot_timeline(timeline, args.timeline)


if __name__ == '__main__':
    main()
