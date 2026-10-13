# Simulator-labelled detector dataset

Tools that built, checked and evaluated `src/cave_explorer/models/artifact_yolov8n_autolabel.pt`.
Images, labels and training runs live in `datasets/autolabel/` (git-ignored).

Labels come from Gazebo, not from hand-drawn boxes: during collection only, the
robot carries a bounding-box camera and a semantic segmentation camera with the
same pose, field of view, resolution and rate as its RGB camera, so their stamps
match the RGB frames exactly. The robot and code used in the demo are unchanged.
Ground-truth poses from the world file are used to label and to score, never by
the runtime code.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| Label plugins | `collect/make_world.py` | Copy of `mars_cave.sdf` with a Label plugin per artefact (label = class index + 1) |
| Random layouts | `collect/make_variant.py` | Moves every artefact to a random free place of a SLAM map the robot built: 3 m clearance, 8 m apart, 5 m from the start, random yaw; same rules for every class. `collect/worlds/` holds the four layouts used |
| Collect | `collect/batch.sh` → `collect.sh`, `collector.py` | Full autonomy in each layout; keeps a frame per 0.5 m / 15 deg moved (negatives per 2 m / 45 deg); a box is labelled only if at least 150 of its pixels show the object |
| Thin | `collect/thin.py` | Lists frames at a coarser pose step, dropping near-duplicates (no image deleted) |
| Review | `collect/make_review.py` | Local `review.html`: stratified sample (small boxes / low fill / other) or every frame (`--all`); mark bad frames, export `review.json` |
| Splits | `train/finalise_splits.py` | Final lists minus reviewed-bad frames and frames with a box mostly inside a larger box of its class (semantic segmentation cannot tell their pixels apart) |
| Evaluate | `train/evaluate.py` | Per-class precision / recall and cross-class confusions at the node's confidence |
| Simulation score | `sim/eval_batch.sh` → `run.sh`, `score.py` | Full autonomy runs scored against artefact mesh centres (`GT_REF=origin` for model origins: the ice mesh is 2.75 m from its origin) |
| Localisation replay | `sim/frame_recorder.py`, `sim/replay.py` | Record every frame with depth and transform (`FRAMES=1`), replay detector and tracker variants offline |

`sim/cleanup.sh` stops everything these scripts start. Runs use `ROS_DOMAIN_ID=91`
and a headless Gazebo (`sim/startup_headless.launch.py`: the course startup without RViz).

## Data used (2026-10-13)

| Split | Source | Images | Negatives | Boxes (sphere / mushrooms / alien / ice / crystals) |
|---|---|---|---|---|
| train | layouts 1-3 | 1207 | 111 | 349 / 456 / 282 / 456 / 431 |
| val | layout 4 | 331 | 37 | 114 / 101 / 68 / 138 / 79 |
| test | original layout, never trained on | 311 | 46 | 64 / 87 / 51 / 107 / 77 |

Review: 200 sampled training frames, 1 error (a far green_alien hidden behind a
near one, which led to the same-class overlap rule); all 315 test frames, 4
unrecognisable slivers removed.

## Commands

```bash
tools/autolabel/collect/batch.sh                       # collect 4 layouts + test, ~80 min
BGR=true tools/autolabel/sim/eval_batch.sh src/cave_explorer/models/artifact_yolov8n_autolabel.pt datasets/autolabel/sim_eval new 2
BGR=false tools/autolabel/sim/eval_batch.sh src/cave_explorer/models/artifact_yolov8n.pt datasets/autolabel/sim_eval old 2
```

`BGR` must match the model: the autolabel model takes BGR, the earlier
`artifact_yolov8n.pt` was trained on channel-swapped images and takes RGB.
