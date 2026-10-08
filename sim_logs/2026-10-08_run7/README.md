# Simulation run 7 — 2026-10-08

Output of `cave_explorer_autonomy.launch.py` (the `cave_explorer_node` log) for one
full run in `mars_cave.sdf`, started with the three launch files as in the task sheet.

## Code

`main` at `ca95b2e`, plus a stuck watchdog in `behaviour_manager.py` that was not yet
committed (it gives up a goal after 20 s with less than 0.5 m of progress; see the
"no progress" lines). Detector: `models/artifact_yolov8n.pt`, RGB input, confidence
0.5. Inspected classes: `green_crystals`, `white_sphere`.

## Result

- Exploration ended after 1081 s (18 min) with the camera having seen 99 % of the
  free space ("camera has seen 99%").
- Inspections: 5 succeeded (3 `white_sphere`, 2 `green_crystals`), 1 deferred.
- 13 s to 222 s: the robot was physically stuck at about (6.95, 3.07) next to the
  `green_crystals` near the start, which the lidar cannot see. The watchdog gave
  up 10 goals in a row; changing goal did not free it.

## What is not in the log

Only state-machine decisions and localisation failures are logged. Per-frame
detections (class, confidence, box) and images were not recorded in this run.
