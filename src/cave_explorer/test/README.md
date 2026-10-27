# Module tests

Unit tests on controlled inputs (synthetic maps, detections, depth images and
transforms); no simulator needed. Run from the repository root:

```bash
python3 -m pytest src/cave_explorer/test -q
```

| File | Module |
|---|---|
| `test_artifact_tracker.py` | depth localisation, association, fusion, track merging |
| `test_artifact_obstacles.py` | artefact obstacle points for Nav2 |
| `test_inspection_planner.py` | viewpoints, clearance, line of sight, inspection success |
| `test_behaviour_manager.py` | explore / inspect state machine, retries, revisit, stuck recovery |
| `test_navigation_client.py` | Nav2 goals: rejection, cancellation, stale callbacks, false success |
| `test_frontier_explorer.py`, `test_coverage_explorer.py`, `test_nbv_explorer.py` | exploration modes |
| `test_cave_geometry.py` | widest area and bottlenecks on synthetic corridors (Advanced 2) |
| `test_online_roadmap.py` | roadmap construction, validation and planning (Advanced 4) |

End-to-end behaviour is checked with full simulation runs (`tools/autolabel/sim/`).
`test_navigation_client.py` occasionally segfaults inside rclpy's multi-threaded
executor (about 1 run in 17); rerun if it does.
