# 49274SpaceRobotics-Team7-Project

ROS 2 Humble workspace for the 49274 Space Robotics team project (Team 7). A
simulated rover in Gazebo Fortress explores an unknown Martian cave on its own,
detects and localises five kinds of artefact with its camera, and drives up to
inspect the chosen ones, while mapping with SLAM Toolbox and driving with Nav2.

```text
explore → detect artefacts in the camera image → localise them in the map
        → drive to a safe close-range viewpoint and inspect → record as visited → resume exploring
```

Demonstration video: [Google Drive](https://drive.google.com/file/d/1z75lHha7Im_7oUyTluVJm6KNTtczVPQP/view?usp=sharing)

## Results

In the course cave (`mars_cave.sdf`), every run of the final system:

- found **15 / 15** artefacts (all five classes), mean localisation error **0.53 m**;
- inspected **6 / 6** chosen artefacts (all `green_crystals` and `white_sphere`) on the first attempt, with no repeated inspections and no time stuck;
- finished exploring in **398 s / 257 m** on average, against 598 s / 422 m for our first frontier-then-coverage method (three runs each).

The detector, a YOLOv8n retrained on simulator-labelled images, scores mAP50 **0.99** on a held-out test layout (mean precision 0.94, recall 0.95 at confidence 0.5).

## Tasks

| Task | What we built |
|---|---|
| Perception 1: dataset | 1849 frames labelled by Gazebo's bounding-box and segmentation cameras, with human review; random artefact layouts for training, the original layout kept as the test set |
| Perception 2: detection | YOLOv8n (COCO pre-trained); boxes drawn on `/detections_image` |
| Perception 3: localisation | Foreground depth inside each box, back-projected and transformed to `map`; same-class tracks associated by a gate from the measured width, fused by median, converged duplicates merged |
| Planning 1: exploration | `info_tour`: lidar frontiers and unseen camera regions clustered, a viewpoint per cluster by ray casting, visited in travelling-salesman order (Held-Karp / 2-opt) |
| Planning 2: inspection | Viewpoints at 16 bearings around the artefact, outside a keep-out disk sized from the measured width, with clearance and line of sight; success only when the artefact is held in view at the right range and heading |
| Planning 3: behaviour switching | State machine with one retry from a new side, deferral, a post-exploration revisit, a stuck watchdog and a visited record |
| Advanced 2: cave geometry | Distance transform of the live map: widest reachable area, and bottlenecks from ridge minima confirmed by a disk-cut connectivity test |
| Advanced 4: online roadmap | A trail roadmap (where the robot has driven) and an observed roadmap (adaptive samples on the medial axis of seen free space), both re-validated on every map; A* path to any point clicked in RViz |

## Requirements

- Ubuntu 22.04, ROS 2 Humble desktop, Gazebo Fortress.
- ROS packages:

```bash
sudo apt update
sudo apt install \
  git python3-colcon-common-extensions python3-setuptools python3-opencv python3-scipy \
  ros-humble-ros-ign-bridge ros-humble-ros-ign-gazebo ros-humble-ros-gz-bridge \
  ros-humble-robot-localization ros-humble-slam-toolbox \
  ros-humble-navigation2 ros-humble-nav2-bringup \
  ros-humble-xacro ros-humble-robot-state-publisher ros-humble-rviz2 \
  ros-humble-rclpy ros-humble-cv-bridge ros-humble-tf2-ros \
  ros-humble-geometry-msgs ros-humble-nav-msgs ros-humble-nav2-msgs \
  ros-humble-sensor-msgs ros-humble-visualization-msgs ros-humble-vision-msgs \
  ros-humble-ament-index-python ros-humble-launch-ros ros-humble-teleop-twist-keyboard
```

- Python packages for the detector. Keep NumPy below 2 so `cv_bridge` still works:

```bash
pip install --user ultralytics "numpy<2"
```

## Build

```bash
git clone https://github.com/treap45/49274SpaceRobotics-Team7-Project.git
cd 49274SpaceRobotics-Team7-Project
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select cave_explorer
source install/setup.bash
```

The simulation assets and the detector weights (`src/cave_explorer/models/`) are in the repository.

## Run

Open three terminals at the repository root and source the workspace in each:

```bash
source /opt/ros/humble/setup.bash && source install/setup.bash
```

1. Simulation and RViz. Wait for Gazebo and RViz to open:

   ```bash
   ros2 launch cave_explorer cave_explorer_startup.launch.py
   ```

2. SLAM and Nav2. Wait for `Managed nodes are active`:

   ```bash
   ros2 launch cave_explorer cave_explorer_navigation.launch.py
   ```

3. Autonomy, cave geometry (Advanced 2) and online roadmap (Advanced 4). The rover starts moving after about 20 s and stops with `EXHAUSTED` when nothing is left to see:

   ```bash
   ros2 launch cave_explorer cave_explorer_autonomy.launch.py
   ```

Launch arguments of the autonomy launch:

| Argument | Default | Meaning |
|---|---|---|
| `exploration_mode` | `info_tour` | `two_phase`, `interleaved`, `information_gain` or `info_tour` (the four versions compared in the report) |
| `detector_model` | `models/artifact_yolov8n_autolabel.pt` | YOLO weights |
| `detector_input_bgr` | `true` | `false` for the earlier `models/artifact_yolov8n.pt` |
| `roadmap_record_path` | empty | Pickle file to record the roadmap for `roadmap_replay.py` |

In RViz you can see the detection boxes, artefact markers (unvisited / visited), frontiers with the planned tour (magenta), camera coverage, the cave geometry features and both roadmaps. Use **Publish Point** on mapped space to ask the roadmaps for a path.

To pause the whole run, use Gazebo's pause button (not the Nav2 panel's Pause, which stops only Nav2). Stop the launches with `Ctrl+C` in the order 3, 2, 1.

## Tests

```bash
python3 -m pytest src/cave_explorer/test -q
```

246 unit tests on controlled inputs, covering every module (tracker, inspection planner, behaviour manager, explorers, navigation client, geometry, roadmap). `test_navigation_client.py` occasionally segfaults inside rclpy's multi-threaded executor (about 1 run in 17); rerun if it does.

## Repository layout

```text
src/cave_explorer/
├── cave_explorer/            # node and modules
│   ├── cave_explorer.py      #   main node: wires the modules together
│   ├── interfaces.py         #   shared dataclasses and state enums
│   ├── perception.py         #   YOLO detector and box annotation
│   ├── artifact_tracker.py   #   depth localisation and track fusion
│   ├── artifact_obstacles.py #   artefacts as Nav2 obstacles (low parts the lidar misses)
│   ├── inspection_planner.py #   close-range viewpoints and inspection success
│   ├── behaviour_manager.py  #   explore / inspect state machine
│   ├── navigation_client.py  #   the only Nav2 action client
│   ├── frontier_explorer.py, coverage_explorer.py, nbv_explorer.py   # exploration modes
│   ├── cave_geometry.py, cave_geometry_node.py                       # Advanced 2
│   ├── online_roadmap.py, online_roadmap_node.py, roadmap_replay.py  # Advanced 4
│   └── dataset_collector.py  #   early hand-collected dataset
├── launch/ config/ urdf/ worlds/ models/
└── test/                     # unit tests
src/artifact_dataset/         # images and YOLO labels that trained and tested the detector
tools/autolabel/              # dataset collection, auto-labelling, training, evaluation, simulation scoring
Workplan.md                   # team split, module interfaces and acceptance criteria
```

See [tools/autolabel/README.md](tools/autolabel/README.md) for how the dataset was built and how runs are scored, and [src/artifact_dataset/README.md](src/artifact_dataset/README.md) for the dataset itself.

## Troubleshooting

- **Robot never moves, `Waiting for navigate_to_pose`:** Nav2 did not come up. Stop everything, make sure no old `lifecycle_manager` or `ekf_node` is still running, and start again.
- **No detection boxes:** a wrong `detector_model` path turns the detector off; leave the argument empty to use the default.
- **Package not found:** build the workspace and source `install/setup.bash` in that terminal.

## Team

| Member | Main work |
|---|---|
| Nuochen Li | Perception: dataset, detector |
| Zhibo Wang | Localisation, inspection planning, Advanced 2 |
| Tianyang Shen | Team lead; exploration, navigation, behaviour, integration, testing, Advanced 4 |

Built on the course starter code and simulation resources; their attribution is retained.
