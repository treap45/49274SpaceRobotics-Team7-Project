# 49274SpaceRobotics-Team7-Project

ROS 2 workspace for the 49274 Space Robotics team project. A simulated rover uses Gazebo Fortress, SLAM Toolbox and Nav2 to navigate a cave.

## Current status

The repository currently contains the course starter implementation:

- Cave world, rover model and sensor bridges.
- SLAM and Nav2 launch configurations.
- Camera-based stop-sign detection and RViz overlays.
- Navigation to a fixed goal, return to the origin, then random predefined goals.

Artefact localisation, frontier exploration and autonomous inspection are still under development. The current artefact markers indicate the **robot's position**, not estimated object positions.

## Requirements

- Ubuntu 22.04.
- ROS 2 Humble desktop installed and configured.
- Gazebo Fortress installed.
- GitHub access to this private repository.

With the ROS 2 apt repository already configured, install the build tools and runtime dependencies:

```bash
sudo apt update
sudo apt install \
  git python3-colcon-common-extensions python3-setuptools python3-opencv \
  ros-humble-ros-ign-bridge ros-humble-ros-ign-gazebo ros-humble-ros-gz-bridge \
  ros-humble-robot-localization ros-humble-slam-toolbox \
  ros-humble-navigation2 ros-humble-nav2-bringup \
  ros-humble-xacro ros-humble-robot-state-publisher ros-humble-rviz2 \
  ros-humble-rclpy ros-humble-cv-bridge ros-humble-tf2-ros \
  ros-humble-geometry-msgs ros-humble-nav-msgs ros-humble-nav2-msgs \
  ros-humble-sensor-msgs ros-humble-visualization-msgs \
  ros-humble-ament-index-python ros-humble-launch-ros
```

## Clone and build

This repository is a complete workspace. Clone it into a new directory:

```bash
git clone https://github.com/treap45/49274SpaceRobotics-Team7-Project.git
cd 49274SpaceRobotics-Team7-Project
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select cave_explorer
source install/setup.bash
```

The supplied simulation assets are included. No separate model download or previous assignment packages are required.

## Run

Open three terminals at the cloned repository root. In **each terminal**, run:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### Terminal 1 — Simulation and RViz

```bash
ros2 launch cave_explorer cave_explorer_startup.launch.py
```

Wait for Gazebo to load the cave and rover.

### Terminal 2 — Mapping and navigation

```bash
ros2 launch cave_explorer cave_explorer_navigation.launch.py
```

Wait for SLAM and Nav2 to become ready. The **2D Goal Pose** tool in RViz can be used to test navigation before launching autonomy.

### Terminal 3 — Starter autonomy

```bash
ros2 launch cave_explorer cave_explorer_autonomy.launch.py
```

The rover starts the template goal sequence. The annotated camera stream is published on `/detections_image`.

Stop each launch with `Ctrl+C`. Mapping and autonomy can be restarted without restarting Gazebo.

## Project structure

```text
src/cave_explorer/
├── cave_explorer/    # Python ROS node
├── launch/          # Simulation, navigation and autonomy entry points
├── config/          # Nav2, SLAM, sensor bridge, RViz and detector configuration
├── urdf/            # Rover description and meshes
├── worlds/          # Cave world, artefact meshes and textures
├── package.xml
└── setup.py
```

Main implementation: [cave_explorer.py](src/cave_explorer/cave_explorer/cave_explorer.py).

## Troubleshooting

- **Package not found:** build the workspace and source `install/setup.bash` in the current terminal.
- **Waiting for navigate_to_pose:** start the navigation launch and check that Nav2 is ready.
- **Waiting for transform:** check that the simulator is running and SLAM is publishing the map transform.
- **Missing simulation assets:** build the package with its `worlds/` and `urdf/` directories intact.

## Development

Create feature branches from `main` and merge through reviewed pull requests. Verify build and launch behaviour before merging.

Build outputs, internal reports, meeting notes, collected datasets, trained weights and recordings are excluded from version control.

The project includes course-provided starter code and simulation resources. Original attribution and supplied resource metadata are retained.
