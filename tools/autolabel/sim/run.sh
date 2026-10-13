#!/bin/bash
# usage: [MODEL=weights.pt] [BGR=true|false] [FRAMES=1] run.sh <outdir>
WS=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
source /opt/ros/humble/setup.bash
source $WS/install/setup.bash
export ROS_DOMAIN_ID=91 QT_QPA_PLATFORM=offscreen
S=$(dirname "$0"); D=$1; mkdir -p $D
ros2 launch $S/startup_headless.launch.py > $D/startup.log 2>&1 &
sleep 25
ros2 launch cave_explorer cave_explorer_navigation.launch.py > $D/nav.log 2>&1 &
sleep 20
python3 $S/recorder.py $D/record.pkl --ros-args -p use_sim_time:=true > $D/recorder.log 2>&1 &
python3 $S/markers.py $D/markers.json --ros-args -p use_sim_time:=true > $D/markers.log 2>&1 &
[ -n "$FRAMES" ] && python3 $S/frame_recorder.py $D --ros-args -p use_sim_time:=true > $D/frames.log 2>&1 &
ros2 launch cave_explorer cave_explorer_autonomy.launch.py ${MODEL:+detector_model:=$MODEL} detector_input_bgr:=${BGR:-true} > $D/autonomy.log 2>&1 &
wait
