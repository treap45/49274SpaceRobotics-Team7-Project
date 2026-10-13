#!/bin/bash
# usage: collect.sh <out_dir> <run_name> [world.sdf]
WS=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
source /opt/ros/humble/setup.bash; source $WS/install/setup.bash
export ROS_DOMAIN_ID=91 QT_QPA_PLATFORM=offscreen
H=$(dirname "$(readlink -f "$0")"); O=$1; R=$2; W=${3:-$H/mars_cave_labelled.sdf}; L=$O/logs/$R; mkdir -p $L
[ -f $H/mars_cave_labelled.sdf ] || python3 $H/make_world.py $WS/src/cave_explorer/worlds/mars_cave.sdf $H/mars_cave_labelled.sdf
ros2 launch $H/collect_startup.launch.py world:=$W > $L/startup.log 2>&1 &
sleep 25
ros2 launch cave_explorer cave_explorer_navigation.launch.py > $L/nav.log 2>&1 &
sleep 20
python3 $H/collector.py $O $R --ros-args -p use_sim_time:=true > $L/collector.log 2>&1 &
ros2 launch cave_explorer cave_explorer_autonomy.launch.py > $L/autonomy.log 2>&1 &
wait
