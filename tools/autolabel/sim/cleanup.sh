#!/bin/bash
# Stop every process started by the test simulations; leave foxglove_bridge alone.
for p in $(pgrep -f "autolabel/(sim/run|collect/collect).sh"); do
  g=$(ps -o pgid= -p $p | tr -d ' '); [ -n "$g" ] && kill -TERM -- -$g 2>/dev/null
done
sleep 3
pat='/opt/ros/humble/(lib|bin)/|ign gazebo|ruby /usr/bin/ign|install/cave_explorer/lib|autolabel/sim/(recorder|markers|frame_recorder).py|autolabel/collect/collector.py|startup_headless|collect_startup'
for sig in TERM KILL; do
  pids=$(ps -eo pid=,cmd= | grep -E "$pat" | grep -v -E "foxglove|grep" | awk '{print $1}')
  [ -n "$pids" ] && kill -$sig $pids 2>/dev/null; sleep 3
done
ps -eo pid=,cmd= | grep -E "$pat" | grep -v -E "foxglove|grep" | cut -c1-100
