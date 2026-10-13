#!/bin/bash
# Collect: 4 random layouts -> train, the original layout -> test. One run at a time.
# usage: batch.sh [data_dir]   (default <workspace>/datasets/autolabel)
H=$(dirname "$(readlink -f "$0")"); CLEAN=$H/../sim/cleanup.sh
WS=$(cd $H/../../.. && pwd); DATA=${1:-$WS/datasets/autolabel}
[ -f $H/mars_cave_labelled.sdf ] || python3 $H/make_world.py $WS/src/cave_explorer/worlds/mars_cave.sdf $H/mars_cave_labelled.sdf
run() {   # <out_dir> <run_name> <world>
  setsid bash $H/collect.sh $1 $2 $3 < /dev/null > /dev/null 2>&1 &
  local log=$1/logs/$2/autonomy.log start=$(date +%s)
  while [ $(( $(date +%s) - start )) -lt 1080 ]; do
    grep -qE "EXHAUSTED|Traceback" $log 2>/dev/null && break; sleep 10
  done
  echo "$2: $(grep -oE 'EXHAUSTED.*' $log | head -1) after $(( $(date +%s) - start )) s; $(tail -1 $1/logs/$2/collector.log | grep -oE 'kept.*')"
  bash $CLEAN > /dev/null 2>&1; sleep 5
}
for s in 1 2 3 4; do run $DATA/train variant_$s $H/worlds/variant_$s.sdf; done
run $DATA/test original $H/mars_cave_labelled.sdf
echo BATCH DONE
