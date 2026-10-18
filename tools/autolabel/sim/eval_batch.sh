#!/bin/bash
# usage: [BGR=true|false] [FRAMES=1] [EXPLORATION=two_phase|interleaved|information_gain|info_tour] eval_batch.sh <model.pt> <out_dir> <name> <runs>
# Runs the full autonomy <runs> times, each until EXHAUSTED or 18 min, and scores it.
H=$(dirname "$(readlink -f "$0")"); MODEL=$1; O=$2; N=$3; K=$4
for i in $(seq 1 $K); do
  D=$O/${N}_run$i
  MODEL=$MODEL BGR=${BGR:-true} FRAMES=${FRAMES:-} EXPLORATION=${EXPLORATION:-info_tour} setsid bash $H/run.sh $D < /dev/null > /dev/null 2>&1 &
  start=$(date +%s)
  while [ $(( $(date +%s) - start )) -lt 1080 ]; do
    grep -qE "EXHAUSTED|Traceback" $D/autonomy.log 2>/dev/null && break; sleep 10
  done
  sleep 5
  echo "$N run$i: $(grep -oE 'EXHAUSTED.{0,60}' $D/autonomy.log | head -1) after $(( $(date +%s) - start )) s; $(grep -o 'Detector loaded: .*' $D/autonomy.log)"
  python3 $H/score.py $D/markers.json | tail -20
  bash $H/cleanup.sh > /dev/null 2>&1; sleep 5
done
echo EVAL DONE
