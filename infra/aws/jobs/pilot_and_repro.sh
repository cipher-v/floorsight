#!/bin/bash
# Pilot on COOL with the convex-fill workaround, saving every non-convex mask polygon; then replay each saved polygon
# with cv2.fillPoly in a fresh process on COOL and on stock (one process per polygon, so a crash is attributed exactly).
set -uo pipefail
J=/opt/floorsight/jobs/$JOB
if [ -z "${DETACHED:-}" ]; then
  [ -e "$J" ] && { echo "job dir $J exists"; exit 2; }
  mkdir -p "$J"; cp "$0" "$J/run.sh"; chown -R floorsight: "$J"
  systemd-run --unit="fs-$JOB" --uid=floorsight --setenv=DETACHED=1 --setenv=JOB="$JOB" --setenv=BUCKET="$BUCKET" --property=WorkingDirectory=/opt/floorsight/current bash "$J/run.sh"
  echo "started fs-$JOB"; exit 0
fi
if [ -e "$J/STARTED" ]; then exit 0; fi; date -u +%FT%TZ > "$J/STARTED"
exec >> "$J/log.txt" 2>&1
A=/opt/floorsight/current
mkdir -p "$J/polys"
aws s3 cp --region us-east-1 "s3://$BUCKET/inputs/floor-calibration-camA.json" "$J/cal.json" --only-show-errors
T0=$(date +%s)
FS_DEBUG_FILLPOLY=$J/polys/p PYTHONFAULTHANDLER=1 /opt/floorsight/perc -P "$A/scripts/real_clip_perception.py" --clip /opt/floorsight/data/private/clips/screen-002.mp4 \
  --name screen-002 --output "$J/screen-002" --models /opt/floorsight/models --calibration "$J/cal.json" --person-model rtdetrv2 --segmenter slimsam77 --vocab gdino \
  --hf-index /opt/floorsight/jobs/setup-01/hf-models.json --vehicle-source coco --vocab-every 5 --person-conf 0.2 --vehicle-conf 0.3 --track-person-conf 0.3 > "$J/screen-002.log" 2>&1
echo "perception exit=$? wall_s=$(( $(date +%s) - T0 )) saved_nonconvex=$(ls $J/polys | wc -l)"
for w in /opt/floorsight/perc-stock /opt/floorsight/perc; do
  ok=0; bad=0; badlist=""
  for f in $J/polys/*.npz; do
    $w -c "
import sys, numpy as np, cv2
d = np.load('$f'); m = np.zeros(tuple(int(v) for v in d['shape']), np.uint8)
cv2.fillPoly(m, [d['pts']], 1)
" >/dev/null 2>&1 && ok=$((ok+1)) || { bad=$((bad+1)); badlist="$badlist $(basename $f)"; }
  done
  echo "replay $w: ok=$ok failed=$bad first:${badlist:0:200}"
done
B=$(ls $J/polys | head -1); [ -n "$B" ] && /opt/floorsight/perc -c "import numpy as np; d=np.load('$J/polys/$B'); print('example', d['shape'].tolist(), d['pts'].tolist())"
for f in $(cd $J/polys && ls | head -400); do :; done
aws s3 cp --region us-east-1 --recursive "$J" "s3://$BUCKET/jobs/$JOB/" --exclude "*.mp4" --only-show-errors
echo DONE > "$J/DONE"
