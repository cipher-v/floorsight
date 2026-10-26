#!/bin/bash
# Debug job: reproduce the COOL fillPoly crash in danger_zone.step, then replay the last polygon on both OpenCV builds.
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
aws s3 cp --region us-east-1 "s3://$BUCKET/inputs/floor-calibration-camA.json" "$J/cal.json" --only-show-errors
FS_DEBUG_FILLPOLY=$J/last.npz PYTHONFAULTHANDLER=1 /opt/floorsight/perc -P "$A/scripts/real_clip_perception.py" --clip /opt/floorsight/data/private/clips/screen-002.mp4 \
  --name screen-002 --output "$J/run" --models /opt/floorsight/models --calibration "$J/cal.json" --person-model rtdetrv2 --segmenter slimsam77 --vocab gdino \
  --hf-index /opt/floorsight/jobs/setup-01/hf-models.json --vehicle-source coco --vocab-every 5 --person-conf 0.2 --vehicle-conf 0.3 --track-person-conf 0.3 > "$J/perc.log" 2>&1
echo "perception exit=$?"
for w in /opt/floorsight/perc-stock /opt/floorsight/perc; do
  $w - "$J/last.npz" <<'PY' 2>&1 | tail -3
import sys, numpy as np, cv2
d = np.load(sys.argv[1])
pts, shape = d["pts"], tuple(int(v) for v in d["shape"])
print(cv2.__version__, "shape", shape, "n_pts", len(pts), "min", pts.min(0).tolist(), "max", pts.max(0).tolist(), "dtype", pts.dtype, flush=True)
print("pts", pts.tolist()[:40], flush=True)
m = np.zeros(shape, np.uint8)
cv2.fillPoly(m, [pts], 1)
print("fillPoly ok", int(m.sum()), flush=True)
PY
  echo "  replay rc=${PIPESTATUS[0]} ($w)"
done
aws s3 cp --region us-east-1 "$J/log.txt" "s3://$BUCKET/jobs/$JOB/log.txt" --only-show-errors
aws s3 cp --region us-east-1 "$J/last.npz" "s3://$BUCKET/jobs/$JOB/last.npz" --only-show-errors
echo DONE > "$J/DONE"
