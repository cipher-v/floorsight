#!/bin/bash
# AWS job: COOL vs stock OpenCV on the same c8g.xlarge, same code, same real frames.
#  A. scripts/benchmark/cool_bench.py on 4 real clips (per-op latency + 4-process throughput), COOL then stock;
#  B. the real-clip perception stage on the first FR frames of screen-002, interleaved COOL, stock, stock, COOL.
# Raw frames are only read for timing; nothing but timings is written. Unattended upgrades are paused meanwhile.
# Needs: JOB, BUCKET. Optional: FR (frames for B, default 30), VOCAB, SEG, PM, TILES.
set -uo pipefail
J=/opt/floorsight/jobs/$JOB
if [ -z "${DETACHED:-}" ]; then
  [ -e "$J" ] && { echo "job dir $J exists"; exit 2; }
  mkdir -p "$J"; cp "$0" "$J/run.sh"; chown -R floorsight: "$J"
  systemd-run --unit="fs-$JOB" --setenv=DETACHED=1 --setenv=JOB="$JOB" --setenv=BUCKET="$BUCKET" --setenv=FR="${FR:-30}" \
    --setenv=VOCAB="${VOCAB:-omdet}" --setenv=SEG="${SEG:-sam-vit-base}" --setenv=PM="${PM:-yolox_l}" --setenv=TILES="${TILES:-3}" --setenv=EXTRA="${EXTRA:-}" \
    --property=WorkingDirectory=/opt/floorsight/current bash "$J/run.sh"
  echo "started fs-$JOB"; exit 0
fi
if [ -e "$J/STARTED" ]; then echo "restart refused $(date -u +%FT%TZ)" >> "$J/restarts.txt"; exit 0; fi
date -u +%FT%TZ > "$J/STARTED"
exec >> "$J/log.txt" 2>&1
set -x
A=/opt/floorsight/current
U="sudo -u floorsight -H"
systemctl stop unattended-upgrades apt-daily.timer apt-daily-upgrade.timer 2>/dev/null
pkill -f unattended-upgrade 2>/dev/null; sleep 5
F=$J/footage; mkdir -p "$F"
i=1; for c in screen-002 clip-dev-004 screen-033 clip-dev-002; do ln -s "/opt/floorsight/data/private/clips/$c.mp4" "$F/cam$i.mp4"; i=$((i+1)); done
chown -R floorsight: "$J"
# A. OpenCV operations
for leg in cool stock; do
  PYX=/opt/floorsight/perc; [ $leg = stock ] && PYX=/opt/floorsight/perc-stock
  $U $PYX "$A/scripts/benchmark/cool_bench.py" --footage "$F" --label "c8g.xlarge-$leg" --frames 60 --warmup 15 --workers 4 --seconds 15 \
     --no-pipeline --price-usd-h 0.1595 --software-usd-h $([ $leg = cool ] && echo 0.01 || echo 0) --out "$J/bench-ops-$leg.json"
  echo "ops $leg exit=$?"
done
# B. whole perception stage, same detectors, interleaved
CAL=$J/cal.json; aws s3 cp --region us-east-1 "s3://$BUCKET/inputs/floor-calibration-camA.json" "$CAL" --only-show-errors; chown floorsight: "$CAL"
n=0
for leg in cool stock stock cool; do
  n=$((n+1)); PYX=/opt/floorsight/perc; [ $leg = stock ] && PYX=/opt/floorsight/perc-stock
  T0=$(date +%s.%N)
  $U $PYX -P "$A/scripts/real_clip_perception.py" --clip /opt/floorsight/data/private/clips/screen-002.mp4 --name screen-002 \
     --output "$J/perc-$n-$leg" --models /opt/floorsight/models --calibration "$CAL" --max-frames "$FR" --vocab "$VOCAB" --segmenter "$SEG" \
     --person-model "$PM" --person-tiles "$TILES" --hf-index /opt/floorsight/jobs/setup-01/hf-models.json $EXTRA > "$J/perc-$n-$leg.log" 2>&1
  echo "perc $n $leg exit=$? wall_s=$(python3 -c "import time;print(round(time.time()-$T0,1))")"
done
python3 - "$J" > "$J/bench-summary.json" <<'PY'
import json, sys, glob, os
J = sys.argv[1]; out = {"ops": {}, "perception": []}
for leg in ("cool", "stock"):
    p = f"{J}/bench-ops-{leg}.json"
    if os.path.exists(p):
        d = json.load(open(p)); lat = d.get("latency", {})
        out["ops"][leg] = {"opencv": d["opencv"]["version"], "module": d["opencv"]["module_path"], "frame_size": lat.get("frame_size"),
                           "per_op_p50_ms": {k: v["p50_ms"] for k, v in lat.get("per_op", {}).items()},
                           "extra_p50_ms": {k: v["p50_ms"] for k, v in lat.get("extra_ops_not_in_e2e", {}).items()},
                           "e2e_core_mean_ms": lat.get("e2e_core", {}).get("mean_ms"), "e2e_server_mean_ms": lat.get("e2e_server_with_jpeg_decode", {}).get("mean_ms"),
                           "throughput": d.get("throughput"), "cost": d.get("cost")}
for p in sorted(glob.glob(f"{J}/perc-*-*/screen-002-result.json")):
    d = json.load(open(p)); t = d.get("timing", {})
    out["perception"].append({"run": os.path.basename(os.path.dirname(p)), "seconds": d.get("seconds"), "frames": t.get("detector_stages_s", {}).get("frames"),
                              "detector_s": t.get("detector_s"), "stages": t.get("detector_stages_s"), "build": t.get("opencv_build"), "cv2": t.get("cv2_file")})
print(json.dumps(out, indent=1))
PY
systemctl start apt-daily.timer apt-daily-upgrade.timer unattended-upgrades 2>/dev/null
aws s3 cp --region us-east-1 --recursive "$J" "s3://$BUCKET/jobs/$JOB/" --exclude "footage/*" --exclude "perc-*/*.mp4" --only-show-errors
echo DONE > "$J/DONE"; aws s3 cp --region us-east-1 "$J/DONE" "s3://$BUCKET/jobs/$JOB/DONE" --only-show-errors
