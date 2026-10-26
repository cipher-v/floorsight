#!/bin/bash
# AWS job: real-clip perception (Apache-2.0 detectors) on the 6 clips -> filtered videos, floor records, results;
# then the fail-closed privacy self-test, the screen-002 demo-moment figure and the app replay export.
# Needs: JOB, BUCKET. Optional: PY (/opt/floorsight/perc = COOL, /opt/floorsight/perc-stock = stock wheel), CLIPS,
#        VOCAB, SEG, PM (person model), TILES, MAXF (max frames per clip, 0 = all), EXPORT (1 = build the app replay),
#        SAVE_DET=1 (keep every frame's detector output in $J/cache/<clip>.pkl), LOAD_DET=<dir> (replay those instead of
#        running the detectors: tracking, floor and zone logic only, minutes instead of hours), CODE (code folder).
set -uo pipefail
J=/opt/floorsight/jobs/$JOB
if [ -z "${DETACHED:-}" ]; then
  [ -e "$J" ] && { echo "job dir $J exists"; exit 2; }
  mkdir -p "$J"; cp "$0" "$J/run.sh"; chown -R floorsight: "$J"
  systemd-run --unit="fs-$JOB" --uid=floorsight --setenv=DETACHED=1 --setenv=JOB="$JOB" --setenv=BUCKET="$BUCKET" \
    --setenv=PY="${PY:-/opt/floorsight/perc}" --setenv=CLIPS="${CLIPS:-screen-002 clip-dev-004 screen-032 clip-dev-002 screen-033 clip-dev-003}" \
    --setenv=VOCAB="${VOCAB:-omdet}" --setenv=SEG="${SEG:-sam-vit-base}" --setenv=PM="${PM:-yolox_l}" --setenv=TILES="${TILES:-3}" --setenv=EXTRA="${EXTRA:-}" \
    --setenv=MAXF="${MAXF:-0}" --setenv=EXPORT="${EXPORT:-1}" --setenv=SAVE_DET="${SAVE_DET:-0}" --setenv=LOAD_DET="${LOAD_DET:-}" --setenv=CODE="${CODE:-/opt/floorsight/current}" --property=WorkingDirectory=/opt/floorsight/current bash "$J/run.sh"
  echo "started fs-$JOB"; exit 0
fi
if [ -e "$J/STARTED" ]; then echo "restart refused $(date -u +%FT%TZ)" >> "$J/restarts.txt"; exit 0; fi
date -u +%FT%TZ > "$J/STARTED"
exec >> "$J/log.txt" 2>&1
set -x
A=$CODE  # default /opt/floorsight/current; a job can pin an unpacked bundle while another job uses current
export PYTHONFAULTHANDLER=1  # a native crash prints the Python stack to the clip log
CAL=$J/floor-calibration-camA.json
aws s3 cp --region us-east-1 "s3://$BUCKET/inputs/floor-calibration-camA.json" "$CAL" --only-show-errors
cp "$CAL" "$J/calibration-used.json"  # name the export step expects
echo "code $(readlink -f $A) py=$PY vocab=$VOCAB seg=$SEG person=$PM tiles=$TILES maxf=$MAXF" > "$J/summary.txt"
$PY -c "import cv2;print('cv2',cv2.__version__,cv2.__file__)" >> "$J/summary.txt"
COMMON=(--models /opt/floorsight/models --calibration "$CAL" --vocab "$VOCAB" --segmenter "$SEG" --person-model "$PM" --person-tiles "$TILES"
        --hf-index /opt/floorsight/jobs/setup-01/hf-models.json)
[ "$MAXF" != "0" ] && COMMON+=(--max-frames "$MAXF")
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && COMMON+=($EXTRA)  # e.g. "--vehicle-source coco --vocab-every 5 --person-conf 0.3"
echo "extra: $EXTRA save_det=$SAVE_DET load_det=$LOAD_DET" >> "$J/summary.txt"
STATUS=0
for NAME in $CLIPS; do
  T0=$(date +%s)
  DETS=()
  [ "$SAVE_DET" = "1" ] && { mkdir -p "$J/cache"; DETS=(--save-detections "$J/cache/$NAME.pkl"); }
  [ -n "$LOAD_DET" ] && DETS=(--load-detections "$LOAD_DET/$NAME.pkl")
  $PY -P "$A/scripts/real_clip_perception.py" --clip "/opt/floorsight/data/private/clips/$NAME.mp4" --name "$NAME" --output "$J/$NAME" "${COMMON[@]}" "${DETS[@]}" > "$J/$NAME.log" 2>&1
  RC=$?; echo "$NAME exit=$RC wall_s=$(( $(date +%s) - T0 ))" >> "$J/summary.txt"; [ $RC -eq 0 ] || STATUS=1
  $PY -P -c "import json;d=json.load(open('$J/$NAME/$NAME-result.json'));print('$NAME','floorWarnings',json.dumps(d.get('floorWarnings'))[:600],'closest',json.dumps(d.get('floorClosestMovingObservation'))[:300],'timing',json.dumps(d.get('timing')))" >> "$J/summary.txt" 2>&1
done
# privacy cross-check: every person the accepted (pre-23-Oct) run saw must be pixelated in this run too
mkdir -p "$J/reference"
aws s3 cp --region us-east-1 --recursive "s3://$BUCKET/inputs/reference-ultralytics/" "$J/reference/" --only-show-errors
$PY -P "$A/scripts/privacy_crosscheck.py" --run "$J" --reference "$J/reference" --out "$J/privacy-crosscheck.json" >> "$J/summary.txt" 2>&1
# fail-closed privacy self-test: frames 3 and 7 must come out black and be listed as withheld
$PY -P "$A/scripts/real_clip_perception.py" --clip /opt/floorsight/data/private/clips/clip-dev-004.mp4 --name failclosed --output "$J/failclosed-test" \
  --max-frames 12 --force-privacy-failure-frame 3 --force-privacy-failure-frame 7 "${COMMON[@]}" ${LOAD_DET:+--load-detections "$LOAD_DET/clip-dev-004.pkl"} > "$J/failclosed-test.log" 2>&1
$PY -P - "$J/failclosed-test" >> "$J/summary.txt" 2>&1 <<'PY'
import sys, json, cv2
from pathlib import Path
d = Path(sys.argv[1]); cap = cv2.VideoCapture(str(d / 'failclosed-overlay.mp4')); means = []
while True:
    ok, f = cap.read()
    if not ok: break
    means.append(float(f[100:, :].mean()))
res = json.loads((d / 'failclosed-result.json').read_text())
dark = [i for i, m in enumerate(means) if m < 25]
v = {'framesEncoded': len(means), 'darkFrames': dark, 'withheldInHealth': [w['frame'] for w in res['privacy']['withheld']], 'opencv': cv2.__version__}
v['pass'] = dark == [3, 7] and v['withheldInHealth'] == [3, 7]
(d / 'failclosed-verdict.json').write_text(json.dumps(v, indent=2)); print('FAILCLOSED', json.dumps(v))
PY
if [ -d "$J/screen-002" ]; then
  $PY -P "$A/scripts/demo_moment.py" --run "$J/screen-002" --name screen-002 --out "$J/demo" > "$J/demo-screen-002.log" 2>&1
  echo "demo screen-002 exit=$?" >> "$J/summary.txt"; grep '^DEMO' "$J/demo-screen-002.log" >> "$J/summary.txt"
fi
if [ "$EXPORT" = "1" ] && [ $STATUS -eq 0 ]; then
  $PY -P "$A/scripts/export_app_replay.py" --perception "$J" --clips $CLIPS --out "$J/replay" > "$J/export.log" 2>&1
  echo "export exit=$?" >> "$J/summary.txt"
fi
echo "STATUS=$STATUS" >> "$J/summary.txt"
# privacy-filtered outputs only (no raw frames exist in the job folder); raw clips stay in data/private
aws s3 cp --region us-east-1 --recursive "$J" "s3://$BUCKET/jobs/$JOB/" --only-show-errors
echo DONE > "$J/DONE"; aws s3 cp --region us-east-1 "$J/DONE" "s3://$BUCKET/jobs/$JOB/DONE" --only-show-errors
