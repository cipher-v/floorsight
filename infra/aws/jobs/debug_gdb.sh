#!/bin/bash
# Debug job (root): native backtrace of the COOL crash under gdb.
set -uo pipefail
J=/opt/floorsight/jobs/$JOB
if [ -z "${DETACHED:-}" ]; then
  [ -e "$J" ] && { echo "job dir $J exists"; exit 2; }
  mkdir -p "$J"; cp "$0" "$J/run.sh"; chown -R floorsight: "$J"
  systemd-run --unit="fs-$JOB" --setenv=DETACHED=1 --setenv=JOB="$JOB" --setenv=BUCKET="$BUCKET" --setenv=PYX="${PYX:-/opt/floorsight/perc}" --property=WorkingDirectory=/opt/floorsight/current bash "$J/run.sh"
  echo "started fs-$JOB"; exit 0
fi
if [ -e "$J/STARTED" ]; then exit 0; fi; date -u +%FT%TZ > "$J/STARTED"
exec >> "$J/log.txt" 2>&1
command -v gdb >/dev/null || DEBIAN_FRONTEND=noninteractive apt-get install -y -q gdb >/dev/null
A=/opt/floorsight/current
aws s3 cp --region us-east-1 "s3://$BUCKET/inputs/floor-calibration-camA.json" "$J/cal.json" --only-show-errors; chown -R floorsight: "$J"
# resolve the interpreter + environment the wrapper would use, then run python itself under gdb
set -a; [ "$PYX" = /opt/floorsight/perc ] && . /etc/floorsight.opencv.env; set +a
[ "$PYX" = /opt/floorsight/perc-stock ] && unset PYTHONPATH LD_LIBRARY_PATH
VENV=/opt/floorsight/venv-perc; [ "$PYX" = /opt/floorsight/perc-stock ] && VENV=/opt/floorsight/venv-stock
export HF_HOME=/opt/floorsight/models/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
gdb -q -batch -ex "set pagination off" -ex "set startup-with-shell off" -ex run -ex "bt 25" -ex "info sharedlibrary kleidi" --args $VENV/bin/python -P "$A/scripts/real_clip_perception.py" \
  --clip /opt/floorsight/data/private/clips/screen-002.mp4 --name screen-002 --output "$J/run" --models /opt/floorsight/models --calibration "$J/cal.json" \
  --person-model rtdetrv2 --segmenter slimsam77 --vocab gdino --hf-index /opt/floorsight/jobs/setup-01/hf-models.json --vehicle-source coco --vocab-every 5 \
  --person-conf 0.2 --vehicle-conf 0.3 --track-person-conf 0.3 ${MAXF:+--max-frames $MAXF} 2>&1 | grep -vE "FutureWarning|warnings.warn|UserWarning|from scipy|New Thread|Thread .* exited|WARN:|Downloading|libthread_db" | tail -60
aws s3 cp --region us-east-1 "$J/log.txt" "s3://$BUCKET/jobs/$JOB/log.txt" --only-show-errors
echo DONE > "$J/DONE"
