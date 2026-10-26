#!/bin/bash
# AWS job: closed-loop interlock test in the simulated warehouse (scripts/interlock_sim.py), interlock OFF and ON.
# Needs: JOB, BUCKET. Optional: CODE (app folder, default current), DURATION (s, default 40), MODES (default off,on).
# Output: s3 jobs/<JOB>/ (results JSON, per-frame records, composite videos; all simulated footage, no real people)
set -uo pipefail
J=/opt/floorsight/jobs/$JOB
if [ -z "${DETACHED:-}" ]; then
  [ -e "$J" ] && { echo "job dir $J exists"; exit 2; }
  mkdir -p "$J"; cp "$0" "$J/run.sh"; chown -R floorsight: "$J"
  systemd-run --unit="fs-$JOB" --uid=floorsight --property=Nice=10 --setenv=DETACHED=1 --setenv=JOB="$JOB" --setenv=BUCKET="$BUCKET" \
    --setenv=CODE="${CODE:-/opt/floorsight/current}" --setenv=DURATION="${DURATION:-40}" --setenv=MODES="${MODES:-off,on}" \
    --property=WorkingDirectory=/opt/floorsight bash "$J/run.sh"
  echo "started fs-$JOB"; exit 0
fi
if [ -e "$J/STARTED" ]; then echo "restart refused $(date -u +%FT%TZ)" >> "$J/restarts.txt"; exit 0; fi
date -u +%FT%TZ > "$J/STARTED"
exec >> "$J/log.txt" 2>&1
cd "$CODE"
echo "code: $(readlink -f "$CODE")"
HOME=$J /opt/floorsight/py scripts/interlock_sim.py --out "$J/out" --duration "$DURATION" --modes "$MODES"
echo "SIM_EXIT=$?"
aws s3 cp --region us-east-1 --recursive "$J" "s3://$BUCKET/jobs/$JOB/" --only-show-errors
echo DONE > "$J/DONE"; aws s3 cp --region us-east-1 "$J/DONE" "s3://$BUCKET/jobs/$JOB/DONE" --only-show-errors
