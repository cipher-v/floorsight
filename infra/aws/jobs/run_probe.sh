#!/bin/bash
# AWS job: Apache-2.0 detector candidates vs the accepted detections, on real frames (numbers only leave the machine).
# Needs: JOB, BUCKET. Optional: ONLY (comma list of candidates), FRAMES.
set -uo pipefail
J=/opt/floorsight/jobs/$JOB
if [ -z "${DETACHED:-}" ]; then
  [ -e "$J" ] && { echo "job dir $J exists"; exit 2; }
  mkdir -p "$J"; cp "$0" "$J/run.sh"; chown -R floorsight: "$J"
  systemd-run --unit="fs-$JOB" --uid=floorsight --setenv=DETACHED=1 --setenv=JOB="$JOB" --setenv=BUCKET="$BUCKET" \
    --setenv=ONLY="${ONLY:-}" --setenv=FRAMES="${FRAMES:-12}" --property=WorkingDirectory=/opt/floorsight/current bash "$J/run.sh"
  echo "started fs-$JOB"; exit 0
fi
# a job runs once: if systemd (or anything) restarts the unit, record it and stop instead of silently re-running
if [ -e "$J/STARTED" ]; then echo "restart refused $(date -u +%FT%TZ)" >> "$J/restarts.txt"; exit 0; fi
date -u +%FT%TZ > "$J/STARTED"
exec >> "$J/log.txt" 2>&1
set -x
mkdir -p "$J/reference"
aws s3 cp --region us-east-1 --recursive "s3://$BUCKET/inputs/reference-ultralytics/" "$J/reference/" --only-show-errors
/opt/floorsight/perc /opt/floorsight/current/scripts/detector_probe.py --clips /opt/floorsight/data/private/clips --reference "$J/reference" \
  --models /opt/floorsight/models --hf /opt/floorsight/jobs/setup-01/hf-models.json --out "$J/probe.json" --frames "$FRAMES" --only "$ONLY"
echo "PROBE_EXIT=$?"
aws s3 cp --region us-east-1 "$J/probe.json" "s3://$BUCKET/jobs/$JOB/probe.json" --only-show-errors
aws s3 cp --region us-east-1 "$J/log.txt" "s3://$BUCKET/jobs/$JOB/log.txt" --only-show-errors
echo DONE > "$J/DONE"
