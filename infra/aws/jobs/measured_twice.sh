#!/bin/bash
# The screen-002 "measured twice" chart: 20 Oct run (earlier detectors) vs a 23 Oct AWS run. Numbers only, no video.
# Needs: BUCKET, NEW_JOB (e.g. perc-full-03), CODE (app folder, e.g. /opt/floorsight/app-20261023-r22). Output: s3 jobs/<NEW_JOB>/video-inputs/
# Optional: SCRIPT_S3 (a draft of the chart script; the bundle copy is used otherwise), PERSON_B, LABEL_B.
set -uo pipefail
O=/opt/floorsight/jobs/$NEW_JOB/video-inputs; H=/opt/floorsight/jobs/$NEW_JOB/history-v7c
mkdir -p "$O" "$H"
aws s3 cp --region us-east-1 --recursive "s3://$BUCKET/inputs/video/history-v7c/" "$H/" --exclude "*.mp4" --only-show-errors
S="$CODE/scripts/measured_twice.py"
[ -n "${SCRIPT_S3:-}" ] && { S=$O/measured_twice.py; aws s3 cp --region us-east-1 "$SCRIPT_S3" "$S" --only-show-errors; }
chown -R floorsight: "$O" "$H"
sudo -u floorsight /opt/floorsight/perc -P "$S" \
  --a "$H" screen-002 "20 Oct, earlier detectors" --b "/opt/floorsight/jobs/$NEW_JOB/screen-002" screen-002 "${LABEL_B:-23 Oct, AWS, Apache-2.0 detectors}" \
  --person-a P1 --person-b "${PERSON_B:-P1}" --out "$O/screen-002-measured-twice.jpg" 2>&1 | tail -3
aws s3 cp --region us-east-1 --recursive "$O" "s3://$BUCKET/jobs/$NEW_JOB/video-inputs/" --only-show-errors && echo uploaded
