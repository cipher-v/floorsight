#!/bin/bash
# Check the private owner review page in headless Chromium on the instance. Needs: BUCKET, JOB, SRC (s3 prefix of the page),
# SCRIPT_S3 (the checker). Results (verification receipt, desktop/mobile screenshots) to s3 jobs/<JOB>/.
set -uo pipefail
J=/opt/floorsight/jobs/$JOB; [ -e "$J" ] && { echo "job dir $J exists"; exit 2; }
mkdir -p "$J/page"
aws s3 cp --region us-east-1 --recursive "s3://$BUCKET/$SRC" "$J/page/" --only-show-errors
aws s3 cp --region us-east-1 "$SCRIPT_S3" "$J/verify.py" --only-show-errors
chown -R floorsight: "$J"
sudo -u floorsight env PLAYWRIGHT_BROWSERS_PATH=/opt/floorsight/browsers /opt/floorsight/venv-browser/bin/python "$J/verify.py" "$J/page" "$JOB" 2>&1 | tail -40
for f in review-verification.json desktop.png mobile.png; do [ -f "$J/page/$f" ] && aws s3 cp --region us-east-1 "$J/page/$f" "s3://$BUCKET/jobs/$JOB/$f" --only-show-errors; done; echo uploaded
