#!/bin/bash
# Floor results of a perception job vs the accepted 20 Oct run, clip by clip (numbers only). Needs: BUCKET, JOB.
set -uo pipefail
O=/opt/floorsight/jobs/$JOB/compare; mkdir -p "$O"
aws s3 cp --region us-east-1 "s3://$BUCKET/inputs/history/v7c-summary.txt" "$O/v7c-summary.txt" --only-show-errors
chown -R floorsight: "$O"
sudo -u floorsight /opt/floorsight/perc -P /opt/floorsight/current/scripts/compare_runs.py --old-summary "$O/v7c-summary.txt" --new "/opt/floorsight/jobs/$JOB" --out "$O/compare-v7c.json" 2>&1 | tail -30
aws s3 cp --region us-east-1 "$O/compare-v7c.json" "s3://$BUCKET/jobs/$JOB/compare/compare-v7c.json" --only-show-errors && echo uploaded
