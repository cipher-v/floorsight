#!/bin/bash
# COOL-vs-stock chart from a finished benchmark job (numbers only). Needs: BUCKET, JOB. Output: s3 jobs/<JOB>/bench-chart.jpg (+ .json)
set -uo pipefail
J=/opt/floorsight/jobs/$JOB
[ -f "$J/DONE" ] || { echo "benchmark $JOB not finished"; exit 2; }
sudo -u floorsight /opt/floorsight/perc -P /opt/floorsight/current/scripts/benchmark/bench_chart.py "$J/bench-summary.json" "$J/bench-chart.jpg" 2>&1 | tail -5
aws s3 cp --region us-east-1 "$J/bench-chart.jpg" "s3://$BUCKET/jobs/$JOB/bench-chart.jpg" --only-show-errors
aws s3 cp --region us-east-1 "$J/bench-chart.json" "s3://$BUCKET/jobs/$JOB/bench-chart.json" --only-show-errors && echo uploaded
