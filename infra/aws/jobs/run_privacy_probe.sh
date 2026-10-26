#!/bin/bash
# AWS job: which person detector recovers the heads a perception run left uncovered (numbers only).
# Needs: JOB, BUCKET, PERC_JOB, CODE (app folder holding scripts/privacy_probe.py).
set -uo pipefail
J=/opt/floorsight/jobs/$JOB
if [ -z "${DETACHED:-}" ]; then
  [ -e "$J" ] && { echo "job dir $J exists"; exit 2; }
  mkdir -p "$J"; cp "$0" "$J/run.sh"; chown -R floorsight: "$J"
  systemd-run --unit="fs-$JOB" --uid=floorsight --setenv=DETACHED=1 --setenv=JOB="$JOB" --setenv=BUCKET="$BUCKET" --setenv=PERC_JOB="$PERC_JOB" \
    --setenv=CODE="$CODE" --property=WorkingDirectory=/opt/floorsight bash "$J/run.sh"
  echo "started fs-$JOB"; exit 0
fi
if [ -e "$J/STARTED" ]; then exit 0; fi; date -u +%FT%TZ > "$J/STARTED"
exec >> "$J/log.txt" 2>&1
P=/opt/floorsight/jobs/$PERC_JOB
/opt/floorsight/perc -P "$CODE/scripts/privacy_probe.py" --crosscheck "$P/privacy-crosscheck.json" --reference "$P/reference" \
  --clips /opt/floorsight/data/private/clips --models /opt/floorsight/models --hf /opt/floorsight/jobs/setup-01/hf-models.json --out "$J/probe.json"
echo "PROBE_EXIT=$?"
aws s3 cp --region us-east-1 --recursive "$J" "s3://$BUCKET/jobs/$JOB/" --only-show-errors
echo DONE > "$J/DONE"
