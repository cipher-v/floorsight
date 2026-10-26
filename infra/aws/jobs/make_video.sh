#!/bin/bash
# AWS job: build the demo video from a section spec (uploaded to s3 jobs/<JOB>/spec.json by the caller).
# Needs: JOB, BUCKET. Optional: NOPOLLY=1 for captions only.
set -uo pipefail
J=/opt/floorsight/jobs/$JOB
if [ -z "${DETACHED:-}" ]; then
  [ -e "$J" ] && { echo "job dir $J exists"; exit 2; }
  mkdir -p "$J"; cp "$0" "$J/run.sh"; chown -R floorsight: "$J"
  systemd-run --unit="fs-$JOB" --uid=floorsight --setenv=DETACHED=1 --setenv=JOB="$JOB" --setenv=BUCKET="$BUCKET" --setenv=NOPOLLY="${NOPOLLY:-}" --setenv=EXTRA_IN="${EXTRA_IN:-}" \
    --property=WorkingDirectory=/opt/floorsight/current bash "$J/run.sh"
  echo "started fs-$JOB"; exit 0
fi
if [ -e "$J/STARTED" ]; then echo "restart refused $(date -u +%FT%TZ)" >> "$J/restarts.txt"; exit 0; fi
date -u +%FT%TZ > "$J/STARTED"
exec >> "$J/log.txt" 2>&1
set -x
set -a; . /etc/floorsight.env; set +a
aws s3 cp --region us-east-1 "s3://$BUCKET/jobs/$JOB/spec.json" "$J/spec.json" --only-show-errors
# stills, charts and the 20 Oct filtered clip the spec refers to as $J/in/... (instance-local outputs are referenced in place)
aws s3 cp --region us-east-1 --recursive "s3://$BUCKET/inputs/video/" "$J/in/" --only-show-errors
[ -n "${EXTRA_IN:-}" ] && for x in $EXTRA_IN; do aws s3 cp --region us-east-1 --recursive "$x" "$J/in/" --only-show-errors; done
/opt/floorsight/py /opt/floorsight/current/scripts/video/build_video.py "$J/spec.json" "$J/out" ${NOPOLLY:+--no-polly}
echo "VIDEO_EXIT=$?"
aws s3 cp --region us-east-1 --recursive "$J" "s3://$BUCKET/jobs/$JOB/" --exclude "out/segments/*" --exclude "out/joined.mp4" --only-show-errors
echo DONE > "$J/DONE"; aws s3 cp --region us-east-1 "$J/DONE" "s3://$BUCKET/jobs/$JOB/DONE" --only-show-errors
