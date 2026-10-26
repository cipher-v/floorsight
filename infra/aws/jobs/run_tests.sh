#!/bin/bash
# AWS job: full Floorsight test suite on the instance (app venv, COOL OpenCV), result to S3 jobs/<JOB>/.
# Needs: JOB, BUCKET.
set -uo pipefail
J=/opt/floorsight/jobs/$JOB
if [ -z "${DETACHED:-}" ]; then
  [ -e "$J" ] && { echo "job dir $J exists"; exit 2; }
  mkdir -p "$J"; cp "$0" "$J/run.sh"; chown -R floorsight: "$J"
  systemd-run --unit="fs-$JOB" --uid=floorsight --property=Nice=15 --setenv=DETACHED=1 --setenv=JOB="$JOB" --setenv=BUCKET="$BUCKET" \
    --setenv=CODE="${CODE:-/opt/floorsight/current}" --property=WorkingDirectory=/opt/floorsight bash "$J/run.sh"
  echo "started fs-$JOB"; exit 0
fi
if [ -e "$J/STARTED" ]; then echo "restart refused $(date -u +%FT%TZ)" >> "$J/restarts.txt"; exit 0; fi
date -u +%FT%TZ > "$J/STARTED"
exec >> "$J/log.txt" 2>&1
cd "$CODE"
echo "code: $(readlink -f "$CODE")"
# test inputs that are not in the code bundle: the simulated warehouse (synthetic) and a scratch artifact root
[ -e data/sim/demo_001/site.json ] || aws s3 cp --region us-east-1 --recursive "s3://$BUCKET/inputs/sim/demo_001/" data/sim/demo_001/ --only-show-errors
export FLOORSIGHT_TEST_ARTIFACT_ROOT=$J/artifacts; mkdir -p "$FLOORSIGHT_TEST_ARTIFACT_ROOT"
/opt/floorsight/py -c "import cv2; print('cv2', cv2.__version__, cv2.__file__)"
HOME=$J PYTHONFAULTHANDLER=1 /opt/floorsight/py -m pytest -p no:cacheprovider -rf -v tests > "$J/pytest.txt" 2>&1
echo "PYTEST_EXIT=$?"
grep -E "passed|failed|error" "$J/pytest.txt" | tail -3
aws s3 cp --region us-east-1 "$J/pytest.txt" "s3://$BUCKET/jobs/$JOB/pytest.txt" --only-show-errors
aws s3 cp --region us-east-1 "$J/log.txt" "s3://$BUCKET/jobs/$JOB/log.txt" --only-show-errors
echo DONE > "$J/DONE"
