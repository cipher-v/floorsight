#!/bin/bash
# Privacy cross-check on the clips a running perception job has finished so far (boxes only).
# Needs: JOB. Uses the reference detections downloaded by an earlier job (REF_JOB, default perc-full-02).
set -uo pipefail
J=/opt/floorsight/jobs/$JOB
R=/opt/floorsight/jobs/${REF_JOB:-perc-full-02}/reference
sudo -u floorsight /opt/floorsight/perc -P /opt/floorsight/current/scripts/privacy_crosscheck.py --run "$J" --reference "$R" --out "/tmp/crosscheck-$JOB.json" 2>&1 | tail -2
