#!/bin/bash
# AWS job (27 Oct, audit C1): replay the floor records of a real-clip app run through the r55 interlock and the fixed one,
# and list what each would have commanded (no video decoded, no detector run; seconds). Result to S3 jobs/<JOB>/.
# Needs: JOB, BUCKET. Optional: APPJOB (default app-run-09), OLD (the interlock.py to compare against).
set -uo pipefail
J=/opt/floorsight/jobs/$JOB
[ -e "$J" ] && { echo "job dir $J exists"; exit 2; }
mkdir -p "$J"
APPJOB=${APPJOB:-app-run-09}
OLD=${OLD:-/opt/floorsight/app-20261024-r55/floorsight/interlock.py}
LOG=/opt/floorsight/jobs/$APPJOB/log.txt
P=$(grep -m1 -oE '^\+ P=/opt/floorsight/jobs/[^ ]+' "$LOG" | cut -d= -f2)
REPLAY=$(grep -m1 -oE '^\+ RP=[^ ]+' "$LOG" | cut -d= -f2)
RP=${REPLAY:-$P/replay}
echo "app job $APPJOB: replay folder $RP; old interlock $OLD; code $(readlink -f /opt/floorsight/current)" | tee "$J/result.txt"
[ -d "$RP" ] && [ -f "$OLD" ] || { echo "missing replay folder or old interlock" | tee -a "$J/result.txt"; exit 3; }
cd /opt/floorsight/current
# /opt/floorsight/py sets its own PYTHONPATH (the COOL build), so the code folder goes on sys.path here
/opt/floorsight/py -c 'import runpy, sys; sys.path.insert(0, "/opt/floorsight/current"); sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")'   scripts/interlock_replay_check.py "$RP" "$OLD" --out "$J/result.json" >> "$J/result.txt" 2>&1
echo "EXIT=$?" >> "$J/result.txt"
cat "$J/result.txt"
aws s3 cp --region us-east-1 "$J/result.txt" "s3://$BUCKET/jobs/$JOB/result.txt" --only-show-errors
aws s3 cp --region us-east-1 "$J/result.json" "s3://$BUCKET/jobs/$JOB/result.json" --only-show-errors
