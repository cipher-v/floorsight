#!/bin/bash
# Export an existing perception job's outputs into an app replay folder (privacy-filtered videos + floor records).
# Needs: JOB (the perception job), CLIPS.
set -euo pipefail
J=/opt/floorsight/jobs/$JOB
OUT=$J/${OUTNAME:-replay}
[ -e "$OUT" ] && { echo "$OUT exists"; exit 2; }
sudo -u floorsight /opt/floorsight/perc -P /opt/floorsight/current/scripts/export_app_replay.py --perception "$J" --clips $CLIPS --out "$OUT" 2>&1 | tail -5
ls -la "$OUT" | head
