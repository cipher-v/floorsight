#!/bin/bash
# Key frames from a job's privacy-filtered OVERLAY video (no raw pixels), side by side with per-frame floor numbers.
# Needs: JOB, BUCKET, CLIP, TIMES (space-separated seconds). Output: s3 jobs/<JOB>/keyframes/*.jpg
set -uo pipefail
J=/opt/floorsight/jobs/$JOB; O=$J/keyframes; mkdir -p $O
V=$J/$CLIP/$CLIP-overlay.mp4
for t in $TIMES; do
  ffmpeg -v error -y -ss "$t" -i "$V" -frames:v 1 -q:v 3 -vf "scale=1280:-2" "$O/$CLIP-t$t.jpg"
done
python3 - "$J/$CLIP/$CLIP-detections.jsonl" "$TIMES" > "$O/numbers.txt" <<'PY'
import json, sys
rows = {round(r["t"], 2): r for r in map(json.loads, open(sys.argv[1]))}
for t in map(float, sys.argv[2].split()):
    r = min(rows.values(), key=lambda r: abs(r["t"] - t))
    fp = r["floor"].get("persons", {})
    print(f"t={r['t']:.2f}", {pid: {k: (v.get(k) if not isinstance(v.get(k), dict) else v.get(k, {}).get("median")) for k in ("vehicle", "clearance_m", "pass_m", "ttc_s", "warn")} for pid, v in fp.items() if v.get("vehicleMoving")})
PY
cat $O/numbers.txt
aws s3 cp --region us-east-1 --recursive "$O" "s3://$BUCKET/jobs/$JOB/keyframes/" --only-show-errors
