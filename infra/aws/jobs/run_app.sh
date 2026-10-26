#!/bin/bash
# AWS job: the real clips inside the Floorsight app on the instance (private: 127.0.0.1 only).
#   replay folder (from a perception job) -> server (vision replay, real agent; Bedrock planner if LLM=bedrock)
#   -> evidence mirrored to S3, alerts to SNS, track snapshots to DynamoDB, metrics to CloudWatch
#   -> headless Chromium records the dashboard; a SCRIPTED STAND-IN reviewer approves/rejects as the agent recommends.
# Needs: JOB, BUCKET, PERC_JOB (perception job with replay/). Optional: LLM (bedrock|local), SPEED, VOCAB, SEG, MAXS.
# SOURCE=simlive instead runs the same app on the closed-loop simulated warehouse (4 rendered cameras, the interlock
# drives the simulated forklift; no PERC_JOB, no real footage); SIMDUR = its length in seconds.
set -uo pipefail
J=/opt/floorsight/jobs/$JOB
if [ -z "${DETACHED:-}" ]; then
  [ -e "$J" ] && { echo "job dir $J exists"; exit 2; }
  mkdir -p "$J"; cp "$0" "$J/run.sh"; chown -R floorsight: "$J"
  systemd-run --unit="fs-$JOB" --uid=floorsight --setenv=DETACHED=1 --setenv=JOB="$JOB" --setenv=BUCKET="$BUCKET" --setenv=PERC_JOB="${PERC_JOB:-}" \
    --setenv=LLM="${LLM:-bedrock}" --setenv=SPEED="${SPEED:-0.5}" --setenv=VOCAB="${VOCAB:-omdet}" --setenv=SEG="${SEG:-sam-vit-base}" \
    --setenv=MAXS="${MAXS:-900}" --setenv=REPLAY="${REPLAY:-replay}" --setenv=SOURCE="${SOURCE:-realclip}" --setenv=SIMDUR="${SIMDUR:-40}" \
    --property=WorkingDirectory=/opt/floorsight/current bash "$J/run.sh"
  echo "started fs-$JOB"; exit 0
fi
if [ -e "$J/STARTED" ]; then echo "restart refused $(date -u +%FT%TZ)" >> "$J/restarts.txt"; exit 0; fi
date -u +%FT%TZ > "$J/STARTED"
exec >> "$J/log.txt" 2>&1
set -x
A=/opt/floorsight/current
P=/opt/floorsight/jobs/${PERC_JOB:-none}
RP=$P/$REPLAY
[ "$SOURCE" = simlive ] || [ -d "$RP" ] || { echo "no replay folder $RP"; exit 2; }
mkdir -p "$J/private"
python3 - "$J/private/raw-clips.json" <<'PY'
import json, sys, glob, os
json.dump({os.path.basename(p)[:-4]: p for p in glob.glob("/opt/floorsight/data/private/clips/*.mp4")}, open(sys.argv[1], "w"), indent=1)
PY
# AWS settings written by Terraform (S3, SNS, DynamoDB, CloudWatch, Bedrock model id)
set -a; . /etc/floorsight.env; set +a
export FLOORSIGHT_CLOUD=aws FLOORSIGHT_SESSION_ID="$JOB" FLOORSIGHT_LLM="$LLM" FLOORSIGHT_S3_PREFIX="evidence/$JOB/"
unset FLOORSIGHT_TOKEN FLOORSIGHT_ORIGIN_SECRET
export FLOORSIGHT_REDETECT_CMD="env FLOORSIGHT_MODELS=/opt/floorsight/models FLOORSIGHT_HF_INDEX=/opt/floorsight/jobs/setup-01/hf-models.json FLOORSIGHT_RAW_CLIPS=$J/private/raw-clips.json FLOORSIGHT_VOCAB=$VOCAB FLOORSIGHT_SEGMENTER=$SEG /opt/floorsight/perc -P $A/scripts/second_look.py"
PORT=8811
cd "$A"
if [ "$SOURCE" = simlive ]; then
  /opt/floorsight/py -m floorsight.server --source simlive --sim-duration "$SIMDUR" --vision real --agent real \
    --speed "$SPEED" --no-drop --host 127.0.0.1 --port $PORT --runtime-dir "$J/runtime" > "$J/server.log" 2>&1 &
else
  /opt/floorsight/py -m floorsight.server --source realclip --sim-dir "$RP" --vision replay --agent real \
    --speed "$SPEED" --no-drop --host 127.0.0.1 --port $PORT --runtime-dir "$J/runtime" > "$J/server.log" 2>&1 &
fi
SRV=$!
for i in $(seq 1 90); do curl -sf "http://127.0.0.1:$PORT/healthz" > /dev/null && break; sleep 1; done
curl -sf "http://127.0.0.1:$PORT/api/site" > "$J/api-site.json" || { echo "server did not start"; tail -30 "$J/server.log"; kill $SRV; exit 3; }
PLAYWRIGHT_BROWSERS_PATH=/opt/floorsight/browsers /opt/floorsight/venv-browser/bin/python "$A/infra/aws/jobs/drive_dashboard.py" \
  "http://127.0.0.1:$PORT" "$J/session" --max-s "$MAXS" | tee "$J/session.log"
curl -sf "http://127.0.0.1:$PORT/api/state" > "$J/api-state-final.json"
curl -sf "http://127.0.0.1:$PORT/api/report" > "$J/api-report.json"
kill -INT $SRV 2>/dev/null
for i in $(seq 1 20); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null
python3 - "$J" <<'PY'
import json, sys, glob
from pathlib import Path
out = Path(sys.argv[1])
st = json.loads((out / "api-state-final.json").read_text())
ag = st["agent"]; tr = ag["trace"]
modes = {}
for s in tr:
    modes[s.get("mode")] = modes.get(s.get("mode"), 0) + 1
summary = {"mode": st["mode"], "components": st["components"], "cloud": st.get("cloud"), "privacy": st["privacy"],
           "events": [(e["event_id"], e["type"], e["severity"], e["status"], e["summary"][:140]) for e in ag["events"]],
           "approvals": [(a["approval_id"], a["status"], (a.get("payload") or {}).get("card")) for a in ag["approvals"]],
           "alerts": [(a["alert_id"], a["audience"], a["message"][:120]) for a in ag["alerts"]],
           "steps": len(tr), "stepModes": modes, "tools": sorted({s["tool"] for s in tr}),
           "human_decisions": [(s["event_id"], s["tool_input"].get("reviewer"), s["decision"]) for s in tr if s["tool"] == "human_decision"],
           "trace_files": sorted(glob.glob(str(out / "runtime" / "traces" / "*"))),
           "interlock": st.get("interlock")}
(out / "run-summary.json").write_text(json.dumps(summary, indent=1))
print("SUMMARY", json.dumps(summary)[:4000])
PY
# the session video is a recording of the privacy-filtered dashboard; raw clips are never served by the app
aws s3 cp --region us-east-1 --recursive "$J" "s3://$BUCKET/jobs/$JOB/" --exclude "private/*" --only-show-errors
echo DONE > "$J/DONE"; aws s3 cp --region us-east-1 "$J/DONE" "s3://$BUCKET/jobs/$JOB/DONE" --only-show-errors
