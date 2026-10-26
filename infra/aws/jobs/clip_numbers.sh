#!/bin/bash
# Floor numbers of one finished clip: warnings, closest moving pass, and (optional) one person's clearance around time T.
# Needs: JOB, CLIP. Optional: PID, T.
set -uo pipefail
sudo -u floorsight /opt/floorsight/perc -P - "/opt/floorsight/jobs/$JOB/$CLIP" "$CLIP" "${PID:-}" "${T:-}" <<'PY'
import json, sys
from pathlib import Path
d, name, pid, t = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
r = json.loads((d / f"{name}-result.json").read_text())
print("warnings", [(w["person"], w.get("confirmedT"), w.get("endT"), (w.get("closestAfter") or {}).get("clearance_m"), w.get("warningLead_s")) for w in r.get("floorWarnings") or []])
print("danger", [(e["person"], e.get("startT"), e.get("endT")) for e in r.get("floorEvents") or []])
print("closest", r.get("floorClosestMovingObservation"), "keyFrame", r.get("keyFrame"), "skips", r.get("dangerZoneSkips"))
tm = r.get("timing") or {}
print("s_per_frame", round((tm.get("detector_s") or 0) / max(1, r.get("frames") or 1), 2), "frames", r.get("frames"))
if pid and t:
    t = float(t)
    for line in (d / f"{name}-detections.jsonl").read_text().splitlines():
        row = json.loads(line)
        if abs(row["t"] - t) <= 0.5:
            p = (row.get("floor") or {}).get("persons", {}).get(pid)
            if p:
                c = p["clearance_m"]
                print(round(row["t"], 2), pid, "gap", c.get("median"), c.get("p5"), c.get("p95"), "moving", p.get("vehicleMoving"), "veh", p.get("vehicle"))
PY
