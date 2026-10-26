#!/bin/bash
# For a clip of a perception job: every reference person whose head region is < 80 % covered (box size, coverage, frame).
set -uo pipefail
sudo -u floorsight /opt/floorsight/perc -P - "/opt/floorsight/jobs/$JOB" "/opt/floorsight/jobs/${REF_JOB:-perc-full-02}/reference" "$CLIP" <<'PY'
import json, sys, importlib.util
from collections import Counter
from pathlib import Path
spec = importlib.util.spec_from_file_location("pc", "/opt/floorsight/current/scripts/privacy_crosscheck.py"); pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)
run, ref, clip = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
regions = {r["f"]: r["heads"] + r["faces"] for r in map(json.loads, (run / clip / f"{clip}-privacy-regions.jsonl").read_text().splitlines())}
rows = []
for r in map(json.loads, (ref / f"{clip}-detections.jsonl").read_text().splitlines()):
    if r["frame"] not in regions: continue
    for p in r["persons"]:
        c = pc.coverage(pc.head_region(p["box"]), regions[r["frame"]])
        if c < 0.8:
            x1, y1, x2, y2 = p["box"]
            rows.append({"f": r["frame"], "id": p["id"], "h_px": round(y2 - y1), "w_px": round(x2 - x1), "cover": round(c, 2), "conf": p.get("conf"), "xy": [round(x1), round(y1)]})
print("uncovered", len(rows), "by person", Counter(x["id"] for x in rows).most_common(8))
for x in rows[:40]: print(x)
PY
