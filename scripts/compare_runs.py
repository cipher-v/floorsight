"""Compare two perception runs clip by clip (floor warnings, danger events, closest moving pass).

    python scripts/compare_runs.py --old-summary OLD/summary.txt --new NEW_DIR [--out compare.json]

OLD is the accepted run's summary.txt (one line per clip with floorEvents / closest / floorWarnings JSON); NEW_DIR holds
<clip>/<clip>-result.json. Prints a table and writes JSON. Numbers only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def old_rows(summary: Path) -> dict:
    rows = {}
    for ln in summary.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^((?:screen|clip)-[\w-]+) floorEvents (\[.*?\]) legacy .*? closest (\{.*?\}) turn .*? warnings (\[.*\])\s*$", ln)
        if m:
            rows[m.group(1)] = {"events": json.loads(m.group(2)), "closest": json.loads(m.group(3)), "warnings": json.loads(m.group(4))}
    return rows


def brief_w(ws):
    return [{"person": w["person"], "t": w.get("confirmedT"), "closestAfter_m": (w.get("closestAfter") or {}).get("clearance_m"),
             "lead_s": w.get("warningLead_s")} for w in ws]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-summary", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    old = old_rows(a.old_summary)
    out = {}
    for clip in sorted(set(old) | {p.name for p in a.new.iterdir() if (p / f"{p.name}-result.json").exists()}):
        n = json.loads((a.new / clip / f"{clip}-result.json").read_text()) if (a.new / clip / f"{clip}-result.json").exists() else None
        o = old.get(clip)
        out[clip] = {
            "old": None if o is None else {"warnings": brief_w(o["warnings"]), "dangerEvents": len(o["events"]), "closest": o["closest"]},
            "new": None if n is None else {"warnings": brief_w(n.get("floorWarnings") or []), "dangerEvents": len(n.get("floorEvents") or []),
                                           "closest": n.get("floorClosestMovingObservation"), "skips": n.get("dangerZoneSkips"),
                                           "s_per_frame": round(n["seconds"] / max(1, n["frames"]), 2) if n.get("seconds") and n.get("frames") else None},
        }
        oc = (o or {}).get("closest") or {}
        nc = ((out[clip]["new"] or {}).get("closest")) or {}
        print(f"{clip:13s} OLD warn {len((o or {}).get('warnings', [])):2d} danger {len((o or {}).get('events', [])):2d} closest {oc.get('clear', '-')!s:>6} @ {oc.get('t', '-')!s:>5}"
              f" | NEW warn {len((out[clip]['new'] or {}).get('warnings', [])):2d} danger {(out[clip]['new'] or {}).get('dangerEvents', '-')!s:>2} closest {nc.get('clear', '-')!s:>6} @ {nc.get('t', '-')!s:>5}")
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
