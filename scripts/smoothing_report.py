"""Before/after tables for motion smoothing: the same cached detections run through the 23 Oct (raw) and 24 Oct
(smoothed) tracking and floor logic. Numbers only; reads <clip>-detections.jsonl and <clip>-result.json.

    python scripts/smoothing_report.py RAW_RUN SMOOTH_RUN --old-summary 20OCT_SUMMARY --out-noise N.json --out-floor F.json

Prints two Markdown tables (noise per clip; floor results per clip with the 20 Oct run) for the technical report.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLIPS = ["screen-002", "clip-dev-004", "screen-032", "clip-dev-002", "screen-033", "clip-dev-003"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def floor_numbers(run: Path, clip: str) -> dict:
    res = json.loads((run / clip / f"{clip}-result.json").read_text())
    rows = [json.loads(ln) for ln in (run / clip / f"{clip}-detections.jsonl").read_text().splitlines() if ln.strip()]
    frames_with_zone = sum(1 for r in rows if (r.get("floor") or {}).get("vehicles"))
    return {"closest": res.get("floorClosestMovingObservation"),
            "warnings": [{"person": w["person"], "confirmedT": w.get("confirmedT"), "closestAfter_m": (w.get("closestAfter") or {}).get("clearance_m"),
                          "lead_s": w.get("warningLead_s")} for w in res.get("floorWarnings") or []],
            "framesWithForkliftOnFloor": frames_with_zone, "frames": len(rows),
            "motionSmoothing": (res.get("motionSmoothing") or {}).get("counts")}


def person_closest(run: Path, clip: str, pid: str, at_t: float | None = None) -> dict:
    rows = [json.loads(ln) for ln in (run / clip / f"{clip}-detections.jsonl").read_text().splitlines() if ln.strip()]
    best, at = None, None
    for r in rows:
        p = (r.get("floor") or {}).get("persons", {}).get(pid)
        if not p or p["clearance_m"]["median"] is None:
            continue
        if p.get("vehicleMoving") and (best is None or p["clearance_m"]["median"] < best["clear"]):
            best = {"clear": p["clearance_m"]["median"], "p5": p["clearance_m"]["p5"], "p95": p["clearance_m"]["p95"], "t": r["t"]}
        if at_t is not None and abs(r["t"] - at_t) < 0.021:
            at = p["clearance_m"]["median"]
    return {"closestWhileMoving": best, "atT": at}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("smooth", type=Path)
    ap.add_argument("--old-summary", type=Path, required=True)
    ap.add_argument("--out-noise", type=Path, required=True)
    ap.add_argument("--out-floor", type=Path, required=True)
    a = ap.parse_args()
    noise_m, cmp_m = _load("audit_motion_noise"), _load("compare_runs")
    old = cmp_m.old_rows(a.old_summary)
    noise, floor = {}, {}
    for c in CLIPS:
        noise[c] = {"raw": noise_m.clip_noise(a.raw / c / f"{c}-detections.jsonl"),
                    "smoothed": noise_m.clip_noise(a.smooth / c / f"{c}-detections.jsonl")}
        floor[c] = {"20oct": {"closest": (old.get(c) or {}).get("closest"), "warnings": len((old.get(c) or {}).get("warnings") or [])},
                    "raw": floor_numbers(a.raw, c), "smoothed": floor_numbers(a.smooth, c)}
    floor["screen-002"]["seatedOperatorP1"] = {"raw": person_closest(a.raw, "screen-002", "P1", 6.18),
                                               "smoothed": person_closest(a.smooth, "screen-002", "P1", 6.18)}
    a.out_noise.write_text(json.dumps({"schema": "floorsight.motion-noise.v1", "note": "same cached detections; raw = 23 Oct tracking and floor logic, smoothed = 24 Oct (v10); turningFlips counts the turning decision (the recorded flag where present, else most draws turning)", "clips": noise}, indent=1))
    a.out_floor.write_text(json.dumps({"schema": "floorsight.floor-compare.v2", "clips": floor}, indent=1))
    f2 = lambda x: "-" if x is None else f"{x:.2f}"
    print("| Clip | Footprint jump, 9 in 10 frames (m) | Worst jump (m) | Heading swings > 30° | Turning on/off | Forklift on the floor (frames) | Worst one-frame speed change (m/s) | Person jump, 9 in 10 (m) | One-frame warning blips |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in CLIPS:
        r, s = noise[c]["raw"], noise[c]["smoothed"]
        fr_r, fr_s = floor[c]["raw"], floor[c]["smoothed"]
        print(f"| {c} | {f2(r['forklift']['footprintJump_m']['p90'])} → {f2(s['forklift']['footprintJump_m']['p90'])} | {f2(r['forklift']['footprintJump_m']['max'])} → {f2(s['forklift']['footprintJump_m']['max'])}"
              f" | {r['forklift']['headingChange_deg']['over30']} → {s['forklift']['headingChange_deg']['over30']} | {r['forklift']['turningFlips']} → {s['forklift']['turningFlips']}"
              f" | {fr_r['framesWithForkliftOnFloor']} → {fr_s['framesWithForkliftOnFloor']} of {fr_s['frames']}"
              f" | {f2(r['forklift']['speedChange_mps']['max'])} → {f2(s['forklift']['speedChange_mps']['max'])}"
              f" | {f2(r['persons']['floorJumpNearMovingForklift_m']['p90'])} → {f2(s['persons']['floorJumpNearMovingForklift_m']['p90'])} | {r['persons']['singleFrameWarnSpikes']} → {s['persons']['singleFrameWarnSpikes']} |")
    print()
    print("| Clip | Closest moving pass, 20 Oct / 23 Oct raw / 24 Oct smoothed (m) | Confirmed warnings, 20 Oct / raw / smoothed |")
    print("|---|---|---|")
    for c in CLIPS:
        o, r, s = floor[c]["20oct"], floor[c]["raw"], floor[c]["smoothed"]
        print(f"| {c} | {f2((o['closest'] or {}).get('clear'))} / {f2((r['closest'] or {}).get('clear'))} / {f2((s['closest'] or {}).get('clear'))}"
              f" | {o['warnings']} / {len(r['warnings'])} / {len(s['warnings'])} |")
    print()
    print("screen-002 seated operator P1:", json.dumps(floor["screen-002"]["seatedOperatorP1"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
