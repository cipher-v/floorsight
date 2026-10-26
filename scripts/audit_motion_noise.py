"""How jumpy are a perception run's floor tracks? Numbers only (reads <clip>-detections.jsonl, no video).

    python scripts/audit_motion_noise.py RUN_DIR [CLIP ...] [--out noise.json]

Per clip: forklift footprint jumps, size changes, heading and turn flicker, speed flicker, "moving" flicker, track
fragments (ids that live only a few frames), person floor-point jumps and warning on/off flicker. Used to compare a run
before and after motion smoothing.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

CLIPS = ["screen-002", "clip-dev-004", "screen-032", "clip-dev-002", "screen-033", "clip-dev-003"]


def area(poly):
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def centroid(poly):
    """Area centre of a polygon (the mean of its corners moves when the corner count changes, the area centre does not)."""
    a = cx = cy = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        c = x1 * y2 - x2 * y1
        a += c
        cx += (x1 + x2) * c
        cy += (y1 + y2) * c
    if abs(a) < 1e-9:
        return st.mean(q[0] for q in poly), st.mean(q[1] for q in poly)
    return cx / (3 * a), cy / (3 * a)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def flips(seq):
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def clip_noise(path: Path) -> dict:
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    dt = rows[1]["t"] - rows[0]["t"] if len(rows) > 1 else 0.04
    veh, per = {}, {}
    life_v, life_p = {}, {}
    for r in rows:
        for v in r.get("vehicles", []):
            life_v.setdefault(v["id"], []).append(r["frame"])
            veh.setdefault(v["id"], {}).setdefault("movingFlag", []).append((r["frame"], bool(v.get("moving"))))
        for p in r.get("persons", []):
            if not p.get("role"):
                life_p.setdefault(p["id"], []).append(r["frame"])
        for v in (r.get("floor") or {}).get("vehicles", []):
            fp = v.get("footprint") or []
            if not fp:
                continue
            cx, cy = centroid(fp)
            # the turning decision: the explicit flag where the run records it (v10), else most draws turning
            turning = v["turning"] if v.get("turning") is not None else v["pTurning"] >= 0.5
            veh[v["id"]].setdefault("floor", []).append((r["frame"], cx, cy, area(fp), v["heading_rad"], v["speed_mps"][1],
                                                         turning, v["stoppingDistance_m"][1]))
        for pid, p in ((r.get("floor") or {}).get("persons") or {}).items():
            if p.get("xy_m"):
                per.setdefault(pid, []).append((r["frame"], p["xy_m"][0], p["xy_m"][1], bool(p.get("warn")),
                                                p.get("vehicleMoving"), (p.get("clearance_m") or {}).get("median")))
    jumps, areas, dhead, dspeed, turnflip, moveflip, speed_gaps, dstop = [], [], [], [], 0, 0, 0, []
    for vid, d in veh.items():
        fl = d.get("floor", [])
        for a, b in zip(fl, fl[1:]):
            if b[0] - a[0] != 1:
                continue
            jumps.append(math.hypot(b[1] - a[1], b[2] - a[2]))
            dh = abs((b[4] - a[4] + math.pi) % (2 * math.pi) - math.pi)
            dhead.append(math.degrees(dh))
            dspeed.append(abs(b[5] - a[5]))
            dstop.append(abs(b[7] - a[7]))
        areas += [x[3] for x in fl]
        turnflip += flips([bool(x[6]) for x in fl])
        sp = [x[5] > 0.3 for x in fl]
        speed_gaps += sum(1 for i in range(1, len(sp) - 1) if sp[i - 1] and not sp[i] and sp[i + 1])  # moving, 0, moving
        moveflip += flips([m for _, m in d.get("movingFlag", [])])
    pj, warnflip, warn_single = [], 0, 0
    for pid, d in per.items():
        for a, b in zip(d, d[1:]):
            if b[0] - a[0] == 1 and b[4]:
                pj.append(math.hypot(b[1] - a[1], b[2] - a[2]))
        w = [x[3] for x in d]
        warnflip += flips(w)
        warn_single += sum(1 for i in range(1, len(w) - 1) if w[i] and not w[i - 1] and not w[i + 1])
    short_v = sum(1 for f in life_v.values() if len(f) < 10)
    short_p = sum(1 for f in life_p.values() if len(f) < 10)
    r3 = lambda x: None if x is None else round(x, 3)
    return {
        "frames": len(rows), "dt_s": round(dt, 4),
        "forklift": {"footprintJump_m": {"median": r3(st.median(jumps)) if jumps else None, "p90": r3(pct(jumps, .9)), "max": r3(max(jumps)) if jumps else None},
                     "footprintArea_m2": {"min": r3(min(areas)) if areas else None, "max": r3(max(areas)) if areas else None,
                                          "cv": r3(st.pstdev(areas) / st.mean(areas)) if len(areas) > 1 else None},
                     "headingChange_deg": {"median": r3(st.median(dhead)) if dhead else None, "p90": r3(pct(dhead, .9)),
                                           "over30": sum(1 for x in dhead if x > 30)},
                     "speedChange_mps": {"p90": r3(pct(dspeed, .9)), "max": r3(max(dspeed)) if dspeed else None},
                     "stopDistanceChange_m": {"p90": r3(pct(dstop, .9)), "max": r3(max(dstop)) if dstop else None},
                     "turningFlips": turnflip, "speedDropouts": speed_gaps, "movingFlagFlips": moveflip},
        "tracks": {"vehicleIds": len(life_v), "vehicleIdsUnder10Frames": short_v, "personIds": len(life_p), "personIdsUnder10Frames": short_p},
        "persons": {"floorJumpNearMovingForklift_m": {"median": r3(st.median(pj)) if pj else None, "p90": r3(pct(pj, .9)), "max": r3(max(pj)) if pj else None},
                    "warnFlips": warnflip, "singleFrameWarnSpikes": warn_single},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = {}
    for c in a.clips or CLIPS:
        p = a.run / c / f"{c}-detections.jsonl"
        if p.exists():
            out[c] = clip_noise(p)
    for c, n in out.items():
        f, t, p = n["forklift"], n["tracks"], n["persons"]
        print(f"{c:13s} jump p90 {f['footprintJump_m']['p90']} max {f['footprintJump_m']['max']} | area {f['footprintArea_m2']['min']}-{f['footprintArea_m2']['max']} cv {f['footprintArea_m2']['cv']}"
              f" | heading p90 {f['headingChange_deg']['p90']} deg >30: {f['headingChange_deg']['over30']} | turn flips {f['turningFlips']} | speed dropouts {f['speedDropouts']}"
              f" | moving flips {f['movingFlagFlips']} | v ids {t['vehicleIds']} (short {t['vehicleIdsUnder10Frames']}) p ids {t['personIds']} (short {t['personIdsUnder10Frames']})"
              f" | person jump p90 {p['floorJumpNearMovingForklift_m']['p90']} | warn flips {p['warnFlips']} spikes {p['singleFrameWarnSpikes']}")
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
