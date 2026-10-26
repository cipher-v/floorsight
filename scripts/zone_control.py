"""Control test of the forklift danger-zone alert on REAL forklift motion with VIRTUAL people. GCP ONLY.

No real clip in the screened dataset has a person inside a forklift's danger zone (closest moving passes
1.0-1.4 m, outside every zone), so the alert has never been seen firing on real footage. This test replays a
perception run's recorded forklift outlines (per-frame truck masks from <name>-detections.jsonl) through the
same DangerZone code and alert rule, and places virtual people on the floor:
  V-path  standing where the forklift's footprint will be LEAD_S seconds later (on its path): must alert
          before the forklift reaches them;
  V-side  SIDE_NEAR m to the side of that point: borderline, reported with its probabilities;
  V-far   SIDE_FAR m to the side: must stay silent.
Each virtual person is projected into the image with the nominal camera (feet and a 1.70 m head) and then
handled exactly like a detected person (per-draw floor positions, same rule, 3-frame confirmation).
Carried-load masks are not stored per frame, so the replayed unit is the truck mask alone and the blind zone
is not exercised. A virtual person is not a real near miss; this checks the logic, not the world.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

HEAD_M = 1.70
CONTACT_M = 0.3   # clearance treated as contact
EVENT_MIN_FRAMES = 3


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def vehicles_of(row):
    return [{"id": v["id"], "polygons": [np.asarray(q, np.float32) for q in (v.get("polygons") or []) if len(q) >= 3],
             "loadPolygons": [], "moving": v.get("moving", False), "estimated": v.get("estimated", False)} for v in row["vehicles"]]


def virtual_person(cam, xy, size):
    foot = cam.project(np.array([[xy[0], xy[1], 0.0]]))[0]
    head = cam.project(np.array([[xy[0], xy[1], HEAD_M]]))[0]
    if not (np.isfinite(foot).all() and np.isfinite(head).all()):
        return None
    if not (0 <= foot[0] < size[0] and 0 <= foot[1] < size[1]):
        return None
    h = foot[1] - head[1]
    if h <= 4:
        return None
    return {"box": [foot[0] - h / 6, head[1], foot[0] + h / 6, foot[1]], "foot": [float(foot[0]), float(foot[1])]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="perception output folder of one clip")
    ap.add_argument("--name", required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lead-s", type=float, default=2.0)
    ap.add_argument("--side-near", type=float, default=1.5)
    ap.add_argument("--side-far", type=float, default=3.0)
    ap.add_argument("--draws", type=int, default=64)
    a = ap.parse_args()
    dzm = load("danger_zone")
    res = json.loads((a.run / f"{a.name}-result.json").read_text())
    fps = float(res["fps"])
    fs = res["frameSize"]
    size = (int(fs["width"]), int(fs["height"])) if isinstance(fs, dict) else (int(fs[0]), int(fs[1]))
    rows = [json.loads(line) for line in (a.run / f"{a.name}-detections.jsonl").read_text().splitlines() if line.strip()]

    # pass 1: the forklift's own path (nominal footprint of every moving frame)
    dz = dzm.DangerZone(a.calibration, size, fps, draws=a.draws)
    path = {}
    for r in rows:
        out = dz.step(r["t"], vehicles_of(r), [])
        for v in out["vehicles"]:
            if v["speed_mps"][1] >= dzm.MOVING_MPS:
                fp = np.asarray(v["footprint"], float)
                path.setdefault(v["id"], []).append({"frame": r["frame"], "t": r["t"], "c": fp.mean(0), "fp": fp, "speed": v["speed_mps"][1]})
    if not path:
        raise SystemExit("no moving forklift in this run")
    vid = max(path, key=lambda k: len(path[k]))
    P = path[vid]
    # reference: a moving frame early enough that the footprint LEAD_S later is still on record
    later = [i for i, p in enumerate(P) if any(abs(q["t"] - (p["t"] + a.lead_s)) < 0.5 / fps + 1e-6 for q in P)]
    if not later:
        raise SystemExit("moving stretch shorter than the lead time")
    i_ref = later[len(later) // 3]
    ref = P[i_ref]
    tgt = min(P, key=lambda q: abs(q["t"] - (ref["t"] + a.lead_s)))
    d = tgt["c"] - ref["c"]
    u = d / max(np.linalg.norm(d), 1e-6)
    left = np.array([-u[1], u[0]])
    # side people go to the side nearer the camera when possible (more likely in view)
    side = left if (left @ (dz.camera_xy - tgt["c"])) > 0 else -left
    people = {"V-path": tgt["c"], "V-side": tgt["c"] + side * a.side_near, "V-far": tgt["c"] + side * a.side_far}

    # pass 2: same code, same rule, virtual people as detections
    dz = dzm.DangerZone(a.calibration, size, fps, draws=a.draws)
    streak = {k: 0 for k in people}
    wstreak = {k: 0 for k in people}
    first_warn = {k: None for k in people}
    log = {k: [] for k in people}
    first_alert = {k: None for k in people}
    for r in rows:
        persons = []
        for pid, xy in people.items():
            vp = virtual_person(dz.nominal, xy, size)
            if vp is not None:
                persons.append({"id": pid, "box": vp["box"], "foot": vp["foot"], "operator": False})
        out = dz.step(r["t"], vehicles_of(r), persons)
        moving_ids = {v["id"] for v in out["vehicles"] if v["speed_mps"][1] >= dzm.MOVING_MPS}
        for pid in people:
            rr = out["persons"].get(pid)
            if rr is None:
                streak[pid] = 0; wstreak[pid] = 0
                continue
            moving = rr.get("vehicle") in moving_ids
            p_path = max(rr["pInZone"]["stop"], rr["pInZone"]["tail"])
            clear = rr["clearance_m"]
            reasons = [n for n, ok in (("in stopping/swing path", p_path >= 0.5), ("hidden in driver blind zone", rr["pInZone"]["blind"] >= 0.5),
                                       ("closer than 0.8 m", clear["median"] < 0.8)) if ok and moving]
            streak[pid] = streak[pid] + 1 if reasons else 0
            if streak[pid] >= EVENT_MIN_FRAMES and first_alert[pid] is None:
                first_alert[pid] = {"t": round(r["t"], 3), "frame": r["frame"], "reasons": reasons}
            warn = bool(moving and rr.get("pWarn", 0.0) >= 0.5)
            wstreak[pid] = wstreak[pid] + 1 if warn else 0
            if wstreak[pid] >= EVENT_MIN_FRAMES and first_warn[pid] is None:
                first_warn[pid] = {"t": round(r["t"], 3), "frame": r["frame"], "ttc_s": rr.get("ttc_s")}
            log[pid].append({"t": round(r["t"], 3), "pStop": rr["pInZone"]["stop"], "pTail": rr["pInZone"]["tail"],
                             "clear": {k: (round(v, 3) if np.isfinite(v) else None) for k, v in clear.items()},
                             "moving": moving, "alert": bool(reasons), "confirmed": streak[pid] >= EVENT_MIN_FRAMES,
                             "pWarn": round(rr.get("pWarn", 0.0), 3), "warnConfirmed": wstreak[pid] >= EVENT_MIN_FRAMES})
    summary = {}
    for pid in people:
        lg = log[pid]
        contact = next((x["t"] for x in lg if x["clear"]["median"] is not None and x["clear"]["median"] < CONTACT_M), None)
        cl = [x["clear"]["median"] for x in lg if x["clear"]["median"] is not None]
        fa = first_alert[pid]
        summary[pid] = {"floorXY_m": [round(float(v), 3) for v in people[pid]], "framesInView": len(lg),
                        "minClearance_m": round(min(cl), 3) if cl else None, "maxPStop": round(max((x["pStop"] for x in lg), default=0.0), 3),
                        "maxPTail": round(max((x["pTail"] for x in lg), default=0.0), 3), "firstConfirmedAlert": fa, "contactT": contact,
                        "warningBeforeContact_s": (round(contact - fa["t"], 3) if (fa and contact is not None) else None),
                        "alarmBeforeContact_s": (round(contact - fa["t"], 3) if (fa and contact is not None) else None),
                        "firstConfirmedWarning": first_warn[pid],
                        "warning3sBeforeContact_s": (round(contact - first_warn[pid]["t"], 3) if (first_warn[pid] and contact is not None) else None)}
    expect = {"V-path": "alert before contact", "V-side": "borderline (reported)", "V-far": "no alert"}
    verdict = {"V-path": bool(summary["V-path"]["firstConfirmedAlert"]) and (summary["V-path"]["contactT"] is None or summary["V-path"]["warningBeforeContact_s"] > 0),
               "V-far": summary["V-far"]["firstConfirmedAlert"] is None,
               "V-path warned earlier than the alarm": bool(summary["V-path"]["firstConfirmedWarning"] and summary["V-path"]["firstConfirmedAlert"]
                                                            and summary["V-path"]["firstConfirmedWarning"]["t"] < summary["V-path"]["firstConfirmedAlert"]["t"])}
    a.out.mkdir(parents=True, exist_ok=True)
    doc = {"schema": "floorsight.zone-control.v1", "opencv": cv2.__version__, "clip": a.name, "vehicle": vid,
           "reference": {"t": ref["t"], "speed_mps": round(ref["speed"], 2)}, "leadS": a.lead_s, "sideNear_m": a.side_near, "sideFar_m": a.side_far,
           "expected": expect, "pass": verdict, "people": summary, "log": log,
           "limits": ["Virtual people on real forklift motion: tests the alert logic, not a real near miss.",
                      "Replayed unit = recorded truck mask only (carried-load masks are not stored per frame): no blind zone tested; the unit is shorter than with its load.",
                      "Same calibration, priors, draws and 3-frame rule as the perception run."]}
    (a.out / f"{a.name}-zone-control.json").write_text(json.dumps(doc, indent=1))
    figure(a, dz, P, people, log, summary, ref, tgt)
    print("CONTROL", a.name, json.dumps({"pass": verdict, "people": {k: {x: v[x] for x in ("minClearance_m", "maxPStop", "firstConfirmedAlert", "contactT", "alarmBeforeContact_s",
                                                                                            "firstConfirmedWarning", "warning3sBeforeContact_s")} for k, v in summary.items()}}))
    return 0


def figure(a, dz, P, people, log, summary, ref, tgt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"V-path": "#d62728", "V-side": "#ff7f0e", "V-far": "#2ca02c"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})
    for k, p in enumerate(P):
        if k % 5 == 0:
            fp = np.vstack([p["fp"], p["fp"][:1]])
            ax.plot(fp[:, 0], fp[:, 1], color="#7f7f7f", lw=0.8, alpha=0.6)
    cs = np.array([p["c"] for p in P])
    ax.plot(cs[:, 0], cs[:, 1], color="#1f77b4", lw=2, label="forklift path (recorded)")
    for pid, xy in people.items():
        fa = summary[pid]["firstConfirmedAlert"]
        fw = summary[pid]["firstConfirmedWarning"]
        ax.scatter([xy[0]], [xy[1]], s=120, color=col[pid], edgecolor="k", zorder=5,
                   label=f"{pid}: " + (f"warn {fw['t']:.2f} s, " if fw else "no warning, ") + (f"alarm {fa['t']:.2f} s" if fa else "no alarm"))
    ax.scatter([dz.camera_xy[0]], [dz.camera_xy[1]], marker="^", color="k", s=60, label="camera (below)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("floor x (m)"); ax.set_ylabel("floor y (m)")
    ax.set_title("Floor map: real forklift path, virtual people")
    ax.legend(fontsize=8, loc="best")
    for pid, lg in log.items():
        t = [x["t"] for x in lg]
        cl = [x["clear"]["median"] if x["clear"]["median"] is not None else np.nan for x in lg]
        lo = [x["clear"]["p5"] if x["clear"]["p5"] is not None else np.nan for x in lg]
        hi = [x["clear"]["p95"] if x["clear"]["p95"] is not None else np.nan for x in lg]
        bx.plot(t, cl, color=col[pid], lw=2, label=f"{pid} floor clearance (median)")
        bx.fill_between(t, lo, hi, color=col[pid], alpha=0.15)
        al = [x["t"] for x in lg if x["confirmed"]]
        if al:
            bx.scatter(al, [0.05] * len(al), marker="|", s=200, color=col[pid])
        wl = [x["t"] for x in lg if x.get("warnConfirmed")]
        if wl:
            bx.scatter(wl, [0.30] * len(wl), marker="|", s=120, color=col[pid], alpha=0.45)
        if summary[pid]["contactT"] is not None:
            bx.axvline(summary[pid]["contactT"], color=col[pid], ls=":", lw=1)
    bx.axhline(0.8, color="k", ls="--", lw=1, label="0.8 m near-miss distance")
    bx.set_ylim(0, 6); bx.set_xlabel("clip time (s)"); bx.set_ylabel("floor clearance to forklift (m), 90% band")
    bx.set_title("Clearance over time; ticks: lower = alarm, upper (faint) = 3 s warning; dotted = contact")
    bx.grid(alpha=0.3); bx.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Alert control on real forklift motion ({a.name}, {ref['speed']:.1f} m/s at the reference): virtual people, not a real near miss", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out / f"{a.name}-zone-control.png", dpi=110)
    img = cv2.imread(str(a.out / f"{a.name}-zone-control.png"))
    cv2.imwrite(str(a.out / f"{a.name}-zone-control.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    raise SystemExit(main())
