"""Demo-moment figure for one real clip: what Floorsight said, and when, around the closest pass. GCP ONLY.

Reads a perception run's per-frame records and result (no raw video: stills come from the privacy-filtered
overlay video) and writes:
  <name>-demo-timeline.jpg  floor clearance of one person to the forklift over time (median and 90% band),
                            the 0.8 m near-miss line, P(warning) and P(in stopping path), the confirmed warning
                            span, CHECKING frames, confirmed alarms, and the old pixel rule's alarm span;
  <name>-demo-strip.jpg     three overlay stills: warning confirmed, one second later, closest moment;
  <name>-demo.json          the numbers behind both.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--person", default=None, help="default: the person of the closest moving observation")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = json.loads((a.run / f"{a.name}-result.json").read_text())
    rows = [json.loads(x) for x in (a.run / f"{a.name}-detections.jsonl").read_text().splitlines() if x.strip()]
    pid = a.person or (res.get("floorClosestMovingObservation") or {}).get("person")
    if pid is None:
        raise SystemExit("no person given and no closest moving observation")
    T, med, lo, hi, pw, ps, chk, mov = [], [], [], [], [], [], [], []
    PT, PM, TR = [], [], []  # v7b: predicted closest pass in the next 3 s; the forklift's fitted turn rate
    for r in rows:
        pr = (r.get("floor") or {}).get("persons", {}).get(pid)
        if pr is None or pr["clearance_m"]["median"] is None:
            continue
        T.append(r["t"]); med.append(pr["clearance_m"]["median"]); lo.append(pr["clearance_m"]["p5"]); hi.append(pr["clearance_m"]["p95"])
        pw.append(pr.get("pWarn", 0.0)); ps.append(max(pr["pInZone"]["stop"], pr["pInZone"]["tail"])); mov.append(pr.get("vehicleMoving", False))
        if pr.get("alert"):
            chk.append((r["t"], pr["clearance_m"]["median"]))
        if pr.get("pass_m") and pr.get("vehicleMoving"):
            PT.append(r["t"]); PM.append(pr["pass_m"]["median"])
        veh = next((v for v in (r.get("floor") or {}).get("vehicles", []) if v["id"] == pr.get("vehicle")), None)
        if veh is not None and veh["speed_mps"][1] >= 0.3:
            TR.append((r["t"], abs(veh.get("turnRateFit_radps", 0.0)), veh.get("turnRateSE_radps", 9.0)))
    T, med, lo, hi, pw, ps = map(np.array, (T, med, lo, hi, pw, ps))
    warns = [w for w in (res.get("floorWarnings") or []) if w["person"] == pid]
    alarms = [e for e in (res.get("floorEvents") or []) if e["person"] == pid]
    pix = [e for e in (res.get("events") or []) if e["person"] == pid]
    closest = res.get("floorClosestMovingObservation")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.fill_between(T, lo, hi, color="#1f77b4", alpha=0.18, label="floor clearance, 90% band")
    ax.plot(T, med, color="#1f77b4", lw=2, label=f"floor clearance of {pid} to the forklift (median)")
    ax.axhline(0.8, color="k", ls="--", lw=1, label="0.8 m near-miss distance")
    if PT:
        ax.plot(PT, PM, color="#2ca02c", lw=1.6, ls="--", label="predicted closest pass in the next 3 s, if it keeps its course")
    turn_t = None
    if closest:
        near = [x for x in TR if closest["t"] - 2.0 <= x[0] <= closest["t"] + 0.2]
        turn_t = next((x[0] for x in near if x[1] >= 0.3 and x[2] <= 0.5 * x[1]), None)
        if turn_t is not None:
            ax.axvline(turn_t, color="#9467bd", ls=":", lw=1.5, label=f"forklift starts a measured turn ({turn_t:.2f} s)")
    for w in warns:
        ax.axvspan(w["confirmedT"], w["endT"], color="#ff9f1c", alpha=0.25, label="WARNING confirmed (3 s look-ahead)")
    for e in alarms:
        ax.axvspan(e["startT"], e["endT"], color="#d62728", alpha=0.25, label="ALARM confirmed")
    if chk:
        ax.scatter([c[0] for c in chk], [c[1] for c in chk], color="#d62728", zorder=6, s=40, label="CHECKING: alarm condition met, not 3 frames yet")
    for e in pix:
        ax.axvspan(e["startT"], e["endT"], ymin=0.93, ymax=1.0, color="#7f7f7f", alpha=0.6, label="old pixel rule: alarm")
    if closest:
        ax.annotate(f"closest {closest['clear']:.2f} m at {closest['t']:.2f} s", (closest["t"], closest["clear"]), xytext=(closest["t"] + 0.4, closest["clear"] + 1.2),
                    arrowprops={"arrowstyle": "->"}, fontsize=9)
    ax.set_ylim(0, 6.0)  # the decision region; farther clearances are clipped at the top
    ax.set_xlabel("clip time (s)"); ax.set_ylabel("metres on the floor")
    ax2 = ax.twinx()
    ax2.plot(T, pw, color="#ff9f1c", lw=1.5, label="P(within 0.8 m in the next 3 s)")
    ax2.plot(T, ps, color="#d62728", lw=1, ls=":", label="P(in stopping or tail-swing zone)")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("share of 64 tested variations")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(h1 + h2, l1 + l2):
        if l not in seen:
            seen.add(l); H.append(h); L.append(l)
    ax.legend(H, L, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_title(f"{a.name}: what Floorsight said, and when, for person {pid}")
    fig.tight_layout()
    a.out.mkdir(parents=True, exist_ok=True)
    png = a.out / f"{a.name}-demo-timeline.png"
    fig.savefig(png, dpi=110, bbox_inches="tight")
    cv2.imwrite(str(a.out / f"{a.name}-demo-timeline.jpg"), cv2.imread(str(png)), [cv2.IMWRITE_JPEG_QUALITY, 90])

    # stills from the privacy-filtered overlay
    fps = float(res["fps"])
    marks = []
    if closest:
        before = [(t_, m_) for t_, m_ in zip(PT, PM) if t_ <= closest["t"] - 1.0]
        if before:
            marks.append((f"1 s before: predicted pass {before[-1][1]:.1f} m", before[-1][0]))
        marks.append(("closest moment", closest["t"]))
    if warns:
        marks.append(("WARNING confirmed", warns[0]["confirmedT"]))
    marks.sort(key=lambda m: m[1])
    cap = cv2.VideoCapture(str(a.run / f"{a.name}-overlay.mp4"))
    tiles = []
    for text, tm in marks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(tm * fps)))
        ok, f = cap.read()
        if not ok:
            continue
        f = cv2.resize(f, (800, 450), interpolation=cv2.INTER_AREA)
        cv2.rectangle(f, (0, 0), (800, 40), (20, 20, 20), -1)
        cv2.putText(f, f"{text}  t={tm:.2f} s", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(f)
    if tiles:
        cv2.imwrite(str(a.out / f"{a.name}-demo-strip.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    pre = [(t_, m_) for t_, m_ in zip(PT, PM) if closest and t_ < (turn_t if turn_t is not None else closest["t"])]
    doc = {"schema": "floorsight.demo-moment.v1", "opencv": cv2.__version__, "clip": a.name, "person": pid, "closest": closest,
           "turnStartT": turn_t, "predictedPassBeforeTurn_m": ({"min": round(min(m for _, m in pre[-25:]), 3), "median": round(float(np.median([m for _, m in pre[-25:]])), 3),
                                                                "window_s": [pre[-25:][0][0], pre[-25:][-1][0]]} if pre else None),
           "warnings": warns, "alarms": alarms, "checkingFrames": [{"t": c[0], "clearance_m": c[1]} for c in chk],
           "pixelRuleEvents": [{k: e.get(k) for k in ("startT", "endT", "minPersonHeights")} for e in pix],
           "stills": [{"label": m[0], "t": m[1]} for m in marks]}
    (a.out / f"{a.name}-demo.json").write_text(json.dumps(doc, indent=1))
    print("DEMO", json.dumps({k: doc[k] for k in ("clip", "person", "closest", "checkingFrames", "turnStartT", "predictedPassBeforeTurn_m")}),
          "warnings", json.dumps([{k: w.get(k) for k in ("confirmedT", "endT", "ttcAtConfirm_s", "closestAfter", "warningLead_s")} for w in warns]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
