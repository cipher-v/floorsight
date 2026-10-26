"""One clip, two perception runs, one chart: the floor gap of one person to the forklift over time. AWS ONLY.

    python scripts/measured_twice.py --a RUN_DIR NAME LABEL --b RUN_DIR NAME LABEL --out OUT.jpg [--person-a P1 --person-b P1]

Reads only the per-frame records and result files (no video). For each run: median floor clearance with its 90 % band,
the predicted closest pass while the forklift moves, the confirmed warning spans and the closest moment. Sized for a
1920x1080 video frame. Also writes OUT.json with the numbers drawn.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def series(run: Path, name: str, pid: str | None) -> dict:
    res = json.loads((run / f"{name}-result.json").read_text())
    pid = pid or (res.get("floorClosestMovingObservation") or {}).get("person")
    T, med, lo, hi, PT, PM = [], [], [], [], [], []
    closest = None  # this person's smallest median gap while the forklift moves (the result file's closest may be someone else)
    for line in (run / f"{name}-detections.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pr = (r.get("floor") or {}).get("persons", {}).get(pid)
        if pr is None or pr["clearance_m"]["median"] is None:
            continue
        T.append(r["t"]); med.append(pr["clearance_m"]["median"]); lo.append(pr["clearance_m"]["p5"]); hi.append(pr["clearance_m"]["p95"])
        if pr.get("vehicleMoving") and (closest is None or pr["clearance_m"]["median"] < closest["clear"]):
            closest = {"clear": pr["clearance_m"]["median"], "frame": r.get("frame"), "t": r["t"], "person": pid,
                       "p5": pr["clearance_m"]["p5"], "p95": pr["clearance_m"]["p95"]}
        if pr.get("pass_m") and pr.get("vehicleMoving"):
            PT.append(r["t"]); PM.append(pr["pass_m"]["median"])
    warns = [{"confirmedT": w["confirmedT"], "endT": w["endT"]} for w in (res.get("floorWarnings") or []) if w["person"] == pid]
    return {"person": pid, "T": T, "med": med, "lo": lo, "hi": hi, "PT": PT, "PM": PM, "warnings": warns,
            "closest": closest, "clipClosest": res.get("floorClosestMovingObservation"), "opencv": (res.get("environment") or {}).get("opencv") or res.get("opencv")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs=3, required=True, metavar=("RUN", "NAME", "LABEL"))
    ap.add_argument("--b", nargs=3, required=True, metavar=("RUN", "NAME", "LABEL"))
    ap.add_argument("--person-a"); ap.add_argument("--person-b")
    ap.add_argument("--title", default="screen-002, measured twice: same clip, same floor calibration, two detector stacks")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    runs = [(series(Path(a.a[0]), a.a[1], a.person_a), a.a[2], "#8c6bb1"), (series(Path(a.b[0]), a.b[1], a.person_b), a.b[2], "#1f77b4")]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 17})
    fig, (top, ax) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [1, 5.5]})
    t_end = max(max(s["T"]) for s, _, _ in runs if s["T"]) + 0.3
    for k, (s, label, col) in enumerate(runs):
        ax.fill_between(s["T"], s["lo"], s["hi"], color=col, alpha=0.15)
        ax.plot(s["T"], s["med"], color=col, lw=3, label=f"{label}: floor gap to the forklift (median and 90 % band)")
        row = len(runs) - 1 - k  # warning strip: first run on top
        for w in s["warnings"]:
            top.barh(row, w["endT"] - w["confirmedT"], left=w["confirmedT"], height=0.6, color=col)
        if not s["warnings"]:
            top.text(0.1, row, "no warning", va="center", fontsize=14, color=col)
        c = s["closest"]
        if c:
            ax.scatter([c["t"]], [c["clear"]], color=col, s=110, zorder=6)
            ax.annotate(f"closest {c['clear']:.2f} m\nat {c['t']:.2f} s", (c["t"], c["clear"]),
                        xytext=(c["t"] - 1.9, c["clear"] - 0.2) if k else (c["t"] + 0.35, c["clear"] - 0.05), color=col, fontsize=17,
                        arrowprops={"arrowstyle": "->", "color": col, "lw": 1.5})
    ax.axhline(0.8, color="k", ls="--", lw=1.2, label="0.8 m near-miss distance")
    top.set_ylim(-0.6, len(runs) - 0.4); top.set_title("confirmed warnings", fontsize=15, loc="left")
    top.set_yticks([len(runs) - 1 - k for k in range(len(runs))], [lab.split(",")[0] for _, lab, _ in runs], fontsize=14)
    for k, tick in enumerate(top.get_yticklabels()):
        tick.set_color(runs[k][2])
    top.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, t_end); ax.set_ylim(0, 5.0)
    ax.set_xlabel("clip time (s)"); ax.set_ylabel("metres on the floor")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=14, loc="upper right", frameon=False)
    fig.suptitle(a.title, fontsize=20)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    png = a.out.with_suffix(".png")
    fig.savefig(png, dpi=120)
    cv2.imwrite(str(a.out), cv2.imread(str(png)), [cv2.IMWRITE_JPEG_QUALITY, 92])
    doc = {"schema": "floorsight.measured-twice.v1", "opencv": cv2.__version__,
           "runs": [{"label": label, "run": r[0], "clip": r[1], "person": s["person"], "closest": s["closest"], "clipClosest": s["clipClosest"], "warnings": s["warnings"],
                     "runOpencv": s["opencv"]} for (s, label, _), r in zip(runs, (a.a, a.b))]}
    a.out.with_suffix(".json").write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
