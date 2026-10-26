"""COOL vs stock OpenCV chart from a run_bench.sh summary (bench-summary.json). AWS ONLY (needs nothing but the JSON).

    python scripts/benchmark/bench_chart.py bench-summary.json OUT.jpg

Left: per-operation median latency on real 1080p frames, stock / COOL as a speed-up factor.
Right: the OpenCV-only pipeline (core chain, 4-process throughput) and the whole perception stage (detectors included),
each as frames per second for both builds. Sized for a 1920x1080 video frame; also writes OUT.json with the numbers.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import cv2


def main() -> int:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    d = json.loads(src.read_text())
    cool, stock = d["ops"]["cool"], d["ops"]["stock"]
    ops = {**stock["per_op_p50_ms"], **stock.get("extra_p50_ms", {})}
    cops = {**cool["per_op_p50_ms"], **cool.get("extra_p50_ms", {})}
    speed = sorted(((k, ops[k] / cops[k]) for k in ops if k in cops and cops[k] > 0), key=lambda x: x[1])
    perc = {"cool": [], "stock": []}
    for r in d.get("perception", []):
        leg = "cool" if r["run"].endswith("cool") else "stock"
        frames = (r.get("stages") or {}).get("frames") or r.get("frames")
        secs = r.get("detector_s") or r.get("seconds")  # per-frame work; model loading excluded when recorded
        if secs and frames:
            perc[leg].append(frames / secs)
    rows = {
        "OpenCV core chain, 1 stream": (1000 / stock["e2e_core_mean_ms"], 1000 / cool["e2e_core_mean_ms"]),
        "OpenCV chain, 4 processes": ((stock.get("throughput") or {}).get("aggregate_fps_server_chain"), (cool.get("throughput") or {}).get("aggregate_fps_server_chain")),
    }
    if perc["cool"] and perc["stock"]:
        rows["whole perception stage (detectors incl.)"] = (statistics.mean(perc["stock"]), statistics.mean(perc["cool"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 16})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.1, 1]})
    names = [k for k, _ in speed]; vals = [v for _, v in speed]
    a1.barh(names, vals, color=["#2a9d8f" if v >= 1 else "#e76f51" for v in vals])
    a1.axvline(1.0, color="k", lw=1)
    for i, v in enumerate(vals):
        a1.text(max(v, 1.0) + 0.06, i, f"{v:.2f}x", va="center", fontsize=14)
    a1.set_xlabel("speed-up of COOL over stock (median latency, same frames)")
    a1.set_title("Per OpenCV operation", fontsize=18)
    labels = [k for k in rows if rows[k][0] and rows[k][1]]
    ratio = [rows[k][1] / rows[k][0] for k in labels]
    a2.barh(labels, ratio, color=["#2a9d8f" if v >= 1 else "#e76f51" for v in ratio])
    a2.axvline(1.0, color="k", lw=1)
    for i, (k, v) in enumerate(zip(labels, ratio)):
        sv, cv = rows[k]
        fmt = (lambda x: f"{x:.0f}") if sv >= 10 else (lambda x: f"{x:.3f}")
        a2.text(max(v, 1.0) + 0.02, i, f"{v:.2f}x\n{fmt(cv)} vs {fmt(sv)} fps", va="center", fontsize=14)
    a2.set_yticks(range(len(labels))); a2.set_yticklabels([k.replace(", ", ",\n").replace(" (", "\n(") for k in labels])
    a2.set_xlim(0, max(1.8, max(ratio) + 0.6))
    a2.set_xlabel(f"speed-up of COOL {cool['opencv']} over stock {stock['opencv']}")
    a2.set_title("Pipelines", fontsize=18)
    fig.suptitle("COOL vs stock OpenCV on the same AWS Graviton4 c8g.xlarge, same code, same real frames", fontsize=19)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=120)
    cv2.imwrite(str(out), cv2.imread(str(png)), [cv2.IMWRITE_JPEG_QUALITY, 92])
    doc = {"schema": "floorsight.bench-chart.v1", "speedup_per_op": dict(speed), "pipelines_fps": rows, "perception_fps": perc,
           "cool": cool["opencv"], "stock": stock["opencv"]}
    out.with_suffix(".json").write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
