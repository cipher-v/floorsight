"""Which person-detector setting recovers the heads a run left uncovered? (numbers only; raw frames never leave)

    perc scripts/privacy_probe.py --crosscheck JOB/privacy-crosscheck.json --reference REF_DIR --clips CLIP_DIR \
        --models MODELS --hf HF_INDEX --out probe.json

For every frame the cross-check listed, it runs each candidate on the raw frame and measures, for every reference person
on that frame, how much of the head region the candidate's head regions would cover (>= 80 % = covered). Reports per
candidate: share of reference heads covered on those frames, boxes per frame, seconds per frame.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crosscheck", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-frames", type=int, default=25, help="per clip")
    a = ap.parse_args()
    od, pc = load("open_detectors"), load("privacy_crosscheck")
    hf = {k: v.get("path") for k, v in json.loads(a.hf.read_text()).items()}
    cands = {
        "rtdetr@0.20": lambda: od.RTDetrCoco(hf["PekingU/rtdetr_v2_r50vd"], conf=0.20),
        "rtdetr@0.10": lambda: od.RTDetrCoco(hf["PekingU/rtdetr_v2_r50vd"], conf=0.10),
        "rtdetr960@0.20": lambda: od.RTDetrCoco(hf["PekingU/rtdetr_v2_r50vd"], conf=0.20, size=960),
        "yolox_s-tiles3@0.10": lambda: od.YoloxDnn(a.models / "onnx" / "yolox_s.onnx", tiles=3, conf=0.10),
        "yolox_m-tiles3@0.10": lambda: od.YoloxDnn(a.models / "onnx" / "yolox_m.onnx", tiles=3, conf=0.10),
    }
    cc = json.loads(a.crosscheck.read_text())["clips"]
    frames = []
    for clip, v in cc.items():
        ref = {r["frame"]: r for r in map(json.loads, (a.reference / f"{clip}-detections.jsonl").read_text().splitlines())}
        cap = cv2.VideoCapture(str(a.clips / f"{clip}.mp4"))
        for f in v["frames"][: a.max_frames]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if ok and f in ref:
                frames.append((clip, f, img, [p["box"] for p in ref[f]["persons"]]))
        cap.release()
    report = {"frames": len(frames), "referenceHeads": sum(len(x[3]) for x in frames), "candidates": {}}
    dets = {}
    for name, mk in cands.items():
        det = mk()
        per, n_boxes, t_tot = [], 0, 0.0
        for clip, f, img, refs in frames:
            t0 = time.time(); d = det.detect(img); t_tot += time.time() - t0
            boxes = [b for b, _ in d[od.COCO_PERSON]]
            dets[(name, clip, f)] = boxes
            n_boxes += len(boxes)
            heads = [pc.head_region(b) for b in boxes]
            per += [pc.coverage(pc.head_region(r), heads) >= 0.8 for r in refs]
        report["candidates"][name] = {"covered": round(float(np.mean(per)), 3), "boxesPerFrame": round(n_boxes / len(frames), 1),
                                      "sPerFrame": round(t_tot / len(frames), 2)}
        print(name, report["candidates"][name], flush=True)
    for combo in (("rtdetr@0.20", "yolox_s-tiles3@0.10"), ("rtdetr@0.20", "yolox_m-tiles3@0.10")):
        per = []
        for clip, f, img, refs in frames:
            heads = [pc.head_region(b) for n in combo for b in dets[(n, clip, f)]]
            per += [pc.coverage(pc.head_region(r), heads) >= 0.8 for r in refs]
        report["candidates"][" + ".join(combo)] = {"covered": round(float(np.mean(per)), 3)}
        print(" + ".join(combo), report["candidates"][" + ".join(combo)], flush=True)
    a.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
