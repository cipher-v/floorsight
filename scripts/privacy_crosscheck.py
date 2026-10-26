"""Privacy cross-check: was every person seen by a reference detector run pixelated in this run?

    python scripts/privacy_crosscheck.py --run JOB_DIR --reference REF_DIR --out crosscheck.json

For each clip, each frame and each reference person box, it takes that person's head region (same formula as the
perception stage) and measures how much of it this run's pixelated regions (heads + faces) cover. A head counts as
covered at >= 80 %. Reports per clip: reference heads, covered share, frames with an uncovered head (frame list).
Boxes only; no images are read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def head_region(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 - 0.12 * w, y1 - 0.06 * h, x2 + 0.12 * w, y1 + 0.30 * h)


def coverage(target, regions, shape=(1080, 1920)) -> float:
    x1, y1, x2, y2 = (int(round(v)) for v in target)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(shape[1], x2), min(shape[0], y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 1.0
    m = np.zeros((y2 - y1, x2 - x1), bool)
    for r in regions:
        a1, b1, a2, b2 = (int(round(v)) for v in r)
        a1, b1, a2, b2 = max(a1, x1), max(b1, y1), min(a2, x2), min(b2, y2)
        if a2 > a1 and b2 > b1:
            m[b1 - y1: b2 - y1, a1 - x1: a2 - x1] = True
    return float(m.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-cover", type=float, default=0.8)
    a = ap.parse_args()
    report = {"minCover": a.min_cover, "clips": {}}
    for ref in sorted(a.reference.glob("*-detections.jsonl")):
        clip = ref.name.replace("-detections.jsonl", "")
        mine = a.run / clip / f"{clip}-privacy-regions.jsonl"
        if not mine.exists():
            continue
        regions = {r["f"]: r["heads"] + r["faces"] for r in map(json.loads, mine.read_text().splitlines())}
        n = covered = 0
        bad_frames, worst = [], 1.0
        for r in map(json.loads, ref.read_text().splitlines()):
            if r["frame"] not in regions:
                continue
            frame_bad = False
            for p in r["persons"]:
                c = coverage(head_region(p["box"]), regions[r["frame"]])
                n += 1
                worst = min(worst, c)
                if c >= a.min_cover:
                    covered += 1
                else:
                    frame_bad = True
            if frame_bad:
                bad_frames.append(r["frame"])
        report["clips"][clip] = {"referenceHeads": n, "coveredShare": round(covered / n, 4) if n else None,
                                 "framesWithUncoveredHead": len(bad_frames), "frames": bad_frames[:200], "worstCover": round(worst, 3)}
    a.out.write_text(json.dumps(report, indent=1))
    print("CROSSCHECK", json.dumps({k: {kk: v[kk] for kk in ("referenceHeads", "coveredShare", "framesWithUncoveredHead")} for k, v in report["clips"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
