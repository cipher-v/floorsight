"""Real-footage perception for Floorsight: privacy first; pixel-level, or floor-calibrated with --calibration (v6). GCP ONLY.

Raw pixels never leave this process. Each frame is privacy-filtered (YuNet faces plus the head
region of every person detection, at a deliberately low confidence) before any overlay, video,
image or contact sheet is written. If detection or filtering fails, that frame is withheld as
black and the per-frame health record says so (fail closed).

Observations are image-plane only. "Person-heights" is a perspective-relative pixel ratio, not a
distance: there is no camera calibration, so no metres, speeds, time-to-contact or near-miss
claims. Folder/class labels of the publisher never enter this script; it sees a neutral name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("cvsafe", __import__("pathlib").Path(__file__).with_name("cvsafe.py"))
cvsafe = _ilu.module_from_spec(_spec); _spec.loader.exec_module(cvsafe)
from scipy.optimize import linear_sum_assignment

PRIVACY_PERSON_CONF = 0.10   # recall-oriented: any plausible person gets a blurred head
DISPLAY_PERSON_CONF = 0.30
TRACK_PERSON_CONF = 0.15     # candidates that may extend an existing person track
HEAD_HOLD_FRAMES = 12        # privacy: keep pixelating a lost person's last head region this many frames (more blur only; v8: 8 -> 12)
HEAD_GROW = 0.10             # privacy: head regions pixelated 10 % larger on each side (v8; more blur only)
# Probe development job perception-03 (clip-dev-004 frames 90-190): imgsz 960 beats 1280 for this large
# object and the generic synonym "industrial vehicle" keeps the turning forklift above 0.18 longer.
# No clip-specific prompt (e.g. the load) is used.
FORKLIFT_CONF = 0.18
FORKLIFT_PROMPTS = ["forklift", "forklift truck", "industrial vehicle"]
OPEN_VOCAB_SIZE = 960
LOAD_PROMPTS = ["stack of crates", "crate", "pallet", "box pallet"]  # generic carried loads (v6b, floor mode only)
LOAD_CONF = 0.25
LOAD_INSIDE_MIN = 0.6  # v8: share of a load box inside its forklift's box (+5 %) to count as carried
MOTION_MAX_FRAMES = 60       # a lost vehicle may be carried by image motion for at most ~2.4 s
MOTION_SIZE = (960, 540)
MOVING_FRACTION = 0.12       # share of a vehicle box that must be MOG2 foreground to count as moving
MOVING_MEMORY = 25           # a vehicle counts as moving for 1 s after its last observed motion
STATIC_SHOW_CONF = 0.35      # (v8) a vehicle never seen moving was shown once any frame reached this confidence
STATIC_SHOW_MEDIAN = 0.5     # v9: ... now its typical (median of the last 25 detections) confidence must reach 0.5: a press
                             # machine (median 0.33) and a stack of steel plates (0.35-0.39) were shown as "forklifts";
                             # real forklifts in these clips have medians 0.58-0.71
OPERATOR_HOLD_FRAMES = 25    # v10: a driver stays a driver 1 s after last seen inside the truck (getting down takes longer)
MOTION_WARMUP = 10          # first frames: background model still settling, motion ignored
CLOSE_REL = 2.0              # person-heights (uncalibrated)
VERY_CLOSE_REL = 1.0
EVENT_REL = 1.5
EVENT_MIN_FRAMES = 3
EVENT_END_FRAMES = 5
REVIEW_SIZE = (1280, 720)
LABEL_SCALE = 1.2            # v9: overlay labels 20 % larger (they were hard to read in the demo video)
REQUIRED_OPENCV_MAJOR = 5    # v5 (20 Oct): the competition requires OpenCV 5; refuse anything else


def environment() -> dict:
    """Exact library versions for the result receipt (v5)."""
    import platform
    import scipy
    info = {"opencv": cv2.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version()}
    info["machine"] = platform.machine()
    info["opencvBuildSha256"] = hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()
    for name in ("torch", "transformers", "ultralytics"):
        try:
            info[name] = __import__(name).__version__
        except Exception as error:  # recorded, not hidden
            info[name] = f"unavailable: {type(error).__name__}"
    return info

GREEN, AMBER, RED = (90, 200, 60), (0, 190, 255), (40, 40, 235)
ORANGE, CYAN, WHITE, GREY = (0, 140, 255), (230, 200, 40), (255, 255, 255), (170, 170, 170)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def background_samples(cap, total: int, count: int = 25, mode: str = "seek"):
    """Collect calibration samples; sequential decoding is experimental.

    Some variable-rate files return different pixels for seeking and sequential
    decoding. Keep seeking as the default until a clip passes the equality check.
    Long recordings always seek. Duplicate indices retain their original weight.
    """
    targets = np.linspace(0, max(0, total - 1), count).astype(int).tolist()
    if mode == "seek" or total > count * 64:
        samples = []
        for index in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                samples.append(frame)
        return samples
    wanted = set(targets)
    frames = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for index in range(max(targets) + 1):
        if not cap.grab():
            break
        if index in wanted:
            ok, frame = cap.retrieve()
            if ok:
                frames[index] = frame
    return [frames[index] for index in targets if index in frames]


def temporal_median(samples, mode: str = "compact"):
    """Exact uint8 median with contiguous sample rows and no float64 image.

    The usual 25 samples have an odd count, so the median is an existing value.
    Keep NumPy's averaging semantics if a damaged clip yields an even count.
    """
    if mode == "numpy" or len(samples) % 2 == 0:
        return np.median(np.stack(samples), axis=0).astype(np.uint8)
    pixels = np.stack(samples, axis=-1)
    middle = len(samples) // 2
    pixels.partition(middle, axis=-1)
    return pixels[..., middle].copy()


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


class IouTracker:
    """Hungarian IoU association. A track is shown only after `confirm` observed frames.

    v9 (predict=True, dist_max > 0; used for people): each track's box is first moved by its recent image velocity, and
    detections left unmatched by IoU get a second Hungarian pass on centre distance (in box heights, similar size). In the
    23 Oct run a third of the short person tracks were the same person re-numbered after a box shift, which also restarted
    their 3-frame warning count."""

    def __init__(self, prefix: str, iou_min=0.2, max_missed=12, confirm=3, predict=False, dist_max=0.0):
        self.prefix, self.iou_min, self.max_missed, self.confirm = prefix, iou_min, max_missed, confirm
        self.predict, self.dist_max = predict, dist_max
        self.tracks: list[dict] = []
        self.next_id = 1
        self.dist_matches = 0

    def _expected(self, t: dict) -> np.ndarray:
        b = np.asarray(t["box"], float)
        if not self.predict or "vel" not in t:
            return b
        dx, dy = t["vel"] * (t["missed"] + 1)
        return b + np.array([dx, dy, dx, dy])

    def update(self, boxes: np.ndarray) -> list[dict]:
        live = [t for t in self.tracks if t["missed"] <= self.max_missed]
        prev = np.array([self._expected(t) for t in live]) if live else np.zeros((0, 4))
        iou = iou_matrix(prev, boxes)
        matched_t, matched_d = set(), {}
        if iou.size:
            rows, cols = linear_sum_assignment(1 - iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_min:
                    matched_t.add(r); matched_d[c] = live[r]
        if self.dist_max > 0 and len(prev) and len(boxes):
            ur = [r for r in range(len(live)) if r not in matched_t]
            uc = [c for c in range(len(boxes)) if c not in matched_d]
            if ur and uc:
                cost = np.full((len(ur), len(uc)), 1e6)
                for i, r in enumerate(ur):
                    pb = prev[r]
                    ph = max(1.0, pb[3] - pb[1])
                    for j, c in enumerate(uc):
                        bb = np.asarray(boxes[c], float)
                        bh = max(1.0, bb[3] - bb[1])
                        if not 0.6 <= bh / ph <= 1.67:
                            continue
                        dc = np.hypot((pb[0] + pb[2] - bb[0] - bb[2]) / 2, (pb[1] + pb[3] - bb[1] - bb[3]) / 2) / max(ph, bh)
                        if dc <= self.dist_max:
                            cost[i, j] = dc
                rr, cc = linear_sum_assignment(cost)
                for i, j in zip(rr, cc):
                    if cost[i, j] < 1e6:
                        matched_t.add(ur[i]); matched_d[uc[j]] = live[ur[i]]
                        self.dist_matches += 1
        for r, t in enumerate(live):
            if r not in matched_t:
                t["missed"] += 1
        for c, box in enumerate(boxes):
            t = matched_d.get(c)
            if t is None:
                t = {"id": f"{self.prefix}{self.next_id}", "hits": 0, "missed": 0}
                self.next_id += 1
                live.append(t)
            elif self.predict:
                ob, nb = np.asarray(t["box"], float), np.asarray(box, float)
                step = np.array([(nb[0] + nb[2] - ob[0] - ob[2]) / 2, (nb[1] + nb[3] - ob[1] - ob[3]) / 2]) / (t["missed"] + 1)
                t["vel"] = 0.6 * t.get("vel", step) + 0.4 * step
            t.update(box=box, hits=t["hits"] + 1, missed=0, det=c, estimated=False)
        self.tracks = live
        return [t for t in live if t["missed"] == 0]


def mosaic(img: np.ndarray, box) -> bool:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return False
    roi = img[y1:y2, x1:x2]
    block = max(3, min(x2 - x1, y2 - y1) // 5)
    small = cv2.resize(roi, (max(1, (x2 - x1) // block), max(1, (y2 - y1) // block)), interpolation=cv2.INTER_AREA)
    img[y1:y2, x1:x2] = cv2.GaussianBlur(cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST), (5, 5), 0)
    return True


def soften(img: np.ndarray, box) -> bool:
    """Strong blur for vehicle-cab privacy zones (keeps the vehicle readable, hides the driver)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return False
    k = max(9, ((min(x2 - x1, y2 - y1) // 6) | 1))
    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (k, k), 0)
    return True


def cab_region(box) -> tuple:
    x1, y1, x2, y2 = box
    return (x1, y1, x2, y1 + 0.62 * (y2 - y1))


def nms(items, iou=0.5):
    """OpenCV NMS over (box, conf, ...) tuples; keeps the highest-confidence member of each overlap group."""
    if not items:
        return []
    rects = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b, *_ in items]
    keep = cv2.dnn.NMSBoxes(rects, [float(i[1]) for i in items], 0.0, iou)
    return [items[int(k)] for k in np.array(keep).reshape(-1)]


def grow_box(box, frac: float) -> tuple:
    x1, y1, x2, y2 = box
    dx, dy = frac * (x2 - x1), frac * (y2 - y1)
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)


def head_region(box) -> tuple:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 - 0.12 * w, y1 - 0.06 * h, x2 + 0.12 * w, y1 + 0.30 * h)


def withheld(shape, reason: str) -> np.ndarray:
    frame = np.zeros(shape, np.uint8)
    cv2.putText(frame, "FRAME WITHHELD - privacy filter failed", (60, shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, WHITE, 3, cv2.LINE_AA)
    cv2.putText(frame, reason[:90], (60, shape[0] // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREY, 2, cv2.LINE_AA)
    return frame


def segment_walkway(background: np.ndarray):
    """Green painted walkway from the temporal-median background (pixel geometry, no identity).

    v1 used one fixed HSV range and kept only the near, saturated part of the lane; the far part
    (bluer, hazier) was lost and a worker standing on it was called "outside" (GCP run
    development job perception-02). v2 takes the strict-green component as a seed, learns the lane's
    own hue/saturation, then grows into connected pixels of that paint.
    """
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    strict = cv2.inRange(hsv, (38, 70, 45), (95, 255, 255))
    strict = cv2.morphologyEx(strict, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(strict)
    if count <= 1:
        return None, 0.0, {}
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.max() < 0.004 * strict.size:
        return None, 0.0, {}
    top = int(areas.argmax()) + 1
    if stats[top, cv2.CC_STAT_TOP] + stats[top, cv2.CC_STAT_HEIGHT] < 0.5 * strict.shape[0]:
        return None, 0.0, {"rejected": "largest green region does not reach the floor in the lower half"}
    seed = labels == top
    hue, sat = float(np.median(hsv[..., 0][seed])), float(np.median(hsv[..., 1][seed]))
    diff = np.abs(hsv[..., 0].astype(np.int16) - int(hue))
    # Measured on clip-dev-001 (GCP job perception-05): lane hue 78-84 near, 93-99 far;
    # saturation ~125 near, ~50 mid-aisle, ~45 far. Grey floor saturation ~18.
    loose = ((np.minimum(diff, 180 - diff) <= 20) & (hsv[..., 1] >= 0.33 * sat) & (hsv[..., 2] >= 35)).astype(np.uint8) * 255
    loose = cv2.morphologyEx(loose, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    loose = cv2.morphologyEx(loose, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels = cv2.connectedComponents(loose)
    touching = set(np.unique(labels[seed & (loose > 0)]).tolist()) - {0}
    mask = np.isin(labels, list(touching)).astype(np.uint8) * 255
    return mask, float(mask.astype(bool).mean()), {"seedHue": hue, "seedSat": sat, "seedFraction": float(seed.mean())}


def forklift_footprint(box, polygon):
    """Ground-contact proxy: lowest quarter of the vehicle silhouette (mask if available, else box)."""
    x1, y1, x2, y2 = box
    parts = [q for q in (polygon or []) if q is not None and len(q) >= 3]
    if parts:
        pts = np.concatenate([np.asarray(q, np.float32) for q in parts])
        ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
        band = pts[pts[:, 1] >= ymax - 0.25 * (ymax - ymin)]
        if len(band) >= 2:
            return band
    xs = np.linspace(x1, x2, 40)
    return np.stack([xs, np.full_like(xs, y2)], 1).astype(np.float32)


def fg_centroid(fg, box, sx, sy, grow=0.35):
    """Centroid (full-res pixels) of MOG2 foreground near a box, or None without enough motion."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    rx1, ry1 = int(max(0, (x1 - grow * w) * sx)), int(max(0, (y1 - grow * h) * sy))
    rx2, ry2 = int(min(fg.shape[1], (x2 + grow * w) * sx)), int(min(fg.shape[0], (y2 + grow * h) * sy))
    ys, xs = np.nonzero(fg[ry1:ry2, rx1:rx2])
    if len(xs) < 0.12 * w * h * sx * sy:
        return None
    return np.array([(xs.mean() + rx1) / sx, (ys.mean() + ry1) / sy])


def merge_vehicle_parts(vehicles):
    """Union overlapping or touching vehicle detections (e.g. cab and carried load) into one unit.

    The load is part of the hazard footprint, so the unit's box and outline cover both. Two separate
    vehicles parked touching each other would also merge; that limitation is recorded in the result.
    """
    merged = []
    for b, c, poly, kind in sorted(vehicles, key=lambda v: -v[1]):
        b = np.asarray(b, float)
        for m in merged:
            if iou_matrix(m["box"][None], b[None])[0, 0] > 0.02 or contained(b, m["box"]) > 0.3 or contained(m["box"], b) > 0.3:
                m["box"] = np.array([min(m["box"][0], b[0]), min(m["box"][1], b[1]), max(m["box"][2], b[2]), max(m["box"][3], b[3])])
                m["parts"] += 1
                if poly is not None:
                    m["polys"].append(poly)
                break
        else:
            merged.append({"box": b, "conf": float(c), "polys": [] if poly is None else [poly], "kind": kind, "parts": 1})
    return [(m["box"], m["conf"], m["polys"] or None, m["kind"]) for m in merged]


def motion_fraction(fg, box, sx, sy) -> float:
    x1, y1, x2, y2 = box
    region = fg[int(max(0, y1 * sy)):int(max(0, y2 * sy)), int(max(0, x1 * sx)):int(max(0, x2 * sx))]
    return float(region.mean()) if region.size else 0.0


def track_vehicles(tracker, detections, fg, sx, sy, frame_index=0) -> list[dict]:
    """Detected vehicles, plus confirmed vehicles the detector lost but image motion still carries.

    A lost vehicle keeps its last box size and follows the foreground centroid with the offset
    measured while it was detected. It is shown as ESTIMATED and dropped after MOTION_MAX_FRAMES or
    as soon as there is no motion evidence (no stationary ghosts).
    """
    detections = merge_vehicle_parts(detections)
    boxes = np.array([d[0] for d in detections]) if detections else np.zeros((0, 4))
    tracker.update(boxes)
    shown = []
    warm = frame_index >= MOTION_WARMUP
    for tr in tracker.tracks:
        if tr["hits"] < tracker.confirm:
            continue
        if tr["missed"] == 0:
            b, c, poly, kind = detections[tr["det"]]
            centre = fg_centroid(fg, b, sx, sy)
            if centre is not None:
                tr["offset"] = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) - centre
            tr["kind"], tr["bestConf"] = kind, max(tr.get("bestConf", 0.0), float(c))
            tr["confs"] = (tr.get("confs", []) + [float(c)])[-25:]
            if warm and motion_fraction(fg, b, sx, sy) >= MOVING_FRACTION:
                tr["lastMoving"], tr["everMoved"] = frame_index, True
            if not tr.get("everMoved") and (tr["bestConf"] < STATIC_SHOW_CONF or float(np.median(tr["confs"])) < STATIC_SHOW_MEDIAN):
                continue  # object that has never moved and is only sometimes called a forklift (press machine, plate stack): not shown
            moving = frame_index - tr.get("lastMoving", -10**6) <= MOVING_MEMORY
            shown.append({"id": tr["id"], "box": b, "poly": poly, "kind": kind, "conf": float(c), "estimated": False, "moving": moving})
        elif "offset" in tr:
            if any(iou_matrix(np.asarray(tr["box"])[None], d[None])[0, 0] > 0.1 or contained(tr["box"], d) > 0.3 for d in boxes):
                tr["missed"] = MOTION_MAX_FRAMES + 1  # a fresh detection covers it: retire, never show a duplicate
                continue
            centre = fg_centroid(fg, tr["box"], sx, sy)
            if centre is None:
                continue
            b = tr["box"]
            w, h = b[2] - b[0], b[3] - b[1]
            cx, cy = centre + tr["offset"]
            tr["box"] = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            tr["lastMoving"], tr["everMoved"] = frame_index, True  # carried by motion evidence only
            shown.append({"id": tr["id"], "box": tr["box"], "poly": None, "kind": tr.get("kind", "forklift"), "conf": None,
                          "estimated": True, "moving": True, "framesSinceDetection": int(tr["missed"])})
    return shown


def dashed_rect(img, box, color, thickness=3, dash=18):
    x1, y1, x2, y2 = (int(v) for v in box)
    for xa in range(x1, x2, 2 * dash):
        cv2.line(img, (xa, y1), (min(xa + dash, x2), y1), color, thickness)
        cv2.line(img, (xa, y2), (min(xa + dash, x2), y2), color, thickness)
    for ya in range(y1, y2, 2 * dash):
        cv2.line(img, (x1, ya), (x1, min(ya + dash, y2)), color, thickness)
        cv2.line(img, (x2, ya), (x2, min(ya + dash, y2)), color, thickness)


def contained(inner, outer) -> float:
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))


def label(img, text, org, color, scale=0.9, top=100):
    scale *= LABEL_SCALE
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = int(min(max(0, org[0]), img.shape[1] - tw - 12))  # keep labels inside the frame
    y = int(max(th + top, org[1]))  # and below the header bar
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y + base - 2), (20, 20, 20), -1)
    cv2.putText(img, text, (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


class Detectors:
    """LEGACY (runs before 23 Oct 2026): Ultralytics YOLO11m + YOLOE-11m-seg. AGPL-3.0, not a Floorsight dependency;
    only used with --detector ultralytics-legacy after installing ultralytics yourself. The default is OpenDetectors."""

    def __init__(self, models: Path, loads: bool = False):
        from ultralytics import YOLO
        self.notes = []
        self.person = YOLO(str(models / "yolo11m.pt"))
        self.open_vocab, self.open_names = None, []
        try:
            from ultralytics import YOLOE
            model = YOLOE(str(models / "yoloe-11m-seg.pt"))
            names = ["person"] + FORKLIFT_PROMPTS
            try:
                model.set_classes(names, model.get_text_pe(names))
            except TypeError:
                model.set_classes(names)
            self.open_vocab, self.open_names = model, names
            self.notes.append(f"YOLOE-11m-seg open-vocabulary prompts: {names} (instance masks); YOLO11m COCO person.")
        except Exception as error:  # Recorded, not hidden: fallback is a COCO vehicle proxy.
            self.notes.append(f"YOLOE unavailable ({type(error).__name__}: {error}); vehicle proxy = COCO truck/car.")
        self.faces = cv2.FaceDetectorYN.create(str(models / "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.6, 0.3, 5000)
        self.load_model = None
        if loads:  # separate instance so the forklift prompts and their scores are unchanged
            from ultralytics import YOLOE
            lm = YOLOE(str(models / "yoloe-11m-seg.pt"))
            try:
                lm.set_classes(LOAD_PROMPTS, lm.get_text_pe(LOAD_PROMPTS))
            except TypeError:
                lm.set_classes(LOAD_PROMPTS)
            self.load_model = lm
            self.notes.append(f"YOLOE-11m-seg carried-load prompts: {LOAD_PROMPTS} (conf >= {LOAD_CONF}); attached to the forklift they touch.")

    def run(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        r = self.person.predict(frame, imgsz=1280, conf=PRIVACY_PERSON_CONF, classes=[0, 2, 7], verbose=False)[0]
        boxes, conf, cls = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
        persons = [(b, c) for b, c, k in zip(boxes, conf, cls) if k == 0]
        vehicles = []
        if self.open_vocab is not None:
            o = self.open_vocab.predict(frame, imgsz=OPEN_VOCAB_SIZE, conf=PRIVACY_PERSON_CONF, verbose=False)[0]
            polys = o.masks.xy if o.masks is not None else [None] * len(o.boxes)
            for b, c, k, p in zip(o.boxes.xyxy.cpu().numpy(), o.boxes.conf.cpu().numpy(), o.boxes.cls.cpu().numpy().astype(int), polys):
                if self.open_names[k] in FORKLIFT_PROMPTS and c >= FORKLIFT_CONF:
                    vehicles.append((b, c, p, "forklift"))
                elif self.open_names[k] == "person":
                    persons.append((b, c))  # Second person source, used for privacy recall only.
        else:
            vehicles = [(b, c, None, "vehicle (proxy)") for b, c, k in zip(boxes, conf, cls) if k in (2, 7) and c >= FORKLIFT_CONF]
        self.faces.setInputSize((w, h))
        _, faces = self.faces.detect(frame)
        faces = [] if faces is None else [(f[0], f[1], f[0] + f[2], f[1] + f[3]) for f in faces]
        merged = nms([(b, c) for b, c in persons if c >= TRACK_PERSON_CONF])
        loads = []
        if self.load_model is not None:
            l_ = self.load_model.predict(frame, imgsz=OPEN_VOCAB_SIZE, conf=LOAD_CONF, verbose=False)[0]
            lp = l_.masks.xy if l_.masks is not None else [None] * len(l_.boxes)
            loads = [(b, c, q) for b, c, q in zip(l_.boxes.xyxy.cpu().numpy(), l_.boxes.conf.cpu().numpy(), lp) if q is not None and len(q) >= 3]
        return {"persons": persons, "vehicles": nms(vehicles, 0.4), "faces": faces, "track_persons": merged, "loads": loads}


class CachedDetections:
    """Replays the detector output a previous run saved with --save-detections: the same person, vehicle, face and load
    boxes and outlines per frame, without loading any model. Used to re-run tracking, floor and zone logic in minutes."""

    def __init__(self, path: Path):
        saved = pickle.loads(Path(path).read_bytes())
        self.frames = saved["frames"]
        self.notes = list(saved.get("notes", [])) + [f"Detections replayed from {Path(path).name} (saved by {saved.get('run', '?')})"]
        self.timing = {**saved.get("timing", {}), "replayedFrom": Path(path).name}
        self.clip_sha256 = saved.get("clipSha256")

    def run_at(self, index: int) -> dict:
        return self.frames[index]  # a frame the saved run withheld is missing: KeyError -> this frame is withheld too


class Encoder:
    def __init__(self, path: Path, fps: float, size=REVIEW_SIZE, crf: int = 22):
        self.size = tuple(size)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.size[0]}x{self.size[1]}",
             "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def write(self, frame):
        f_ = frame if (frame.shape[1], frame.shape[0]) == self.size else cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        self.proc.stdin.write(f_.tobytes())

    def close(self) -> int:
        self.proc.stdin.close()
        return self.proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force-privacy-failure-frame", type=int, action="append", default=[])
    parser.add_argument("--max-frames", type=int, default=0, help="Stop early (fail-closed tests only).")
    parser.add_argument("--calibration", type=Path, default=None, help="v6: floor calibration JSON -> floor-clearance alerts and danger zones")
    parser.add_argument("--danger-draws", type=int, default=64)
    parser.add_argument("--detector", choices=["open", "ultralytics-legacy"], default="open",
                        help="open (default): Apache-2.0 stack in open_detectors.py; ultralytics-legacy: the pre-23-Oct AGPL models")
    parser.add_argument("--vocab", choices=["omdet", "gdino", "owlv2"], default="omdet", help="open-vocabulary forklift/load detector")
    parser.add_argument("--segmenter", choices=["sam-vit-base", "slimsam77"], default="sam-vit-base")
    parser.add_argument("--person-model", default="yolox_l", help="YOLOX ONNX name under <models>/onnx")
    parser.add_argument("--person-tiles", type=int, default=3)
    parser.add_argument("--person-conf", type=float, default=None, help="score floor for the COCO person model (default PRIVACY_PERSON_CONF)")
    parser.add_argument("--vehicle-source", choices=["vocab", "coco", "confirmed"], default="vocab",
                        help="forklift boxes: open-vocabulary model, COCO truck/car, or COCO truck/car confirmed by the open-vocabulary model")
    parser.add_argument("--vehicle-conf", type=float, default=None, help="score floor for COCO truck/car boxes (default FORKLIFT_CONF)")
    parser.add_argument("--vocab-every", type=int, default=1, help="run the open-vocabulary model every N frames")
    parser.add_argument("--track-person-conf", type=float, default=None, help="score floor for person tracks (default TRACK_PERSON_CONF)")
    parser.add_argument("--privacy-extra", default=None, help="YOLOX ONNX name (e.g. yolox_s) used as a second, blur-only person source via cv2.dnn")
    parser.add_argument("--detector-workers", type=int, choices=[1, 2], default=1, help="independent detector tasks per frame; privacy checks still run on every frame")
    parser.add_argument("--torch-layout", choices=["contiguous", "channels-last"], default="contiguous", help="model/input memory layout; same weights and precision")
    parser.add_argument("--background-sampling", choices=["auto", "seek"], default="seek", help="seek preserves prior decoding; auto is experimental and requires a per-clip sample equality check")
    parser.add_argument("--background-median", choices=["compact", "numpy"], default="compact", help="exact uint8 temporal median; numpy selects the previous implementation for comparison")
    parser.add_argument("--cpu-threads", type=int, default=None, help="threads per model/OpenCV operation; benchmark 1 and 2 on a 4-core Graviton before increasing")
    parser.add_argument("--hf-index", type=Path, default=None, help="hf-models.json (local snapshot paths)")
    parser.add_argument("--save-detections", type=Path, default=None, help="write every frame's detector output (boxes and outlines, no pixels) to this pickle")
    parser.add_argument("--load-detections", type=Path, default=None, help="replay a pickle written by --save-detections instead of running the detectors")
    parser.add_argument("--no-smoothing", action="store_true", help="v8 behaviour: no Kalman filtering, raw outlines, flickering turn and seated calls")
    args = parser.parse_args()
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            parser.error("--cpu-threads must be positive")
        cv2.setNumThreads(args.cpu_threads)
        if args.load_detections is None and args.detector == "open":
            import torch
            torch.set_num_threads(args.cpu_threads)
    if int(cv2.__version__.split(".")[0]) != REQUIRED_OPENCV_MAJOR:
        raise SystemExit(f"OpenCV {REQUIRED_OPENCV_MAJOR}.x required, found {cv2.__version__}")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    cap = cv2.VideoCapture(str(args.clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.load_detections is not None:
        det = CachedDetections(args.load_detections)
        if det.clip_sha256 and det.clip_sha256 != sha256(args.clip):
            raise SystemExit("saved detections belong to a different clip")
    elif args.detector == "open":
        import importlib.util
        spec = importlib.util.spec_from_file_location("open_detectors", Path(__file__).with_name("open_detectors.py"))
        odm = importlib.util.module_from_spec(spec); spec.loader.exec_module(odm)
        consts = {k: globals()[k] for k in ("PRIVACY_PERSON_CONF", "TRACK_PERSON_CONF", "FORKLIFT_CONF", "FORKLIFT_PROMPTS", "LOAD_PROMPTS", "LOAD_CONF")}
        if args.person_conf is not None:
            consts["PERSON_MODEL_CONF"] = args.person_conf
        if args.vehicle_conf is not None:
            consts["VEHICLE_MODEL_CONF"] = args.vehicle_conf
        if args.track_person_conf is not None:
            consts["TRACK_PERSON_CONF"] = args.track_person_conf
        det = odm.OpenDetectors(args.models, loads=args.calibration is not None, consts=consts, person_onnx=args.person_model,
                                tiles=args.person_tiles, vocab=args.vocab, segmenter=args.segmenter, hf_index=args.hf_index,
                                vehicle_source=args.vehicle_source, vocab_every=args.vocab_every, privacy_extra=args.privacy_extra,
                                workers=args.detector_workers, torch_layout=args.torch_layout)
    else:
        det = Detectors(args.models, loads=args.calibration is not None)
    det_s = 0.0
    saved_detections = {} if args.save_detections is not None else None
    duplicate_boxes = 0
    operator_seen, operator_held = {}, 0  # person id -> (vehicle id, frame) last seen inside that vehicle; frames held (v10)

    # Static background (temporal median of raw frames) only feeds walkway geometry; it is never written.
    background_start = time.perf_counter()
    samples = background_samples(cap, total, mode=args.background_sampling)
    background_sample_s = time.perf_counter() - background_start
    median_start = time.perf_counter()
    background = temporal_median(samples, mode=args.background_median)
    background_median_s = time.perf_counter() - median_start
    walkway, walkway_fraction, walkway_stats = segment_walkway(background)
    del background
    motion = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=24, detectShadows=True)
    for frame in samples:
        motion.apply(cv2.resize(frame, MOTION_SIZE, interpolation=cv2.INTER_AREA))
    del samples, frame
    walkway_top = int(np.nonzero(walkway.any(axis=1))[0].min()) if walkway is not None else None
    walkway_check = cv2.dilate(walkway, np.ones((15, 15), np.uint8)) if walkway is not None else None
    walkway_edges = cv2.findContours(walkway, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if walkway is not None else []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    smoothing = not args.no_smoothing
    people = IouTracker("P", max_missed=25, predict=smoothing, dist_max=0.6 if smoothing else 0.0)
    vehicles = IouTracker("F", iou_min=0.15, max_missed=MOTION_MAX_FRAMES)
    sx = MOTION_SIZE[0] / cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    sy = MOTION_SIZE[1] / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # v9: the overlay is written at the clip's own size (was 1280x720, which blurred the labels once the video scaled it back up)
    encoder = Encoder(args.output / f"{args.name}-overlay.mp4", fps, size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), crf=21)
    clean = Encoder(args.output / f"{args.name}-filtered.mp4", fps,  # v7d: filtered frames, no overlays, full size (app + agent crops)
                    size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), crf=18)
    dz = dzm = None
    if args.calibration is not None:  # v6: floor-based alerts replace the pixel ratio (which is still logged)
        import importlib.util
        spec = importlib.util.spec_from_file_location("danger_zone", Path(__file__).with_name("danger_zone.py"))
        dzm = importlib.util.module_from_spec(spec); spec.loader.exec_module(dzm)
        dzm.SMOOTHING = smoothing
        dz = dzm.DangerZone(args.calibration, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), fps, draws=args.danger_draws)
    f_open, f_events, f_streak, fbest = {}, [], {}, {"clear": float("inf")}
    tbest = {"score": 0.0}  # clearest turning moment (v6d)
    w_open, w_events, w_streak, wkey = {}, [], {}, {}  # v7 warnings and the first confirmed warning frame
    sheet_at = set(np.linspace(0, max(0, total - 1), 16).astype(int).tolist())
    sheet, records, health = [], [], []
    open_events, events, streak = {}, [], {}
    best = {"rel": float("inf")}
    largest = {"area": 0}
    privacy_crop = {"area": 0}
    track_crops: dict[str, tuple] = {}
    audit_frames: list[tuple] = []
    blurred_regions: list[dict] = []  # boxes only: what was pixelated on each frame (privacy cross-check)
    index = 0
    while not args.max_frames or index < args.max_frames:
        ok, raw = cap.read()
        if not ok:
            break
        t = index / fps
        try:
            if index in args.force_privacy_failure_frame:
                raise RuntimeError("forced privacy failure (fail-closed self-test)")
            _t_det = time.perf_counter()
            found = det.run_at(index) if isinstance(det, CachedDetections) else det.run(raw)
            det_s += time.perf_counter() - _t_det
            if saved_detections is not None:
                saved_detections[index] = found
            fg = motion.apply(cv2.resize(raw, MOTION_SIZE, interpolation=cv2.INTER_AREA))
            fg = cv2.morphologyEx((fg == 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            shown_vehicles = track_vehicles(vehicles, found["vehicles"], fg, sx, sy, index)
            filtered = raw.copy()
            cabs = sum(soften(filtered, cab_region(v["box"])) for v in shown_vehicles)  # driver privacy
            heads = sum(mosaic(filtered, grow_box(head_region(b), HEAD_GROW)) for b, _ in found["persons"])
            held = [grow_box(head_region(t["box"]), 0.15) for t in people.tracks if 1 <= t["missed"] <= HEAD_HOLD_FRAMES and t["hits"] >= 2]
            heads += sum(mosaic(filtered, hb) for hb in held)
            faces = sum(mosaic(filtered, (x1 - 4, y1 - 4, x2 + 4, y2 + 4)) for x1, y1, x2, y2 in found["faces"])
            blurred_regions.append({"f": index, "heads": [[round(float(v), 1) for v in grow_box(head_region(b), HEAD_GROW)] for b, _ in found["persons"]]
                                    + [[round(float(v), 1) for v in hb] for hb in held],
                                    "faces": [[round(float(v), 1) for v in (x1 - 4, y1 - 4, x2 + 4, y2 + 4)] for x1, y1, x2, y2 in found["faces"]]})
            status = {"t": round(t, 3), "frame": index, "status": "ok", "headsBlurred": int(heads), "facesBlurred": int(faces),
                      "vehicleCabsBlurred": int(cabs), "personCandidates": len(found["persons"])}
        except Exception as error:
            filtered = withheld(raw.shape, f"{type(error).__name__}: {error}")
            status = {"t": round(t, 3), "frame": index, "status": "withheld", "error": f"{type(error).__name__}: {error}"}
            found = None
        del raw  # Nothing below can reach unfiltered pixels.
        clean.write(filtered)
        health.append(status)
        if status.get("personCandidates") == 0:
            audit_frames.append((index, cv2.resize(filtered, (1600, 900), interpolation=cv2.INTER_AREA)))
            if len(audit_frames) > 12:
                audit_frames.pop(len(audit_frames) // 2)  # keep a spread, bounded memory
        if index in sheet_at:
            sheet.append(cv2.resize(filtered, (480, 270), interpolation=cv2.INTER_AREA))
        canvas = filtered.copy()
        mark_best = mark_large = False
        record = {"t": round(t, 3), "frame": index, "privacy": status["status"], "persons": [], "vehicles": []}
        if found is not None:
            if walkway is not None:
                tint = canvas.copy(); tint[walkway > 0] = CYAN
                canvas = cv2.addWeighted(tint, 0.18, canvas, 0.82, 0)
                cv2.drawContours(canvas, walkway_edges, -1, CYAN, 3, cv2.LINE_AA)
            for v in shown_vehicles:
                b, poly = v["box"], v["poly"]
                parts = [np.asarray(q, np.int32) for q in (poly or []) if len(q) >= 3]
                if parts:
                    layer = canvas.copy(); cvsafe.fill_poly(layer, parts, ORANGE)
                    canvas = cv2.addWeighted(layer, 0.30, canvas, 0.70, 0)
                if v["estimated"]:
                    dashed_rect(canvas, b, ORANGE)
                    label(canvas, f"{v['kind']} {v['id']} - estimated from motion", (b[0], b[1] - 6), ORANGE)
                else:
                    cv2.rectangle(canvas, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), ORANGE, 3)
                    label(canvas, f"{v['kind']} {v['id']} {v['conf']:.2f}" + ("" if v["moving"] or dz is not None else " parked"), (b[0], b[1] - 6), ORANGE)
                v["foot"] = forklift_footprint(b, poly)
                record["vehicles"].append({"id": v["id"], "kind": v["kind"], "box": [round(float(x), 1) for x in b], "estimated": v["estimated"], "moving": v["moving"],
                                           "conf": None if v["conf"] is None else round(v["conf"], 3), "mask": bool(poly), "parts": len(poly) if poly else 0,
                                           # v5: full mask outline (image pixels) for the floor-distance re-test
                                           "polygons": [np.asarray(q, np.float32).round(1).tolist() for q in parts]})
            pboxes = np.array([p[0] for p in found["track_persons"]]) if found["track_persons"] else np.zeros((0, 4))
            pdraw = []
            for tr in people.update(pboxes):
                b, c = found["track_persons"][tr["det"]]
                tr["best"] = max(tr.get("best", 0.0), float(c))
                if tr["hits"] < people.confirm or tr["best"] < DISPLAY_PERSON_CONF:
                    continue
                foot = np.array([(b[0] + b[2]) / 2, b[3]], np.float32)
                height = max(1.0, b[3] - b[1])
                entry = {"id": tr["id"], "box": [round(float(v), 1) for v in b], "conf": round(float(c), 3), "foot": [round(float(v), 1) for v in foot]}
                operator = next((v["id"] for v in shown_vehicles if contained(b, v["box"]) > 0.6), None)
                if smoothing:
                    # v10: a driver stays a driver for OPERATOR_HOLD_FRAMES after last seen inside the truck, while that
                    # truck is still shown: one frame's shrunken forklift box (clip-dev-004, 6.88 s) made the driver a
                    # pedestrian 0.90 m from their own truck
                    if operator is not None:
                        operator_seen[tr["id"]] = (operator, index)
                    elif tr["id"] in operator_seen:
                        held_v, held_f = operator_seen[tr["id"]]
                        if index - held_f <= OPERATOR_HOLD_FRAMES and any(v["id"] == held_v for v in shown_vehicles):
                            operator = held_v
                            operator_held += 1
                rel, nearest, near_pt, near_est, near_moving = None, None, None, False, False
                if operator is None:
                    for v in shown_vehicles:
                        d = np.linalg.norm(v["foot"] - foot, axis=1)
                        k = int(d.argmin())
                        if rel is None or d[k] / height < rel:
                            rel, nearest, near_pt, near_est, near_moving = float(d[k] / height), v["id"], v["foot"][k], v["estimated"], v["moving"]
                if walkway_check is not None:
                    fx, fy = int(np.clip(foot[0], 0, walkway_check.shape[1] - 1)), int(np.clip(foot[1], 0, walkway_check.shape[0] - 1))
                    if fy < walkway_top - 0.02 * walkway_check.shape[0]:
                        entry["walkway"] = "walkway not mapped here"  # beyond the segmented paint: unknown, not "outside"
                    else:
                        entry["walkway"] = "on marked walkway" if walkway_check[fy, fx] else "outside marked walkway"
                color = GREEN
                if operator is not None:
                    color, entry["role"] = GREY, f"operator on {operator}"
                elif rel is not None:
                    entry.update(nearestVehicle=nearest, proximityPersonHeights=round(rel, 2), vehicleEstimated=near_est, vehicleMoving=near_moving)
                    color = (RED if rel < VERY_CLOSE_REL else AMBER if rel < CLOSE_REL else GREEN) if near_moving else (AMBER if rel < CLOSE_REL else GREEN)
                elif "walkway" in entry and entry["walkway"].startswith("outside"):
                    color = AMBER
                pdraw.append({"id": tr["id"], "box": b, "foot": foot, "color": color, "operator": operator, "rel": rel, "near_pt": near_pt,
                              "near_est": near_est, "near_moving": near_moving, "entry": entry})
                record["persons"].append(entry)
                if rel is not None and rel < best["rel"]:
                    best = {"rel": rel, "frame": index, "t": t, "person": tr["id"], "vehicle": nearest}
                    mark_best = True
                area = (b[2] - b[0]) * (b[3] - b[1])
                if area > track_crops.get(tr["id"], (0,))[0]:
                    x1, y1, x2, y2 = (int(v) for v in b); py, px = int(0.15 * (y2 - y1)), int(0.3 * (x2 - x1))
                    tc = filtered[max(0, y1 - py):y2 + py, max(0, x1 - px):x2 + px]
                    if tc.size:
                        track_crops[tr["id"]] = (area, tc.copy(), index)
                if area > largest["area"]:
                    largest, mark_large = {"area": float(area), "frame": index, "t": t, "person": tr["id"]}, True
                if area > privacy_crop["area"]:
                    x1, y1, x2, y2 = b; pad_w, pad_h = 0.6 * (x2 - x1), 0.25 * (y2 - y1)
                    crop = filtered[int(max(0, y1 - pad_h)):int(min(filtered.shape[0], y2 + pad_h)), int(max(0, x1 - pad_w)):int(min(filtered.shape[1], x2 + pad_w))]
                    if crop.size:
                        privacy_crop = {"area": area, "image": crop.copy(), "frame": index, "person": tr["id"]}
                # Temporal confirmation: an alert needs EVENT_MIN_FRAMES consecutive close observations.
                key = tr["id"]
                close = rel is not None and rel < EVENT_REL and near_moving  # parked vehicles do not raise alerts
                s = streak.setdefault(key, {"close": 0, "far": 0})
                s["close"], s["far"] = (s["close"] + 1, 0) if close else (0, s["far"] + 1)
                if close and s["close"] >= EVENT_MIN_FRAMES and key not in open_events:
                    open_events[key] = {"type": "person_close_to_vehicle_uncalibrated", "person": key, "vehicle": nearest,
                                        "startT": round(t - (EVENT_MIN_FRAMES - 1) / fps, 3), "minPersonHeights": rel, "minAtFrame": index,
                                        "vehicleEstimatedAtMin": near_est}
                if key in open_events:
                    ev = open_events[key]
                    if close and rel < ev["minPersonHeights"]:
                        ev.update(minPersonHeights=rel, minAtFrame=index, vehicle=nearest, vehicleEstimatedAtMin=near_est)
                    if s["far"] >= EVENT_END_FRAMES:
                        ev["endT"] = round(t, 3); events.append(open_events.pop(key))
            if smoothing and len(pdraw) > 1:
                # v9: one person seen as two overlapping boxes (screen-002: a seated operator as upper body and as a whole) must
                # count once: one floor point, one warning, one alert. The box on the longer track stays; the twin is drawn thin.
                hits_ = {t_["id"]: t_["hits"] for t_ in people.tracks}
                primary_ = []
                for d_ in sorted(pdraw, key=lambda q: (-hits_.get(q["id"], 0), q["id"])):
                    if d_["operator"] is None:
                        h_ = max(1.0, d_["box"][3] - d_["box"][1])
                        twin = next((k_ for k_ in primary_ if k_["operator"] is None
                                     and max(contained(d_["box"], k_["box"]), contained(k_["box"], d_["box"])) >= 0.6
                                     and np.linalg.norm(np.asarray(k_["foot"], float) - np.asarray(d_["foot"], float)) <= 0.5 * max(h_, k_["box"][3] - k_["box"][1])), None)
                        if twin is not None:
                            d_["duplicateOf"] = d_["entry"]["duplicateOf"] = twin["id"]
                            duplicate_boxes += 1
                            continue
                    primary_.append(d_)
            floor_out, palert = None, {}
            deferred_labels = []  # v7d: vehicle notes drawn after the person labels
            if dz is not None:
                def loads_for(v):
                    # carried loads: at least 60 % of the load box inside this forklift's box (+5 % margin). v8 (23 Oct):
                    # the old rule (anything touching the box + 30 %/20 % margin) attached crate stacks standing on the
                    # floor next to a passing truck once the detector found them, and the "unit" reached the operators.
                    x1, y1, x2, y2 = v["box"]; mw, mh = 0.05 * (x2 - x1), 0.05 * (y2 - y1)
                    vb = np.array([v["box"]], float)
                    grown = (x1 - mw, y1 - mh, x2 + mw, y2 + mh)
                    return [np.asarray(q, np.float32) for b_, c_, q in found.get("loads", [])
                            if contained(b_, grown) >= LOAD_INSIDE_MIN
                            and iou_matrix(np.array([b_], float), vb)[0, 0] < 0.5]  # >= 0.5 overlap: the truck itself, not a load
                vin = [{"id": v["id"], "polygons": [np.asarray(q, np.float32) for q in (v["poly"] or []) if len(q) >= 3],
                        "loadPolygons": loads_for(v), "moving": v["moving"], "estimated": v["estimated"]} for v in shown_vehicles]
                attached_loads = [q for x in vin if not x["estimated"] for q in x["loadPolygons"]]
                floor_out = dz.step(t, vin,
                                    [{"id": d_["id"], "box": d_["entry"]["box"], "foot": d_["entry"]["foot"], "operator": d_["operator"] is not None}
                                     for d_ in pdraw if "duplicateOf" not in d_])
                vsum = {v_["id"]: v_ for v_ in floor_out["vehicles"]}
                moving_ids = {i for i, v_ in vsum.items() if v_["speed_mps"][1] >= dzm.MOVING_MPS}
                dz.draw_on_camera(canvas, floor_out)
                for q in attached_loads:  # carried loads attached to a forklift (thin cyan): what the blind zone uses
                    cv2.polylines(canvas, [np.round(np.clip(np.nan_to_num(np.asarray(q, np.float64), nan=-1e5), -1e5, 1e5)).astype(np.int32)], True, CYAN, 2, cv2.LINE_AA)
                record["floor"] = {"loads": [[round(float(x), 1) for x in b_] for b_, _, _ in found.get("loads", [])], "vehicles": [{k_: (np.round(v_, 3).tolist() if k_ == "footprint" else v_) for k_, v_ in x.items()} for x in floor_out["vehicles"]], "persons": {},
                                   "zones": dz.zone_polygons(floor_out)}
                for d_ in pdraw:
                    r_ = floor_out["persons"].get(d_["id"])
                    if r_ is None:
                        continue
                    veh = r_.get("vehicle")
                    moving = veh in moving_ids
                    p_path = max(r_["pInZone"]["stop"], r_["pInZone"]["tail"]); p_blind = r_["pInZone"]["blind"]
                    clear = r_["clearance_m"]["median"]
                    reasons = [n_ for n_, ok_ in (("in stopping/swing path", p_path >= 0.5), ("hidden in driver blind zone", p_blind >= 0.5),
                                                  ("closer than 0.8 m", clear < 0.8)) if ok_ and moving]
                    xy = np.array(r_["xy_m"], float) if r_.get("xy_m") else dz.nominal.to_plane(np.array(d_["entry"]["foot"], float))[0]
                    p_warn, ttc_ = r_.get("pWarn", 0.0), r_.get("ttc_s")
                    warn = bool(moving and p_warn >= 0.5)
                    palert[d_["id"]] = {"xy": xy, "alert": bool(reasons), "pPath": p_path, "pBlind": p_blind, "clear": r_["clearance_m"], "vehicle": veh, "reasons": reasons,
                                        "warn": warn, "pWarn": p_warn, "ttc": ttc_, "moving": moving, "pass": r_.get("pass_m"), "passT": r_.get("passT_s")}
                    record["floor"]["persons"][d_["id"]] = {"pInZone": r_["pInZone"], "vehicle": veh, "vehicleMoving": moving, "alert": reasons,
                                                            "clearance_m": {k_: (round(float(v_), 3) if np.isfinite(v_) else None) for k_, v_ in r_["clearance_m"].items()},
                                                            "pWarn": round(p_warn, 3), "ttc_s": None if ttc_ is None else {k_: round(v_, 2) for k_, v_ in ttc_.items()}, "warn": warn,
                                                            "pass_m": None if r_.get("pass_m") is None else {k_: round(v_, 3) for k_, v_ in r_["pass_m"].items()},
                                                            "passT_s": None if r_.get("passT_s") is None else round(r_["passT_s"], 2),
                                                            "xy_m": r_.get("xy_m"), "xySigma_m": r_.get("xySigma_m"), "seated": r_.get("seated")}
                    w_ = w_streak.setdefault(d_["id"], {"on": 0, "off": 0})
                    w_["on"], w_["off"] = (w_["on"] + 1, 0) if warn else (0, w_["off"] + 1)
                    palert[d_["id"]]["warnStreak"] = w_["on"]
                    if warn and w_["on"] >= EVENT_MIN_FRAMES and d_["id"] not in w_open:
                        w_open[d_["id"]] = {"type": "forklift_will_pass_within_0.8m", "person": d_["id"], "vehicle": veh,
                                            "startT": round(t - (EVENT_MIN_FRAMES - 1) / fps, 3), "confirmedT": round(t, 3), "confirmedFrame": index,
                                            "ttcAtConfirm_s": ttc_, "minTTC_s": ttc_["median"] if ttc_ else None, "maxPWarn": p_warn}
                        if not wkey:
                            wkey = {"frame": index, "t": t, "person": d_["id"], "ttc_s": ttc_, "pending": True}
                    if d_["id"] in w_open:
                        we_ = w_open[d_["id"]]
                        we_["maxPWarn"] = max(we_["maxPWarn"], p_warn)
                        if ttc_ and (we_["minTTC_s"] is None or ttc_["median"] < we_["minTTC_s"]):
                            we_["minTTC_s"] = ttc_["median"]
                        if w_["off"] >= EVENT_END_FRAMES:
                            we_["endT"] = round(t, 3); w_events.append(w_open.pop(d_["id"]))
                    s_ = f_streak.setdefault(d_["id"], {"on": 0, "off": 0})
                    s_["on"], s_["off"] = (s_["on"] + 1, 0) if reasons else (0, s_["off"] + 1)
                    palert[d_["id"]]["alertStreak"] = s_["on"]
                    if reasons and s_["on"] >= EVENT_MIN_FRAMES and d_["id"] not in f_open:
                        f_open[d_["id"]] = {"type": "person_in_forklift_danger_zone", "person": d_["id"], "vehicle": veh, "reasons": sorted(set(reasons)),
                                            "startT": round(t - (EVENT_MIN_FRAMES - 1) / fps, 3), "minClearance_m": clear, "maxPPath": p_path, "maxPBlind": p_blind, "atFrame": index}
                    if d_["id"] in f_open:
                        ev_ = f_open[d_["id"]]
                        ev_["reasons"] = sorted(set(ev_["reasons"]) | set(reasons))
                        if clear < ev_["minClearance_m"]:
                            ev_.update(minClearance_m=clear, atFrame=index)
                        ev_["maxPPath"] = max(ev_["maxPPath"], p_path); ev_["maxPBlind"] = max(ev_["maxPBlind"], p_blind)
                        if s_["off"] >= EVENT_END_FRAMES:
                            ev_["endT"] = round(t, 3); f_events.append(f_open.pop(d_["id"]))
                    if moving and clear < fbest["clear"]:
                        fbest = {"clear": clear, "frame": index, "t": t, "person": d_["id"], "pending": True}
                for pid_ in sorted(set(w_open) | set(f_open)):  # v7b: a person no longer measured counts as "off"
                    if pid_ in palert:
                        continue
                    for opn, stk, done in ((w_open, w_streak, w_events), (f_open, f_streak, f_events)):
                        if pid_ in opn:
                            st_ = stk.setdefault(pid_, {"on": 0, "off": 0})
                            st_["on"] = 0; st_["off"] += 1
                            if st_["off"] >= EVENT_END_FRAMES:
                                opn[pid_]["endT"] = round(t, 3); opn[pid_]["closedUnseen"] = True; done.append(opn.pop(pid_))
                ins = dz.inset(floor_out, palert)  # pasted before the labels so it never hides them
                canvas[100:100 + ins.shape[0], canvas.shape[1] - ins.shape[1] - 12:canvas.shape[1] - 12] = ins
                for sv in shown_vehicles:  # shown but not measurable on the floor (v6f): say so instead of drawing nothing
                    if sv["id"] not in vsum and not sv["estimated"]:
                        deferred_labels.append(("floor contact not visible: not measured", (sv["box"][0], sv["box"][1] + 34), (40, 140, 230), 0.75, sv["box"][1]))
                for v_ in floor_out["vehicles"]:  # speed, stopping distance and load notes beside each vehicle
                    sv = next((x for x in shown_vehicles if x["id"] == v_["id"]), None)
                    if sv is None:
                        continue
                    lines_ = []
                    if v_["speed_mps"][1] >= dzm.MOVING_MPS:
                        sp, ds = v_["speed_mps"], v_["stoppingDistance_m"]
                        acc_ = v_.get("accel_mps2", [0, 0, 0])[1] if v_.get("accelUsed", 0.0) >= 0.5 else 0.0  # v7b: significant only
                        lines_.append(f"{sp[1]:.1f} m/s" + (" accelerating" if acc_ >= 0.5 else " braking" if acc_ <= -0.5 else "")
                                      + f"  stops in {ds[1]:.1f} m ({ds[0]:.1f}-{ds[2]:.1f})")
                    else:
                        lines_.append("not moving")
                    if v_["pLoadBlocksView"] >= 0.5:
                        lf = v_["pLoadFirst"]
                        lines_.append(f"load {v_['loadHeight_m']:.1f} m tall: blocks driver view" + ("" if lf is None else (" - driving load-first" if lf >= 0.5 else " - load trailing")))
                    cols_ = [(0, 90, 220)] + [(200, 120, 30)] * (len(lines_) - 1)
                    turn_ok = v_.get("turnRateSE_radps", 9.0) <= 0.5 * abs(v_.get("turnRateFit_radps", 0.0))  # turn known to +/-50%
                    if v_.get("cutByFrame"):
                        lines_.append("cut by frame edge: floor outline uncertain"); cols_.append((40, 140, 230))
                    if v_["speed_mps"][1] >= dzm.MOVING_MPS and v_.get("pTurning", 0.0) > 0:
                        sc_ = v_["pTurning"] * (1.0 if turn_ok else 0.3) + 0.01 * v_["speed_mps"][1]  # prefer well-measured turns
                        if sc_ > tbest["score"]:
                            tbest = {"score": sc_, "frame": index, "t": t, "vehicle": v_["id"], "pTurning": v_["pTurning"],
                                     "turnRate_radps": v_["turnRate_radps"], "turnRateSE_radps": v_.get("turnRateSE_radps"),
                                     "turnRateFit_radps": v_.get("turnRateFit_radps"), "turnWellMeasured": bool(turn_ok),
                                     "speed_mps": v_["speed_mps"], "tailLikelyCells": int((floor_out["prob"]["tail"] >= 0.5).sum()),
                                     "tailPossibleCells": int((floor_out["prob"]["tail"] >= 0.05).sum()), "pending": True}
                    if v_.get("pTurning", 0.0) >= 0.5 and v_["speed_mps"][1] >= dzm.MOVING_MPS:
                        lines_.append(f"turning {abs(v_['turnRateFit_radps']):.1f} rad/s (+/-{v_['turnRateSE_radps']:.1f}): rear swings out" if turn_ok
                                      else f"turn rate uncertain (+/-{v_.get('turnRateSE_radps', 0.0):.1f} rad/s)"); cols_.append((200, 60, 200))
                    bx = sv["box"]
                    for i_, ln in enumerate(lines_):
                        deferred_labels.append((ln, (bx[0], bx[1] + 34 + 34 * i_), cols_[i_], 0.75, bx[1]))
            for d_ in pdraw:  # people are drawn after the zones so they stay visible
                b, foot, color = d_["box"], d_["foot"], d_["color"]
                if d_.get("duplicateOf"):
                    cv2.rectangle(canvas, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), GREY, 1)
                    continue
                pa = palert.get(d_["id"])
                if pa is not None and not d_["operator"]:
                    color = RED if pa["alert"] else AMBER if (pa["warn"] or pa["pPath"] >= 0.05 or pa["pBlind"] >= 0.05 or pa["clear"]["median"] < 1.5) else GREEN
                    if pa["warn"] and pa["ttc"]:
                        txt_ = (f"within 0.8 m now ({pa['clear']['median']:.1f} m)" if pa["ttc"]["median"] < 0.05
                                else f"within 0.8 m in {pa['ttc']['median']:.1f} s ({pa['clear']['median']:.1f} m now)")
                        label(canvas, txt_, (b[0], b[3] + 6), color, 0.75, top=b[3] + 6)
                    elif pa["moving"] and pa["pass"] and pa["pass"]["median"] < 3.0 and (pa["passT"] or 0) >= 0.1:
                        label(canvas, f"passes at {pa['pass']['median']:.1f} m in {pa['passT']:.1f} s", (b[0], b[3] + 6), color, 0.75, top=b[3] + 6)
                    elif np.isfinite(pa["clear"]["median"]) and pa["clear"]["median"] < 3.0:
                        label(canvas, f"{pa['clear']['median']:.1f} m from {pa['vehicle']}", (b[0], b[3] + 6), color, 0.75, top=b[3] + 6)
                elif dz is None and d_["rel"] is not None and d_["rel"] < CLOSE_REL:
                    rel, near_pt = d_["rel"], d_["near_pt"]
                    cv2.line(canvas, tuple(int(v) for v in foot), tuple(int(v) for v in near_pt), color, 3, cv2.LINE_AA)
                    label(canvas, f"{rel:.1f} person-heights" + (" (vehicle estimated)" if d_["near_est"] else "") + ("" if d_["near_moving"] else " (parked)"), ((foot[0] + near_pt[0]) / 2, (foot[1] + near_pt[1]) / 2), color, 0.8)
                cv2.rectangle(canvas, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 3)
                cv2.circle(canvas, tuple(int(v) for v in foot), 7, color, -1, cv2.LINE_AA)
                entry = d_["entry"]
                text = f"person {d_['id']}" + (" (operator)" if d_["operator"] else "") + (f" - {entry['walkway']}" if "walkway" in entry and not d_["operator"] else "")
                label(canvas, text, (b[0], b[1] - 6), color, 0.8)
            for txt_, pos_, col_, sc_, top_ in deferred_labels:
                label(canvas, txt_, pos_, col_, sc_, top=top_)
            active = [ev for ev in open_events.values()]
            f_active = list(f_open.values())
            if dz is not None and f_active:
                ev = min(f_active, key=lambda e: e["minClearance_m"])
                cv2.rectangle(canvas, (0, canvas.shape[0] - 70), (canvas.shape[1], canvas.shape[0]), (30, 30, 200), -1)
                cv2.putText(canvas, f"DANGER  person {ev['person']} {' + '.join(ev['reasons'])} of {ev['vehicle']}  (floor, metres)",
                            (24, canvas.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.05, WHITE, 2, cv2.LINE_AA)
            elif dz is not None and any(pa_["alert"] or (pa_["warn"] and k_ not in w_open) for k_, pa_ in palert.items()):
                k_, pa_ = min(((k_, pa_) for k_, pa_ in palert.items() if pa_["alert"] or (pa_["warn"] and k_ not in w_open)),
                              key=lambda kv: (not kv[1]["alert"], kv[1]["clear"]["median"]))
                if pa_["alert"]:
                    what = f"{pa_['reasons'][0]} ({pa_['clear']['median']:.1f} m)"
                    n_ = pa_.get("alertStreak", 1)
                else:
                    what = f"forklift may pass within 0.8 m in {pa_['ttc']['median']:.1f} s" if pa_["ttc"] else "forklift may pass within 0.8 m"
                    n_ = pa_.get("warnStreak", 1)
                cv2.rectangle(canvas, (0, canvas.shape[0] - 70), (canvas.shape[1], canvas.shape[0]), (20, 110, 170), -1)
                cv2.putText(canvas, f"CHECKING  person {k_}: {what} - frame {min(n_, EVENT_MIN_FRAMES)} of {EVENT_MIN_FRAMES} needed",
                            (24, canvas.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)
            elif dz is not None and any(palert.get(k, {}).get("warn") for k in w_open):
                k_ = min((k for k in w_open if palert.get(k, {}).get("warn")), key=lambda k: (palert[k].get("ttc") or {"median": 9.0})["median"])
                pa_ = palert.get(k_, {})
                tt_ = pa_.get("ttc")
                ttxt = ("now" if tt_ and tt_["median"] < 0.05 else f"in {tt_['median']:.1f} s ({tt_['p5']:.1f}-{tt_['p95']:.1f})" if tt_ else "soon")
                cv2.rectangle(canvas, (0, canvas.shape[0] - 70), (canvas.shape[1], canvas.shape[0]), (0, 150, 245), -1)
                cv2.putText(canvas, f"WARNING  person {k_}: {w_open[k_]['vehicle']} will pass within 0.8 m {ttxt} if it keeps its course",
                            (24, canvas.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
            elif dz is not None and active:
                ev = min(active, key=lambda e: e["minPersonHeights"])
                pa = palert.get(ev["person"])
                floor_txt = (f"floor clearance {pa['clear']['median']:.1f} m" if pa and np.isfinite(pa["clear"]["median"])
                             else "no forklift measured on the floor" if pa else "person not seen this frame")
                cv2.rectangle(canvas, (0, canvas.shape[0] - 70), (canvas.shape[1], canvas.shape[0]), (60, 60, 60), -1)
                cv2.putText(canvas, f"NO ALARM  pixels said close ({ev['minPersonHeights']:.1f} person-heights), {floor_txt} for person {ev['person']}",
                            (24, canvas.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)
            elif active:
                ev = min(active, key=lambda e: e["minPersonHeights"])
                cv2.rectangle(canvas, (0, canvas.shape[0] - 70), (canvas.shape[1], canvas.shape[0]), (30, 30, 200), -1)
                cv2.putText(canvas, f"PROXIMITY ALERT  person {ev['person']} close to {ev['vehicle']}  (uncalibrated pixel ratio, not a distance)",
                            (24, canvas.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.05, WHITE, 2, cv2.LINE_AA)
        bar = canvas.copy(); cv2.rectangle(bar, (0, 0), (canvas.shape[1], 92), (15, 15, 15), -1)
        canvas = cv2.addWeighted(bar, 0.72, canvas, 0.28, 0)
        cv2.putText(canvas, "FLOORSIGHT  real factory footage  |  faces and heads blurred before any output", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)
        sub_txt = "floor-calibrated (estimated scale): metres, speeds, danger zones" if dz is not None else "pixel-level, uncalibrated - no metres or speeds"
        cv2.putText(canvas, f"{sub_txt}  |  source: Onal & Dandil 2024, CC BY 4.0  |  t={t:5.2f}s", (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.75, GREY, 2, cv2.LINE_AA)
        if fbest.get("pending"):
            fbest["image"] = canvas.copy(); fbest.pop("pending")
        if tbest.get("pending"):
            tbest["image"] = canvas.copy(); tbest.pop("pending")
        if wkey.get("pending"):
            wkey["image"] = canvas.copy(); wkey.pop("pending")
        if mark_best:
            best["image"] = canvas.copy()
        if mark_large:
            largest["image"] = canvas.copy()
        encoder.write(canvas)
        records.append(record)
        index += 1
    for ev in open_events.values():
        ev["endT"] = round((index - 1) / fps, 3); ev["openAtClipEnd"] = True; events.append(ev)
    for ev in f_open.values():
        ev["endT"] = round((index - 1) / fps, 3); ev["openAtClipEnd"] = True; f_events.append(ev)
    for ev in w_open.values():
        ev["endT"] = round((index - 1) / fps, 3); ev["openAtClipEnd"] = True; w_events.append(ev)
    for ev in w_events:  # what followed each warning: the person's closest floor clearance to a moving forklift
        after = [(r_["t"], pr["clearance_m"]["median"]) for r_ in records for pid_, pr in (r_.get("floor") or {}).get("persons", {}).items()
                 if pid_ == ev["person"] and pr.get("vehicleMoving") and pr["clearance_m"]["median"] is not None and ev["startT"] <= r_["t"] <= ev["endT"] + 2.0]
        if after:
            t_c, c_c = min(after, key=lambda x: x[1])
            ev["closestAfter"] = {"t": round(t_c, 3), "clearance_m": c_c}
            ev["warningLead_s"] = round(t_c - ev["confirmedT"], 3)
    if "image" in wkey:
        cv2.imwrite(str(args.output / f"{args.name}-warn-key.jpg"), cv2.resize(wkey.pop("image"), (1600, 900), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if "image" in fbest:
        cv2.imwrite(str(args.output / f"{args.name}-floor-key.jpg"), cv2.resize(fbest.pop("image"), (1600, 900), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if "image" in tbest:
        cv2.imwrite(str(args.output / f"{args.name}-turn-key.jpg"), cv2.resize(tbest.pop("image"), (1600, 900), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
    encoder_exit = encoder.close()
    clean_exit = clean.close()

    out = args.output
    # Key frame: closest person-vehicle observation; without vehicles, the largest visible person.
    key = best if "image" in best else largest
    key_image = key.pop("image", None)
    largest.pop("image", None); best.pop("image", None)
    if key_image is not None:
        cv2.imwrite(str(out / f"{args.name}-key.jpg"), cv2.resize(key_image, (1600, 900), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if "image" in privacy_crop:
        crop = privacy_crop.pop("image")
        scale = min(3.0, 900 / max(crop.shape[:2]))
        cv2.imwrite(str(out / f"{args.name}-privacy-crop.jpg"), cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST), [cv2.IMWRITE_JPEG_QUALITY, 92])
    audit = out / "privacy-audit"; audit.mkdir()
    for frame_index, image in audit_frames:
        cv2.imwrite(str(audit / f"{args.name}-no-person-candidate-f{frame_index:04d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if track_crops:
        tiles = []
        for tid, (area, crop, frame_index) in sorted(track_crops.items(), key=lambda kv: -kv[1][0])[:16]:
            tile = cv2.resize(crop, (max(1, int(crop.shape[1] * 320 / crop.shape[0])), 320), interpolation=cv2.INTER_NEAREST)
            label(tile, f"{tid} f{frame_index}", (4, 0), WHITE, 0.6, top=6)
            tiles.append(tile)
        rows, row, row_w = [], [], 0
        for tile in tiles:
            if row and row_w + tile.shape[1] > 1600:
                rows.append(row); row, row_w = [], 0
            row.append(tile); row_w += tile.shape[1]
        rows.append(row)
        canvas_w = max(sum(t.shape[1] for t in r) for r in rows)
        sheet_img = np.vstack([np.hstack(r + [np.zeros((320, canvas_w - sum(t.shape[1] for t in r), 3), np.uint8)]) for r in rows])
        cv2.imwrite(str(out / f"{args.name}-person-crops.jpg"), sheet_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if sheet:
        while len(sheet) % 4:
            sheet.append(np.zeros_like(sheet[0]))
        rows = [np.hstack(sheet[i:i + 4]) for i in range(0, len(sheet), 4)]
        cv2.imwrite(str(out / f"{args.name}-filtered-contact.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    webm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out / f"{args.name}-overlay.mp4"), "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "33",
                           "-row-mt", "1", "-deadline", "good", "-cpu-used", "4", "-an", str(out / f"{args.name}-overlay.webm")]).returncode
    poster = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{key.get('t', 0):.3f}", "-i", str(out / f"{args.name}-overlay.mp4"), "-frames:v", "1", "-q:v", "3",
                             str(out / f"{args.name}-poster.jpg")]).returncode

    # Fail-closed self-test on a fresh decode of frame 0: a forced failure must yield a black, withheld frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, raw = cap.read()
    test_frame = withheld(raw.shape, "forced privacy failure (fail-closed self-test)") if ok else None
    self_test = bool(ok and test_frame[100:200, :].max() == 0 and test_frame.mean() < 10)
    del raw
    with (out / f"{args.name}-privacy-regions.jsonl").open("x") as handle:
        for r in blurred_regions:
            handle.write(json.dumps(r) + "\n")
    with (out / f"{args.name}-detections.jsonl").open("x") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    withheld_frames = [h for h in health if h["status"] != "ok"]
    result = {
        "schema": "floorsight.real-clip-perception.v1", "name": args.name, "clipSha256": sha256(args.clip), "environment": environment(),
        "frames": index, "fps": fps, "frameSize": [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))],
        "seconds": round(time.time() - started, 1), "detectors": det.notes,
        "motionSmoothing": None if dz is None else {"on": smoothing, "counts": dz.motion_stats, "personDistanceMatches": people.dist_matches, "duplicatePersonBoxes": duplicate_boxes, "operatorHeldFrames": operator_held,
                                                    "settings": {k_: getattr(dzm, k_) for k_ in ("VEH_ACCEL_SIGMA", "VEH_MEAS_SIGMA", "PER_ACCEL_SIGMA", "PER_MEAS_SIGMA",
                                                                                               "KF_RESTART_S", "COAST_MAX_S", "TURN_ON", "TURN_OFF", "TURN_FRAMES",
                                                                                               "SEAT_FRAMES", "SHAPE_WINDOW_STAT", "SHAPE_SMOOTH_S", "HEADING_MIN_MPS", "HEADING_RATE_MAX",
                                                                                               "VEH_RESET_AFTER", "TURN_ALPHA_SIGMA", "REVERSE_FRAMES", "AXIS_SWAP_FRAMES")},
                                                    "staticShowMedian": STATIC_SHOW_MEDIAN},
        "dangerZoneSkips": None if dz is None else {"vehicleFramesNoFloorContact": getattr(dz, "skipped_no_contact", 0),
                                                    "vehicleFramesCutBySideEdge": getattr(dz, "skipped_side_cut", 0),
                                                    "nonFinitePointsReplaced": getattr(dz, "nonfinite_points", 0)},
        "timing": {"detector_s": round(det_s, 2), "detector_stages_s": {k: round(v, 2) if isinstance(v, float) else v for k, v in getattr(det, "timing", {}).items()},
                   "opencv_build": cv2.getBuildInformation().split("Custom HAL:")[-1].split("\n")[0].strip() if "Custom HAL:" in cv2.getBuildInformation() else "none",
                   "cv2_file": cv2.__file__, "opencv_threads": cv2.getNumThreads(),
                   "cpu_threads_requested": args.cpu_threads, "detector_workers": args.detector_workers,
                   "background_sampling": args.background_sampling, "background_sample_s": background_sample_s,
                   "background_median": args.background_median, "background_median_s": background_median_s},
        "thresholds": {"privacyPersonConf": PRIVACY_PERSON_CONF, "displayPersonConf": DISPLAY_PERSON_CONF, "trackPersonConf": TRACK_PERSON_CONF,
                       "forkliftConf": FORKLIFT_CONF, "forkliftPrompts": FORKLIFT_PROMPTS, "motionMaxFrames": MOTION_MAX_FRAMES,
                       "movingFraction": MOVING_FRACTION, "movingMemoryFrames": MOVING_MEMORY, "staticShowConf": STATIC_SHOW_CONF, "motionWarmupFrames": MOTION_WARMUP,
                       "closePersonHeights": CLOSE_REL, "veryClosePersonHeights": VERY_CLOSE_REL, "eventPersonHeights": EVENT_REL,
                       "eventMinFrames": EVENT_MIN_FRAMES, "eventEndFrames": EVENT_END_FRAMES},
        "walkway": {"found": walkway is not None, "frameFraction": round(walkway_fraction, 4), "mappedFromRow": walkway_top, "method": "seeded HSV paint growth on temporal-median background (v2)", **walkway_stats},
        "privacy": {"framesOk": len(health) - len(withheld_frames), "framesWithheld": len(withheld_frames), "withheld": withheld_frames[:20],
                    "headsBlurredTotal": sum(h.get("headsBlurred", 0) for h in health), "facesBlurredTotal": sum(h.get("facesBlurred", 0) for h in health),
                    "vehicleCabsBlurredTotal": sum(h.get("vehicleCabsBlurred", 0) for h in health),
                    "framesWithEstimatedVehicle": sum(1 for r in records if any(v["estimated"] for v in r["vehicles"])),
                    "framesWithDetectedVehicle": sum(1 for r in records if any(not v["estimated"] for v in r["vehicles"])),
                    "framesWithNoPersonCandidate": sum(1 for h in health if h.get("personCandidates") == 0),
                    "failClosedSelfTest": "pass" if self_test else "fail", "auditFramesSaved": [i for i, _ in audit_frames], "trackCrops": len(track_crops), "forcedFailureFrames": args.force_privacy_failure_frame,
                    "approval": "NOT APPROVED by this script: requires human review of filtered frames; an empty detection list is not a pass."},
        "floorEvents": f_events if dz is not None else None,
        "floorWarnings": w_events if dz is not None else None,
        "floorWarnKey": None if dz is None or "frame" not in wkey else {k: (round(v, 3) if isinstance(v, float) else v) for k, v in wkey.items()},
        "floorTurnKey": None if dz is None or "frame" not in tbest else {k: (round(v, 3) if isinstance(v, float) else v) for k, v in tbest.items()},
        "floorClosestMovingObservation": None if dz is None or not np.isfinite(fbest["clear"]) else {k: (round(v, 3) if isinstance(v, float) else v) for k, v in fbest.items()},
        "dangerZone": None if dz is None else {"calibration": str(args.calibration), "draws": args.danger_draws, "priors": dzm.PRIORS,
                                               "blindReach_m": dzm.BLIND_REACH_M, "tallLoadColumn_m": dzm.TALL_LOAD_COLUMN_M, "movingMps": dzm.MOVING_MPS,
                                               "alertRule": "moving vehicle AND (P(person in stopping or tail-swing zone) >= 0.5 OR P(in blind zone while load-first) >= 0.5 OR median floor clearance < 0.8 m), confirmed over 3 frames",
                                               "warningRule": f"moving vehicle AND P(it comes within {dzm.WARN_CLEAR_M} m within {dzm.WARN_HORIZON_S} s if it keeps its course) >= 0.5, confirmed over 3 frames",
                                               "speedModel": f"straight-line fit over {dzm.WINDOW_S} s (plus its standard error per draw) and the along-track acceleration from a quadratic fit when >= 2 SE, clipped to {dzm.ACCEL_LIMITS} m/s2; zones use max(mean speed, extrapolated speed) and only a positive acceleration (a slowdown estimate never shrinks a zone); no speed below {dzm.SPEED_MIN_SPAN_S} s of track",
                                               "bannerPriority": "DANGER > CHECKING (condition met, fewer than 3 frames) > WARNING > NO ALARM (pixel rule only)"},
        "events": events, "closestObservation": None if best["rel"] == float("inf") else {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()},
        "keyFrame": {"source": "closest person-vehicle observation" if key is best else "largest visible person", "frame": key.get("frame"), "t": key.get("t")},
        "privacyCrop": privacy_crop, "encoderExit": {"h264": encoder_exit, "vp9": webm, "poster": poster, "filtered": clean_exit},
        "limitations": ["Image-plane only: person-heights is a perspective-relative pixel ratio, not metres.",
                        "Touching vehicle detections are merged into one unit (cab plus load); two touching vehicles would also merge.",
                        "Single development clip; not held-out evaluation; no ground-truth labels were used or scored.",
                        "Head regions are estimated from person boxes; small, occluded or seated people can be partially covered."],
        "finishedAt": datetime.now(timezone.utc).isoformat(),
    }
    if saved_detections is not None:
        args.save_detections.write_bytes(pickle.dumps({"frames": saved_detections, "notes": det.notes, "timing": dict(getattr(det, "timing", {})),
                                                       "run": str(out), "clipSha256": result["clipSha256"]}))
    (out / f"{args.name}-result.json").write_text(json.dumps(result, indent=2, default=str))
    close = getattr(det, "close", None)
    if close is not None:
        close()
    print(json.dumps({k: result[k] for k in ("name", "frames", "seconds", "detectors", "walkway", "events", "encoderExit")}, default=str))
    print(json.dumps(result["privacy"], default=str))
    return 0 if encoder_exit == 0 and self_test else 1


if __name__ == "__main__":
    raise SystemExit(main())
