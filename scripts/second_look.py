"""Second detector pass for the Floorsight agent (tool `redetect`), run in the perception environment (Apache-2.0 models only).

Called by floorsight.realclip.ReplayVision.redetect with one argument, a request JSON:
    {"image": crop.png (privacy-filtered pixels only), "origin": [x0, y0] of the crop in the full frame,
     "person_box": [x1,y1,x2,y2], "vehicle_box": [...] (full-frame px, from the tracker), "heading": rad,
     "calibration": {K, R, h}, "annotated": path for the annotated crop}
It upscales the crop, runs the people detector (YOLOX via cv2.dnn) and the forklift/load detector + box-prompted SAM (the same
prompts as the perception stage) again at a higher input size, picks the detections that match the tracker's person
and forklift, and re-measures the floor gap (person foot to the forklift's footprint built from its visible ground
line) with the same calibration. Prints one JSON line. It measures; it never decides.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = Path(os.environ.get("FLOORSIGHT_MODELS", "."))
SCALE_TARGET_H = 900


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    req = json.loads(Path(sys.argv[1]).read_text())
    fc, fr, rcp = _load("floor_calibration"), _load("floor_retest"), _load("real_clip_perception")


    img = cv2.imread(req["image"])
    if img is None:
        print(json.dumps({"error": "crop not readable"}))
        return 0
    ox, oy = req["origin"]
    src, pixels = img, "filtered"
    raw_map = os.environ.get("FLOORSIGHT_RAW_CLIPS")
    if raw_map and req.get("clip") and req.get("frame") is not None:
        path = json.loads(Path(raw_map).read_text()).get(req["clip"])
        if path:
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(req["frame"]))
            ok, raw = cap.read()
            cap.release()
            if ok:
                h, w = img.shape[:2]
                src, pixels = raw[oy:oy + h, ox:ox + w].copy(), "raw-at-edge"
            del raw  # raw pixels never leave this process; annotations go on the filtered crop
    s = max(1.0, min(3.0, SCALE_TARGET_H / src.shape[0]))
    big = cv2.resize(src, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    del src
    # Apache-2.0 stack (open_detectors.py), the same models as the perception stage, on the upscaled crop
    od = _load("open_detectors")
    hf = {k: v.get("path") for k, v in json.loads(Path(os.environ["FLOORSIGHT_HF_INDEX"]).read_text()).items()}
    vocab_ids = {"omdet": "omlab/omdet-turbo-swin-tiny-hf", "gdino": "IDEA-Research/grounding-dino-tiny", "owlv2": "google/owlv2-base-patch16-ensemble"}
    seg_ids = {"sam-vit-base": "facebook/sam-vit-base", "slimsam77": "Zigeng/SlimSAM-uniform-77"}
    vocab_kind, seg_kind = os.environ.get("FLOORSIGHT_VOCAB", "omdet"), os.environ.get("FLOORSIGHT_SEGMENTER", "sam-vit-base")
    people = od.YoloxDnn(MODELS / "onnx" / f"{os.environ.get('FLOORSIGHT_PERSON_MODEL', 'yolox_l')}.onnx", tiles=1, conf=0.25).detect(big)[od.COCO_PERSON]
    pboxes = [(b / s + np.array([ox, oy, ox, oy])) for b, _ in people]
    names = list(rcp.FORKLIFT_PROMPTS) + list(getattr(rcp, "LOAD_PROMPTS", []))
    found = od.OpenVocab(vocab_kind, hf[vocab_ids[vocab_kind]]).detect(big, names, min(rcp.FORKLIFT_CONF, 0.25))
    boxes = [b for b, _, _ in found]
    polys = od.BoxSegmenter(hf[seg_ids[seg_kind]], seg_kind).polygons(big, boxes)
    vb = req["vehicle_box"]
    grow = [vb[0] - 0.3 * (vb[2] - vb[0]), vb[1] - 0.2 * (vb[3] - vb[1]), vb[2] + 0.3 * (vb[2] - vb[0]), vb[3] + 0.2 * (vb[3] - vb[1])]
    unit = []
    for b, q in zip(boxes, polys):
        bf = b / s + np.array([ox, oy, ox, oy])
        if q is not None and len(q) >= 3 and (iou(bf, vb) >= 0.2 or iou(bf, grow) >= 0.2):
            unit.append(np.asarray(q, np.float32) / s + np.array([ox, oy], np.float32))
    person = max(pboxes, key=lambda b: iou(b, req["person_box"]), default=None)
    if person is not None and iou(person, req["person_box"]) < 0.2:
        person = None
    ann = img.copy()
    for b in pboxes:
        x1, y1, x2, y2 = ((b - np.array([ox, oy, ox, oy]))).astype(int)
        cv2.rectangle(ann, (x1, y1), (x2, y2), (60, 200, 60), 2)
    for q in unit:
        cv2.polylines(ann, [(q - np.array([ox, oy])).astype(np.int32)], True, (0, 140, 255), 2, cv2.LINE_AA)
    del big
    out = {"persons": len(pboxes), "vehicles": len(unit), "scale": round(s, 2), "clearance_m": None, "pixels": pixels}
    if person is None or not unit:
        out["note"] = "person not re-detected" if person is None else "forklift not re-detected"
    else:
        cal = req["calibration"]
        cam = fc.Camera(cal["K"], cal["R"], cal["h"])
        px, _ = fr.near_side_pixels(unit, (1920, 1080), edge_px=8, max_band_of_width=0.25)  # same contact band as the danger zones (v8)
        if len(px) < 6:
            out["note"] = "forklift floor contact not visible"
        else:
            ground = cam.to_plane(px, 0.1)
            ground = ground[np.isfinite(ground).all(1)]
            foot = cam.to_plane(np.array([(person[0] + person[2]) / 2, person[3]]))[0]
            heading = float(req.get("heading") or 0.0)
            foot_rect = fr.rect_footprint(ground, heading, np.zeros(2), 3.0, 1.15)
            out["clearance_m"] = round(float(fr.point_rect_distance(foot, foot_rect)), 3)
            out["groundPoints"] = int(len(ground))
            for u, v in px:
                cv2.circle(ann, (int(u - ox), int(v - oy)), 3, (0, 0, 255), -1)
            fx, fy = (person[0] + person[2]) / 2 - ox, person[3] - oy
            cv2.circle(ann, (int(fx), int(fy)), 6, (255, 255, 255), -1)
            cv2.putText(ann, f"re-measured {out['clearance_m']:.2f} m", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if req.get("annotated"):
        cv2.imwrite(req["annotated"], ann)
    out["opencv"] = cv2.__version__
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
