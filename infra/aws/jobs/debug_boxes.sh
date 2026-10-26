#!/bin/bash
# Debug: on a few screen-002 frames, print RT-DETR truck/car boxes, Grounding DINO forklift/load boxes and the SlimSAM
# outline extents for each candidate box (numbers only; raw frames never leave the process).
sudo -u floorsight /opt/floorsight/perc - <<'PY' 2>&1 | grep -v Warning | grep -v warn
import importlib.util, json, cv2, numpy as np
from pathlib import Path
spec = importlib.util.spec_from_file_location("od", "/opt/floorsight/current/scripts/open_detectors.py"); od = importlib.util.module_from_spec(spec); spec.loader.exec_module(od)
hf = {k: v.get("path") for k, v in json.load(open("/opt/floorsight/jobs/setup-01/hf-models.json")).items()}
rt = od.RTDetrCoco(hf["PekingU/rtdetr_v2_r50vd"], conf=0.3)
gd = od.OpenVocab("gdino", hf["IDEA-Research/grounding-dino-tiny"])
sam = od.BoxSegmenter(hf["Zigeng/SlimSAM-uniform-77"], "slimsam77")
cap = cv2.VideoCapture("/opt/floorsight/data/private/clips/screen-002.mp4")
for f in (100, 150):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
    c = rt.detect(img)
    trucks = [(b.round().tolist(), round(s, 2)) for k in (od.COCO_TRUCK, od.COCO_CAR) for b, s in c[k]]
    g = [(b.round().tolist(), round(s, 2), n) for b, s, n in gd.detect(img, ["person", "forklift", "forklift truck", "industrial vehicle", "stack of crates", "crate", "pallet", "box pallet"], 0.15) if od.label_kind(n) in ("forklift", "load")]
    print("frame", f, "RTDETR truck/car", trucks)
    print("   GDINO", g)
    boxes = [np.array(t[0], np.float32) for t in trucks] + [np.array(x[0], np.float32) for x in g if od.label_kind(x[2]) == "forklift"]
    for b, p in zip(boxes, sam.polygons(img, boxes)):
        print("   box", b.round().tolist(), "-> outline y", None if p is None else (int(p[:, 1].min()), int(p[:, 1].max())), "x", None if p is None else (int(p[:, 0].min()), int(p[:, 0].max())))
PY
