"""Compare Apache-2.0 detector candidates with the accepted detections on the same real frames.

    perc scripts/detector_probe.py --clips DIR --reference DIR --models DIR --out probe.json [--frames 12]

Reference = the per-frame detections of the accepted perception runs (people and forklifts after tracking,
forklift outlines from the open-vocabulary segmenter). For each candidate it reports, per clip and overall:
time per frame, share of reference people / forklifts found (IoU >= 0.5), extra boxes, and forklift outline
IoU for the box-prompted segmenters. Raw frames stay on this machine; only numbers are written.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

PERSON_CONF = 0.10   # same recall-oriented floor as the privacy stage
VEHICLE_CONF = 0.18
FORKLIFT_PROMPTS = ["forklift", "forklift truck", "industrial vehicle"]
LOAD_PROMPTS = ["stack of crates", "crate", "pallet", "box pallet"]
SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5)


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def recall(ref, det, thr=0.5):
    if not ref:
        return None, 0
    used, hit = set(), 0
    for r in ref:
        best, bi = 0.0, None
        for i, d in enumerate(det):
            if i in used:
                continue
            v = iou(r, d)
            if v > best:
                best, bi = v, i
        if best >= thr:
            hit += 1
            used.add(bi)
    return hit / len(ref), len(det) - len(used)


def nms(boxes, scores, thr=0.5):
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes], list(map(float, scores)), 0.0, thr)
    return [int(i) for i in np.array(keep).reshape(-1)]


# --------------------------------------------------------------------------- candidates


class YoloxDnn:
    """YOLOX (Megvii, Apache-2.0) COCO detector run by OpenCV's own DNN module."""

    def __init__(self, onnx: Path, size: int = 640, tiles: int = 1):
        self.net = cv2.dnn.readNet(str(onnx))
        self.size, self.tiles = size, tiles
        self.name = f"{onnx.stem}-cv2dnn" + (f"-tiles{tiles}" if tiles > 1 else "")
        grids, strides = [], []
        for s in (8, 16, 32):
            n = size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grid, self.stride = np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)

    def _run(self, img):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        pad = np.full((self.size, self.size, 3), 114, np.uint8)
        rs = cv2.resize(img, (int(round(w * r)), int(round(h * r))), interpolation=cv2.INTER_LINEAR)
        pad[: rs.shape[0], : rs.shape[1]] = rs
        self.net.setInput(cv2.dnn.blobFromImage(pad, 1.0, (self.size, self.size), swapRB=False))
        o = self.net.forward().reshape(-1, 85).copy()
        o[:, :2] = (o[:, :2] + self.grid) * self.stride
        o[:, 2:4] = np.exp(o[:, 2:4]) * self.stride
        sc = o[:, 4:5] * o[:, 5:]
        cls, conf = sc.argmax(1), sc.max(1)
        m = conf >= PERSON_CONF
        o, cls, conf = o[m], cls[m], conf[m]
        b = np.stack([o[:, 0] - o[:, 2] / 2, o[:, 1] - o[:, 3] / 2, o[:, 0] + o[:, 2] / 2, o[:, 1] + o[:, 3] / 2], 1) / r
        return b, cls, conf

    def detect(self, img):
        h, w = img.shape[:2]
        parts = [(0, img)] if self.tiles == 1 else [(x, img[:, x: x + h]) for x in np.linspace(0, w - h, self.tiles).astype(int)]
        if self.tiles > 1:
            parts.append((0, img))
        boxes, scores, kinds = [], [], []
        for x0, im in parts:
            b, c, s = self._run(im)
            for bb, cc, ss in zip(b, c, s):
                if cc == 0:
                    boxes.append([bb[0] + x0, bb[1], bb[2] + x0, bb[3]]); scores.append(float(ss)); kinds.append("person")
                elif cc in (2, 7) and ss >= VEHICLE_CONF:  # car/truck: COCO has no forklift class
                    boxes.append([bb[0] + x0, bb[1], bb[2] + x0, bb[3]]); scores.append(float(ss)); kinds.append("vehicle")
        out = {"person": [], "vehicle": []}
        for k in out:
            idx = [i for i, kk in enumerate(kinds) if kk == k]
            for j in nms([boxes[i] for i in idx], [scores[i] for i in idx]):
                out[k].append((boxes[idx[j]], float(scores[idx[j]])))
        return out


class HFDetector:
    """Transformers object detectors: RT-DETRv2 (closed set), OmDet-Turbo / Grounding DINO / OWLv2 (open vocabulary)."""

    def __init__(self, kind: str, path: str, size: int | None = None):
        import torch
        from transformers import AutoProcessor

        self.torch, self.kind, self.size = torch, kind, size
        self.name = f"{kind}" + (f"-{size}" if size else "")
        if kind == "rtdetrv2":
            from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
            self.proc = RTDetrImageProcessor.from_pretrained(path)
            self.model = RTDetrV2ForObjectDetection.from_pretrained(path).eval()
        elif kind == "omdet":
            from transformers import OmDetTurboForObjectDetection
            self.proc = AutoProcessor.from_pretrained(path)
            self.model = OmDetTurboForObjectDetection.from_pretrained(path, low_cpu_mem_usage=False).eval()
        elif kind == "gdino":
            from transformers import AutoModelForZeroShotObjectDetection
            self.proc = AutoProcessor.from_pretrained(path)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path).eval()
        elif kind == "owlv2":
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            self.proc = Owlv2Processor.from_pretrained(path)
            self.model = Owlv2ForObjectDetection.from_pretrained(path).eval()
        self.labels = ["person"] + FORKLIFT_PROMPTS

    def detect(self, img):
        from PIL import Image

        torch = self.torch
        h, w = img.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        with torch.inference_mode():
            if self.kind == "rtdetrv2":
                kw = {"size": {"height": self.size, "width": self.size}} if self.size else {}
                inp = self.proc(images=pil, return_tensors="pt", **kw)
                res = self.proc.post_process_object_detection(self.model(**inp), target_sizes=torch.tensor([(h, w)]), threshold=PERSON_CONF)[0]
                names = [self.model.config.id2label[int(i)] for i in res["labels"]]
                names = ["person" if n == "person" else "vehicle" if n in ("car", "truck") else n for n in names]
            elif self.kind == "omdet":
                inp = self.proc(images=pil, text=self.labels, return_tensors="pt")
                out = self.model(**inp)
                res = self.proc.post_process_grounded_object_detection(out, text_labels=[self.labels], target_sizes=[(h, w)], threshold=PERSON_CONF, nms_threshold=0.3)[0]
                names = list(res.get("text_labels", res.get("classes", [])))
            elif self.kind == "gdino":
                inp = self.proc(images=pil, text=". ".join(self.labels) + ".", return_tensors="pt")
                out = self.model(**inp)
                res = self.proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=PERSON_CONF, text_threshold=PERSON_CONF, target_sizes=[(h, w)])[0]
                names = list(res.get("text_labels", res.get("labels", [])))
            else:  # owlv2 pads to a square
                texts = [[f"a photo of a {x}" for x in self.labels]]
                inp = self.proc(text=texts, images=pil, return_tensors="pt")
                out = self.model(**inp)
                s = max(h, w)
                res = self.proc.post_process_grounded_object_detection(outputs=out, threshold=PERSON_CONF, target_sizes=torch.tensor([[s, s]]), text_labels=texts)[0]
                names = [x.replace("a photo of a ", "") for x in res.get("text_labels", [])]
        boxes, scores = res["boxes"].cpu().numpy().tolist(), res["scores"].cpu().numpy().tolist()
        out = {"person": [], "vehicle": []}
        for k in out:
            idx = [i for i, n in enumerate(names) if (n == "person") == (k == "person") and (k == "person" or any(p in str(n) for p in FORKLIFT_PROMPTS + ["vehicle"]))]
            idx = [i for i in idx if k == "person" or scores[i] >= VEHICLE_CONF]
            for j in nms([boxes[i] for i in idx], [scores[i] for i in idx]):
                out[k].append((boxes[idx[j]], float(scores[idx[j]])))
        return out


class SamMasks:
    """Box-prompted masks (SAM ViT-B / SlimSAM, Apache-2.0) for forklift outlines."""

    def __init__(self, path: str, name: str):
        import torch
        from transformers import SamModel, SamProcessor

        self.torch, self.name = torch, name
        self.proc, self.model = SamProcessor.from_pretrained(path), SamModel.from_pretrained(path).eval()

    def masks(self, img, boxes):
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        with self.torch.inference_mode():
            inp = self.proc(pil, input_boxes=[[list(map(float, b)) for b in boxes]], return_tensors="pt")
            out = self.model(**inp, multimask_output=False)
            m = self.proc.image_processor.post_process_masks(out.pred_masks, inp["original_sizes"], inp["reshaped_input_sizes"])[0]
        return [x[0].numpy().astype(bool) for x in m]


# --------------------------------------------------------------------------- main


def load_reference(path: Path):
    ref = {}
    for ln in path.read_text().splitlines():
        r = json.loads(ln)
        ref[r["frame"]] = {"person": [p["box"] for p in r["persons"]],
                           "vehicle": [v["box"] for v in r["vehicles"] if not v.get("estimated")],
                           "polys": [v.get("polygons") for v in r["vehicles"] if not v.get("estimated")]}
    return ref


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True, help="hf-models.json from the setup job (local snapshot paths)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--only", default="", help="comma list of candidate names to run")
    a = ap.parse_args()
    import torch

    torch.set_num_threads(cv2.getNumThreads() or 4)
    hf = {k: v.get("path") for k, v in json.loads(a.hf.read_text()).items()}
    builders = {
        "yolox_l-cv2dnn": lambda: YoloxDnn(a.models / "onnx" / "yolox_l.onnx"),
        "yolox_l-cv2dnn-tiles3": lambda: YoloxDnn(a.models / "onnx" / "yolox_l.onnx", tiles=3),
        "yolox_m-cv2dnn-tiles3": lambda: YoloxDnn(a.models / "onnx" / "yolox_m.onnx", tiles=3),
        "rtdetrv2": lambda: HFDetector("rtdetrv2", hf["PekingU/rtdetr_v2_r50vd"]),
        "rtdetrv2-960": lambda: HFDetector("rtdetrv2", hf["PekingU/rtdetr_v2_r50vd"], 960),
        "omdet": lambda: HFDetector("omdet", hf["omlab/omdet-turbo-swin-tiny-hf"]),
        "gdino": lambda: HFDetector("gdino", hf["IDEA-Research/grounding-dino-tiny"]),
        "owlv2": lambda: HFDetector("owlv2", hf["google/owlv2-base-patch16-ensemble"]),
    }
    seg_builders = {"slimsam77": lambda: SamMasks(hf["Zigeng/SlimSAM-uniform-77"], "slimsam77"),
                    "sam-vit-base": lambda: SamMasks(hf["facebook/sam-vit-base"], "sam-vit-base")}
    only = set(filter(None, a.only.split(",")))
    frames = {}
    for refp in sorted(a.reference.glob("*-detections.jsonl")):
        clip = refp.name.replace("-detections.jsonl", "")
        vid = a.clips / f"{clip}.mp4"
        if not vid.exists():
            continue
        ref = load_reference(refp)
        cand = [f for f, r in sorted(ref.items()) if r["person"] or r["vehicle"]]
        pick = [cand[int(i)] for i in np.linspace(0, len(cand) - 1, min(a.frames, len(cand)))]
        cap = cv2.VideoCapture(str(vid))
        for f in pick:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if ok:
                frames[(clip, f)] = (img, ref[f])
        cap.release()
    if not frames:
        raise SystemExit("no frames: check --clips and --reference")
    report = {"opencv": cv2.__version__, "cv2_file": cv2.__file__, "torch": torch.__version__, "threads": torch.get_num_threads(),
              "frames": len(frames), "clips": sorted({c for c, _ in frames}), "detectors": {}, "segmenters": {}}
    for name, mk in builders.items():
        if only and name not in only:
            continue
        r = {"perClip": {}}
        try:
            t0 = time.time(); det = mk(); r["load_s"] = round(time.time() - t0, 1)
            det.detect(next(iter(frames.values()))[0])  # warm-up
            rows, sweep = [], {}
            for (clip, f), (img, ref) in frames.items():
                t0 = time.time(); d = det.detect(img); dt = time.time() - t0
                pr, pe = recall(ref["person"], [b for b, _ in d["person"]]); vr, ve = recall(ref["vehicle"], [b for b, _ in d["vehicle"]])
                rows.append((clip, dt, pr, pe, vr, ve, len(ref["person"]), len(ref["vehicle"])))
                for th in SWEEP:
                    a_, b_ = recall(ref["person"], [b for b, sc in d["person"] if sc >= th]); c_, e_ = recall(ref["vehicle"], [b for b, sc in d["vehicle"] if sc >= th])
                    sweep.setdefault(th, []).append((a_, b_, c_, e_, len(ref["person"]), len(ref["vehicle"])))
            def agg(rs):
                tot_p = sum(x[6] for x in rs); tot_v = sum(x[7] for x in rs)
                return {"ms": round(1000 * float(np.mean([x[1] for x in rs])), 1),
                        "personRecall": round(sum((x[2] or 0) * x[6] for x in rs) / tot_p, 3) if tot_p else None,
                        "personExtra": round(float(np.mean([x[3] for x in rs])), 2),
                        "forkliftRecall": round(sum((x[4] or 0) * x[7] for x in rs) / tot_v, 3) if tot_v else None,
                        "forkliftExtra": round(float(np.mean([x[5] for x in rs])), 2), "refPersons": tot_p, "refForklifts": tot_v}
            r["overall"] = agg(rows)
            r["sweep"] = {}
            for th, xs in sorted(sweep.items()):
                tp, tv = sum(x[4] for x in xs), sum(x[5] for x in xs)
                r["sweep"][str(th)] = {"personRecall": round(sum((x[0] or 0) * x[4] for x in xs) / tp, 3) if tp else None,
                                       "personExtra": round(float(np.mean([x[1] for x in xs])), 2),
                                       "forkliftRecall": round(sum((x[2] or 0) * x[5] for x in xs) / tv, 3) if tv else None,
                                       "forkliftExtra": round(float(np.mean([x[3] for x in xs])), 2)}
            for c in sorted({x[0] for x in rows}):
                r["perClip"][c] = agg([x for x in rows if x[0] == c])
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"; r["trace"] = traceback.format_exc()[-1500:]
        report["detectors"][name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != "perClip"})[:600], flush=True)
    for name, mk in seg_builders.items():
        if only and name not in only:
            continue
        r = {}
        try:
            seg = mk(); ious, times = [], []
            for (clip, f), (img, ref) in frames.items():
                if not ref["vehicle"]:
                    continue
                t0 = time.time(); ms = seg.masks(img, ref["vehicle"]); times.append(time.time() - t0)
                for m, polys in zip(ms, ref["polys"]):
                    if not polys:
                        continue
                    g = np.zeros(img.shape[:2], np.uint8)
                    cv2.fillPoly(g, [np.asarray(p, np.int32) for p in polys if len(p) >= 3], 1)
                    g = g.astype(bool)
                    ious.append(float((m & g).sum() / max(1, (m | g).sum())))
            if not ious:
                raise RuntimeError("no reference forklift outlines in the sampled frames")
            r = {"ms": round(1000 * float(np.mean(times)), 1), "outlineIoU_mean": round(float(np.mean(ious)), 3),
                 "outlineIoU_p10": round(float(np.percentile(ious, 10)), 3), "n": len(ious)}
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"; r["trace"] = traceback.format_exc()[-1500:]
        report["segmenters"][name] = r
        print(name, json.dumps(r)[:400], flush=True)
    a.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
