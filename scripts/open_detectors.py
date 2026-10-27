"""Apache-2.0 detector stack for the real-clip perception stage (replaces the AGPL-3.0 Ultralytics models).

People:    YOLOX (Megvii, Apache-2.0) COCO detector, run by OpenCV's own DNN module (cv2.dnn), on overlapping
           square tiles plus the full frame so far, small workers still get enough pixels.
Forklifts and carried loads: an open-vocabulary detector (OmDet-Turbo, Grounding DINO or OWLv2; all Apache-2.0,
           through Hugging Face transformers) with the same text prompts the earlier stage used.
Outlines:  box-prompted SAM (Meta, Apache-2.0; or SlimSAM, Apache-2.0) masks -> polygons, because the floor
           footprint and clearance need the vehicle's real outline, not its box.
Faces:     YuNet (OpenCV Zoo, MIT) through cv2.FaceDetectorYN.

Same interface as the old Detectors class: run(frame) -> persons, vehicles (box, conf, polygons, kind), faces,
track_persons, loads. Per-stage wall time is accumulated in .timing for the benchmark (cv2.dnn vs torch).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import cv2
import numpy as np

COCO_PERSON, COCO_CAR, COCO_TRUCK = 0, 2, 7
CONFIRM_SCORE, CONFIRM_IOU = 0.30, 0.50  # open-vocabulary confirmation of a COCO truck/car box as a forklift


def _forward(owner, inputs, **kwargs):
    if getattr(owner, "channels_last", False):
        inputs = dict(inputs)
        pixels = inputs.get("pixel_values")
        if pixels is not None and pixels.ndim == 4:
            inputs["pixel_values"] = pixels.to(memory_format=owner.torch.channels_last)
    return owner.model(**inputs, **kwargs)


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / u) if u > 0 else 0.0


def _nms(boxes, scores, thr=0.5):
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes], [float(s) for s in scores], 0.0, thr)
    return [int(i) for i in np.array(keep).reshape(-1)]


def label_kind(label: str) -> str | None:
    """Map an open-vocabulary label to person / forklift / load. Grounding DINO returns phrases built from the prompt
    tokens ("forklift truck", "forklift industrial vehicle", "stack crates"), so exact prompt matching misses them."""
    t = str(label).lower()
    if "forklift" in t or "industrial vehicle" in t:
        return "forklift"
    if "crate" in t or "pallet" in t:
        return "load"
    if "person" in t:
        return "person"
    return None


class YoloxDnn:
    """YOLOX ONNX (raw head outputs) decoded here; letterboxed BGR 0-255 input, as in the YOLOX ONNX demo."""

    def __init__(self, onnx: Path, size: int = 640, tiles: int = 3, conf: float = 0.10):
        self.net = cv2.dnn.readNet(str(onnx))
        self.size, self.tiles, self.conf = size, tiles, conf
        self.name = f"{Path(onnx).stem} via cv2.dnn ({tiles} tiles + full frame)" if tiles > 1 else f"{Path(onnx).stem} via cv2.dnn"
        grids, strides = [], []
        for s in (8, 16, 32):
            n = size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.stride = np.concatenate(strides).astype(np.float32)

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
        m = conf >= self.conf
        o, cls, conf = o[m], cls[m], conf[m]
        b = np.stack([o[:, 0] - o[:, 2] / 2, o[:, 1] - o[:, 3] / 2, o[:, 0] + o[:, 2] / 2, o[:, 1] + o[:, 3] / 2], 1) / r
        return b, cls, conf

    def detect(self, img):
        """-> {class_id: [(box, conf)]} for person/car/truck, merged over tiles with NMS."""
        h, w = img.shape[:2]
        parts = [(0, img)]
        if self.tiles > 1 and w > h:
            parts += [(int(x), img[:, int(x): int(x) + h]) for x in np.linspace(0, w - h, self.tiles)]
        acc: dict[int, list] = {COCO_PERSON: [], COCO_CAR: [], COCO_TRUCK: []}
        for x0, im in parts:
            b, c, s = self._run(im)
            for bb, cc, ss in zip(b, c, s):
                if int(cc) in acc:
                    acc[int(cc)].append(([float(bb[0] + x0), float(bb[1]), float(bb[2] + x0), float(bb[3])], float(ss)))
        out = {}
        for k, items in acc.items():
            keep = _nms([i[0] for i in items], [i[1] for i in items])
            out[k] = [(np.array(items[j][0], np.float32), items[j][1]) for j in keep]
        return out


class RTDetrCoco:
    """RT-DETRv2 (Apache-2.0, COCO) through transformers; same output shape as YoloxDnn.detect."""

    def __init__(self, path: str, conf: float = 0.10, size: int | None = None):
        import torch
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

        self.torch, self.conf, self.size = torch, conf, size
        self.proc = RTDetrImageProcessor.from_pretrained(path)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(path).eval()
        self.name = "RT-DETRv2 R50 (Apache-2.0)" + (f" at {size}" if size else "")

    def detect(self, img):
        from PIL import Image

        h, w = img.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        kw = {"size": {"height": self.size, "width": self.size}} if self.size else {}
        with self.torch.inference_mode():
            inp = self.proc(images=pil, return_tensors="pt", **kw)
            res = self.proc.post_process_object_detection(_forward(self, inp), target_sizes=self.torch.tensor([(h, w)]), threshold=self.conf)[0]
        out = {COCO_PERSON: [], COCO_CAR: [], COCO_TRUCK: []}
        for b, s, lab in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), res["labels"].cpu().numpy()):
            if int(lab) in out:
                out[int(lab)].append((np.clip(b, 0, [w - 1, h - 1, w - 1, h - 1]).astype(np.float32), float(s)))
        for k, items in out.items():  # DETR boxes rarely overlap, but tiles/duplicates can
            keep = _nms([i[0].tolist() for i in items], [i[1] for i in items])
            out[k] = [items[j] for j in keep]
        return out


class OpenVocab:
    """Open-vocabulary detector through transformers. Returns [(box, score, prompt)] above `threshold`."""

    def __init__(self, kind: str, path: str, threads: int | None = None):
        import torch
        from transformers import AutoProcessor

        if threads:
            torch.set_num_threads(threads)
        self.torch, self.kind = torch, kind
        if kind == "omdet":
            from transformers import OmDetTurboForObjectDetection
            self.proc, self.model = AutoProcessor.from_pretrained(path), OmDetTurboForObjectDetection.from_pretrained(path, low_cpu_mem_usage=False).eval()
            self.name = "OmDet-Turbo Swin-T (Apache-2.0)"
        elif kind == "gdino":
            from transformers import AutoModelForZeroShotObjectDetection
            self.proc, self.model = AutoProcessor.from_pretrained(path), AutoModelForZeroShotObjectDetection.from_pretrained(path).eval()
            self.name = "Grounding DINO tiny (Apache-2.0)"
        elif kind == "owlv2":
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            self.proc, self.model = Owlv2Processor.from_pretrained(path), Owlv2ForObjectDetection.from_pretrained(path).eval()
            self.name = "OWLv2 base (Apache-2.0)"
        else:
            raise ValueError(kind)

    def detect(self, img, prompts: list[str], threshold: float):
        from PIL import Image

        torch = self.torch
        h, w = img.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        with torch.inference_mode():
            if self.kind == "omdet":
                inp = self.proc(images=pil, text=prompts, return_tensors="pt")
                res = self.proc.post_process_grounded_object_detection(_forward(self, inp), text_labels=[prompts], target_sizes=[(h, w)],
                                                                       threshold=threshold, nms_threshold=0.3)[0]
                names = list(res.get("text_labels", res.get("classes", [])))
            elif self.kind == "gdino":
                inp = self.proc(images=pil, text=". ".join(prompts) + ".", return_tensors="pt")
                res = self.proc.post_process_grounded_object_detection(_forward(self, inp), inp.input_ids, threshold=threshold,
                                                                       text_threshold=threshold, target_sizes=[(h, w)])[0]
                names = list(res.get("text_labels", res.get("labels", [])))
            else:
                texts = [[f"a photo of a {p}" for p in prompts]]
                inp = self.proc(text=texts, images=pil, return_tensors="pt")
                s = max(h, w)
                res = self.proc.post_process_grounded_object_detection(outputs=_forward(self, inp), threshold=threshold,
                                                                       target_sizes=torch.tensor([[s, s]]), text_labels=texts)[0]
                names = [str(x).replace("a photo of a ", "") for x in res.get("text_labels", [])]
        out = []
        for b, s, n in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), names):
            b = np.clip(b, 0, [w - 1, h - 1, w - 1, h - 1]).astype(np.float32)
            out.append((b, float(s), str(n)))
        return out


class BoxSegmenter:
    """Box-prompted SAM masks -> largest outer polygon per box (image pixels), like the earlier instance masks."""

    def __init__(self, path: str, name: str):
        import torch
        from transformers import SamModel, SamProcessor

        self.torch, self.name = torch, name
        self.proc, self.model = SamProcessor.from_pretrained(path), SamModel.from_pretrained(path).eval()

    def polygons(self, img, boxes):
        if not boxes:
            return []
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        with self.torch.inference_mode():
            inp = self.proc(pil, input_boxes=[[[float(v) for v in b] for b in boxes]], return_tensors="pt")
            out = _forward(self, inp, multimask_output=False)
            masks = self.proc.image_processor.post_process_masks(out.pred_masks, inp["original_sizes"], inp["reshaped_input_sizes"])[0]
        polys = []
        for m in masks:
            mk = m[0].numpy().astype(np.uint8)
            cs = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            c = max(cs, key=cv2.contourArea) if cs else None
            polys.append(c.reshape(-1, 2).astype(np.float32) if c is not None and len(c) >= 3 else None)
        return polys


class OpenDetectors:
    """Drop-in replacement for the Ultralytics Detectors class of real_clip_perception.py."""

    def __init__(self, models: Path, loads: bool, consts: dict, person_onnx: str = "yolox_l", tiles: int = 3,
                 vocab: str = "omdet", segmenter: str = "sam-vit-base", hf_index: Path | None = None,
                 vehicle_source: str = "vocab", vocab_every: int = 1, privacy_extra: str | None = None,
                 workers: int = 1, torch_layout: str = "contiguous"):
        if workers not in (1, 2):
            raise ValueError("detector workers must be 1 or 2")
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="perception") if workers > 1 else None
        self.c = consts
        self.vehicle_source, self.vocab_every, self._n, self._loads_prev, self._fork_regions = vehicle_source, max(1, vocab_every), 0, [], []
        hf_index = hf_index or (models / "hf-models.json")
        hf = {k: v.get("path") for k, v in json.loads(Path(hf_index).read_text()).items()}
        vocab_ids = {"omdet": "omlab/omdet-turbo-swin-tiny-hf", "gdino": "IDEA-Research/grounding-dino-tiny", "owlv2": "google/owlv2-base-patch16-ensemble"}
        seg_ids = {"sam-vit-base": "facebook/sam-vit-base", "slimsam77": "Zigeng/SlimSAM-uniform-77"}
        if person_onnx.startswith("rtdetrv2"):  # "rtdetrv2" or "rtdetrv2-960"
            size = int(person_onnx.split("-")[1]) if "-" in person_onnx else None
            self.person = RTDetrCoco(hf["PekingU/rtdetr_v2_r50vd"], conf=consts.get("PERSON_MODEL_CONF", consts["PRIVACY_PERSON_CONF"]), size=size)
        else:
            self.person = YoloxDnn(models / "onnx" / f"{person_onnx}.onnx", tiles=tiles, conf=consts.get("PERSON_MODEL_CONF", consts["PRIVACY_PERSON_CONF"]))
        # privacy-only second person source (blur list, never tracks): YOLOX through OpenCV's own DNN on 3 tiles + frame
        self.privacy_extra = YoloxDnn(models / "onnx" / f"{privacy_extra}.onnx", tiles=3, conf=consts["PRIVACY_PERSON_CONF"]) if privacy_extra else None
        self.vocab = OpenVocab(vocab, hf[vocab_ids[vocab]])
        self.seg = BoxSegmenter(hf[seg_ids[segmenter]], segmenter)
        if torch_layout not in ("contiguous", "channels-last"):
            raise ValueError("unknown torch memory layout")
        if torch_layout == "channels-last":
            for detector in (self.person, self.vocab, self.seg):
                if hasattr(detector, "torch"):
                    detector.model.to(memory_format=detector.torch.channels_last)
                    detector.channels_last = True
        self.faces = cv2.FaceDetectorYN.create(str(models / "onnx" / "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.6, 0.3, 5000)
        self.loads = loads
        backend = "cv2.dnn" if isinstance(self.person, YoloxDnn) else "torch"
        self.timing = {"person_s": 0.0, "person_backend": backend, "privacy_extra_cv2dnn_s": 0.0, "vocab_torch_s": 0.0, "segment_torch_s": 0.0,
                       "faces_cv2dnn_s": 0.0, "frames": 0, "workers": workers, "torch_layout": torch_layout}
        every = "every frame" if self.vocab_every == 1 else f"every {self.vocab_every}th frame (loads carried with their forklift in between)"
        fk = {"coco": f"{self.person.name} COCO truck/car boxes",
              "confirmed": f"{self.person.name} COCO truck/car boxes kept only where {self.vocab.name} confirms 'forklift' "
                           f"({every}; confirmed regions follow the forklift in between)",
              "vocab": f"{self.vocab.name}, prompts {consts['FORKLIFT_PROMPTS']}"}[vehicle_source]
        self.notes = [f"People: {self.person.name}, COCO person; privacy recall also uses the open-vocabulary 'person' prompt on its frames.",
                      f"Forklifts: {fk}; outlines: box-prompted {segmenter} (Apache-2.0), every frame.",
                      (f"Carried loads: {self.vocab.name}, prompts {consts['LOAD_PROMPTS']} (conf >= {consts['LOAD_CONF']}), {every}." if loads else "No load detection."),
                      "Faces: YuNet 2023mar (OpenCV Zoo, MIT) via cv2.FaceDetectorYN. No Ultralytics code or weights."]
        if self.privacy_extra is not None:
            self.notes.append(f"Privacy (blur only): also {self.privacy_extra.name} person boxes (YOLOX, Apache-2.0) at conf >= {consts['PRIVACY_PERSON_CONF']}.")

    def _carry_loads(self, vboxes):
        """Between open-vocabulary frames: move each remembered load with the forklift it was touching."""
        out = []
        for vb_prev, lb, lp, ls in self._loads_prev:
            best = max(vboxes, key=lambda v: _iou(v[0], vb_prev), default=None)
            if best is None or _iou(best[0], vb_prev) < 0.3:
                continue
            d = np.array([(best[0][0] + best[0][2] - vb_prev[0] - vb_prev[2]) / 2, (best[0][1] + best[0][3] - vb_prev[1] - vb_prev[3]) / 2], np.float32)
            out.append((lb + np.concatenate([d, d]), ls, lp + d))
        return out

    def run(self, frame: np.ndarray) -> dict:
        c = self.c
        h, w = frame.shape[:2]
        vocab_frame = self.vocab_every <= 1 or self._n % self.vocab_every == 0
        self._n += 1
        want_forks = self.vehicle_source in ("vocab", "confirmed")
        prompts = ["person"] + (list(c["FORKLIFT_PROMPTS"]) if want_forks else []) + (list(c["LOAD_PROMPTS"]) if self.loads else [])
        # Start the longest task first so short privacy tasks cannot delay it.
        tasks = {}
        if vocab_frame:
            tasks["vocab_torch_s"] = (self.vocab.detect, (frame, prompts, min(c["PRIVACY_PERSON_CONF"], c["FORKLIFT_CONF"])))
        tasks["person_s"] = (self.person.detect, (frame,))
        if self.privacy_extra is not None:
            tasks["privacy_extra_cv2dnn_s"] = (self.privacy_extra.detect, (frame,))
        tasks["faces_cv2dnn_s"] = (self._faces, (frame,))
        results = self._run_tasks(tasks)
        coco = results["person_s"]
        extra_people = results.get("privacy_extra_cv2dnn_s", {}).get(COCO_PERSON, [])
        found = results.get("vocab_torch_s", [])
        faces = results["faces_cv2dnn_s"]
        persons = list(coco[COCO_PERSON])
        vboxes, loads_raw = [], []
        coco_v = []
        if self.vehicle_source in ("coco", "confirmed"):  # COCO truck/car boxes from the every-frame model
            vc = c.get("VEHICLE_MODEL_CONF", c["FORKLIFT_CONF"])
            coco_v = [(b, s, "forklift") for k in (COCO_TRUCK, COCO_CAR) for b, s in coco[k] if s >= vc]
            if self.vehicle_source == "coco":
                vboxes = coco_v
        vocab_forks = []
        for b, s, n in found:
            kind = label_kind(n)
            if kind == "person":
                persons.append((b, s))  # second person source, privacy recall only
            elif kind == "forklift" and s >= c["FORKLIFT_CONF"]:
                vocab_forks.append((b, s, "forklift"))
            elif kind == "load" and self.loads and s >= c["LOAD_CONF"]:
                loads_raw.append((b, s))
        if self.vehicle_source == "vocab":
            vboxes = vocab_forks
        elif self.vehicle_source == "confirmed":
            # A COCO truck/car box counts as a forklift only where the open-vocabulary model said "forklift" (on its
            # frames) or where a confirmed forklift was on the previous frame (in between): parked machines that COCO
            # calls car/truck are dropped. Confirmed regions follow their forklift frame to frame.
            # Confirmation needs a confident open-vocabulary "forklift" box on the same object (score >= 0.3, IoU >= 0.5):
            # Grounding DINO also says "forklift" at 0.15-0.39 for racks and shelving, and those boxes must neither
            # confirm a parked machine nor be added (merged with the real truck they made one tall, wrong unit).
            strong = [b for b, s, _ in vocab_forks if s >= CONFIRM_SCORE] if vocab_frame else []
            keep, regions = [], []
            for b, s, kind in coco_v:
                confirmed_now = any(_iou(b, r) >= CONFIRM_IOU for r in strong)
                continues = any(_iou(b, r) >= 0.3 for r, _ in self._fork_regions)  # same forklift one frame later
                if confirmed_now or continues:
                    keep.append((b, s, kind)); regions.append((b, 0))
            # regions whose forklift COCO missed this frame stay for up to 25 frames (the tracker carries the truck)
            regions += [(r, age + 1) for r, age in self._fork_regions if age < 25 and not any(_iou(r, k[0]) >= 0.3 for k in keep)]
            self._fork_regions = regions
            vboxes = keep
        keep = _nms([v[0].tolist() for v in vboxes], [v[1] for v in vboxes], 0.4)
        vboxes = [vboxes[i] for i in keep]
        ts = time.perf_counter()
        polys = self.seg.polygons(frame, [v[0] for v in vboxes] + [l[0] for l in loads_raw])
        self.timing["segment_torch_s"] += time.perf_counter() - ts
        vehicles = [(b, s, p, kind) for (b, s, kind), p in zip(vboxes, polys[: len(vboxes)])]  # one polygon per box, like masks.xy
        if vocab_frame:
            loads = [(b, s, p) for (b, s), p in zip(loads_raw, polys[len(vboxes):]) if p is not None and len(p) >= 3]
            self._loads_prev = []
            for b, s, p in loads:  # remember which forklift each load touches
                v = max(vboxes, key=lambda v: _iou(v[0], b), default=None)
                if v is not None and _iou(v[0], b) > 0.0:
                    self._loads_prev.append((v[0].copy(), b.copy(), p.copy(), s))
        else:
            loads = self._carry_loads(vboxes) if self.loads else []
        tm = self.timing
        tm["frames"] += 1
        tp = [(b, s) for b, s in persons if s >= c["TRACK_PERSON_CONF"]]
        keep = _nms([b.tolist() for b, _ in tp], [s for _, s in tp], 0.5)
        persons = persons + list(extra_people)  # blur list only (after the track candidates were taken)
        return {"persons": persons, "vehicles": vehicles, "faces": faces, "track_persons": [tp[i] for i in keep], "loads": loads}

    def _faces(self, frame):
        h, w = frame.shape[:2]
        self.faces.setInputSize((w, h))
        _, found = self.faces.detect(frame)
        return [] if found is None else [(f[0], f[1], f[0] + f[2], f[1] + f[3]) for f in found]

    @staticmethod
    def _timed(fn, args):
        start = time.perf_counter()
        result = fn(*args)
        return result, time.perf_counter() - start

    def _run_tasks(self, tasks):
        # Each model is called once per frame. Drain ALL workers even if one fails:
        # the caller then withholds this frame, and the next frame cannot race a
        # previous call on the same OpenCV DNN net. No privacy stage is skipped.
        if self._pool is None:
            measured = {name: self._timed(fn, args) for name, (fn, args) in tasks.items()}
        else:
            pending = {name: self._pool.submit(self._timed, fn, args) for name, (fn, args) in tasks.items()}
            try:
                measured = {name: future.result() for name, future in pending.items()}
            finally:
                wait(pending.values())
        for name, (_, elapsed) in measured.items():
            self.timing[name] += elapsed
        return {name: result for name, (result, _) in measured.items()}

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
