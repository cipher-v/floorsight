"""On-device privacy filter: faces are destroyed before a frame leaves the edge.

    from floorsight.edge.privacy import blur_faces
    safe, n = blur_faces(frame_bgr)

Detection uses OpenCV's YuNet face detector (cv2.FaceDetectorYN, model from the
official opencv_zoo repo, see models/THIRD_PARTY.md). The detector input is
resized to every frame's own size, so no face is lost to downscaling.

The blur is deliberately irreversible: each padded face box is first pixelated
(downsampled to a handful of blocks, which throws the detail away rather than
hiding it) and then Gaussian-blurred so block edges carry no structure either.
A plain Gaussian blur alone can be partially deconvolved; pixelation cannot.

Overhead warehouse cameras often see people from above, where faces are too
small or too oblique for any face detector. `blur_people_heads` is the fallback
hook: give it person boxes from the vision layer and it destroys the top of
each box (where the head is) with the same filter.
"""

from __future__ import annotations

import os
import threading

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL = os.path.join(_REPO_ROOT, "models", "face_detection_yunet_2023mar.onnx")

SCORE_THRESHOLD = 0.6  # YuNet confidence; lower = more blurring of doubtful faces
NMS_THRESHOLD = 0.3
TOP_K = 5000
PAD_FRACTION = 0.5  # grow each face box by this fraction of its size on every side (covers hair/ears)
PIXEL_BLOCKS = 5  # a padded face becomes at most this many blocks across

Box = tuple[int, int, int, int]  # x, y, w, h in pixels


class FaceDetector:
    """Thread-safe wrapper: FaceDetectorYN keeps per-size state, so calls are serialised."""

    def __init__(self, model_path: str | None = None, score_threshold: float = SCORE_THRESHOLD):
        path = model_path or os.environ.get("FLOORSIGHT_YUNET_MODEL") or DEFAULT_MODEL
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"YuNet model not found at {path}. Download it as described in models/THIRD_PARTY.md"
            )
        self.model_path = path
        self._det = cv2.FaceDetectorYN.create(path, "", (320, 320), score_threshold, NMS_THRESHOLD, TOP_K)
        self._size: tuple[int, int] | None = None
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray) -> list[tuple[Box, float]]:
        """Face boxes (x, y, w, h) with scores, in the image's own pixel coordinates."""
        if image is None or image.size == 0:
            return []
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        h, w = image.shape[:2]
        with self._lock:
            if self._size != (w, h):
                self._det.setInputSize((w, h))
                self._size = (w, h)
            _, faces = self._det.detect(image)
        if faces is None:
            return []
        out = []
        for f in faces:
            x, y, bw, bh = (int(round(v)) for v in f[:4])
            out.append(((x, y, bw, bh), float(f[-1])))
        return out


_local = threading.local()


def get_detector() -> FaceDetector:
    """One detector per thread, loaded on first use.

    Each camera runs on its own edge device in production; per-thread detectors
    mirror that and stop cameras queueing behind one shared model.
    """
    det = getattr(_local, "detector", None)
    if det is None:
        det = FaceDetector()
        _local.detector = det
    return det


def _pad_clip(box: Box, pad: float, w: int, h: int) -> Box | None:
    x, y, bw, bh = box
    px, py = int(round(bw * pad)), int(round(bh * pad))
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(w, x + bw + px), min(h, y + bh + py)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1 - x0, y1 - y0


def destroy_region(image: np.ndarray, box: Box, blocks: int = PIXEL_BLOCKS) -> None:
    """Irreversibly anonymise one region in place: pixelate, then blur."""
    x, y, w, h = box
    roi = image[y : y + h, x : x + w]
    side = max(w, h)
    sw = max(1, int(round(blocks * w / side)))
    sh = max(1, int(round(blocks * h / side)))
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    k = max(3, (min(w, h) // 4) | 1)  # odd kernel, about a quarter of the box
    roi[:] = cv2.GaussianBlur(big, (k, k), 0)


def blur_boxes(image: np.ndarray, boxes: list[Box], pad: float = PAD_FRACTION) -> tuple[np.ndarray, int]:
    out = image.copy()
    h, w = out.shape[:2]
    n = 0
    for b in boxes:
        r = _pad_clip(b, pad, w, h)
        if r is not None:
            destroy_region(out, r)
            n += 1
    return out, n


def blur_faces(image: np.ndarray, detector: FaceDetector | None = None, pad: float = PAD_FRACTION) -> tuple[np.ndarray, int]:
    """Return (copy of image with every detected face destroyed, number of faces)."""
    det = detector or get_detector()
    faces = det.detect(image)
    return blur_boxes(image, [b for b, _ in faces], pad=pad)


def blur_people_heads(image: np.ndarray, boxes: list[Box], head_fraction: float = 0.28) -> tuple[np.ndarray, int]:
    """Fallback for top-down views: destroy the head part (top `head_fraction`) of each person box.

    `boxes` are person bounding boxes (x, y, w, h) in pixels, e.g. Detection.bbox
    for tracks classed as people. Returns (copy, number of boxes blurred).
    """
    heads = []
    for x, y, w, h in boxes:
        hh = max(2, int(round(h * head_fraction)))
        heads.append((x, y, w, hh))
    return blur_boxes(image, heads, pad=0.15)


def privacy_filter(image: np.ndarray, person_boxes: list[Box] | None = None) -> tuple[np.ndarray, int]:
    """Faces first; then, if person boxes are known, heads too. Returns (image, faces_blurred)."""
    out, n = blur_faces(image)
    if person_boxes:
        out, _ = blur_people_heads(out, person_boxes)
    return out, n
