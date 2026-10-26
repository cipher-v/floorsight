"""Local evidence store (S3 replaces it on AWS; same interface)."""

from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np


class LocalEvidenceStore:
    """Writes blurred evidence JPEGs to disk and deletes them after `retention_s`."""

    def __init__(self, root: str, retention_s: float = 24 * 3600, jpeg_quality: int = 85):
        self.root = root
        self.retention_s = retention_s
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self._n = 0
        os.makedirs(root, exist_ok=True)

    def put(self, image: np.ndarray, label: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)[:40]
        with self._lock:
            self._n += 1
            ref = f"{int(time.time() * 1000)}-{self._n:05d}-{safe}"
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if ok:
            with open(os.path.join(self.root, ref + ".jpg"), "wb") as f:
                f.write(buf.tobytes())
        return ref

    def get(self, ref: str) -> bytes | None:
        if not all(ch.isalnum() or ch in "-_" for ch in ref):
            return None
        path = os.path.join(self.root, ref + ".jpg")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def purge_expired(self) -> int:
        """Privacy: remove evidence older than the retention window. Returns files removed."""
        cutoff = time.time() - self.retention_s
        removed = 0
        for name in os.listdir(self.root):
            path = os.path.join(self.root, name)
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        return removed
