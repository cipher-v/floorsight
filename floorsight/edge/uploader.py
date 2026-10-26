"""Edge -> cloud frame uploader (PREPARED: only ever tested against a local stub server).

Wire format, one HTTP request per frame:

    POST {ingest_url}
    Content-Type: image/jpeg
    X-Floorsight-Meta: {"camera_id": ..., "frame_index": ..., ...}   (FrameMeta JSON)
    Authorization: Bearer <token>                                    (if configured)
    body: JPEG bytes of the already-blurred frame

Expected replies: 2xx = stored. 429 / 503 = server busy (Retry-After honoured).
Other 5xx and network errors are retried with exponential backoff; other 4xx
are dropped (retrying a bad request never helps).

Privacy: `submit_raw` runs the face filter before anything is queued, and
`submit` refuses packets whose meta says the blur was not applied. Unblurred
pixels never reach the network code.

Backpressure: frames wait in a small bounded queue. When the link is slower
than the camera, the OLDEST queued frame is dropped so the cloud always gets
the freshest view (a live floor map prefers a current frame over a complete one).
"""

from __future__ import annotations

import collections
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

from floorsight.contracts import FrameMeta
from floorsight.interfaces import FramePacket

log = logging.getLogger(__name__)

META_HEADER = "X-Floorsight-Meta"


@dataclass
class UploaderStats:
    queued: int = 0
    sent: int = 0
    dropped_backpressure: int = 0
    dropped_rejected: int = 0
    dropped_after_retries: int = 0
    retries: int = 0
    last_latency_ms: float = 0.0
    last_error: str = ""
    status_counts: dict = field(default_factory=dict)


class EdgeUploader:
    def __init__(
        self,
        ingest_url: str,
        *,
        max_queue: int = 8,
        jpeg_quality: int = 80,
        timeout_s: float = 5.0,
        max_retries: int = 3,
        backoff_s: float = 0.25,
        max_backoff_s: float = 4.0,
        auth_token: str | None = None,
        workers: int = 1,
    ):
        if not ingest_url.startswith(("http://", "https://")):
            raise ValueError("ingest_url must be http(s)")
        self.url = ingest_url
        self.max_queue = max_queue
        self.jpeg_quality = jpeg_quality
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.max_backoff_s = max_backoff_s
        self.auth_token = auth_token
        self.stats = UploaderStats()
        self._q: collections.deque[tuple[FrameMeta, bytes]] = collections.deque()
        self._cv = threading.Condition()
        self._closing = False
        self._in_flight = 0
        self._busy_until = 0.0
        self._threads = [threading.Thread(target=self._run, name=f"uploader-{i}", daemon=True) for i in range(workers)]
        for t in self._threads:
            t.start()

    # ------------------------------------------------------------------ API

    def submit_raw(self, image: np.ndarray, camera_id: str, frame_index: int, t: float, person_boxes=None) -> FrameMeta:
        """Blur faces on this device, then queue. Returns the FrameMeta that will be sent."""
        from floorsight.edge.privacy import privacy_filter

        safe, n = privacy_filter(image, person_boxes)
        h, w = safe.shape[:2]
        meta = FrameMeta(camera_id=camera_id, frame_index=frame_index, t=t, width=w, height=h, faces_blurred=n, blur_applied=True)
        self.submit(FramePacket(meta, safe))
        return meta

    def submit(self, packet: FramePacket) -> None:
        """Queue an already-blurred frame. Never blocks; drops the oldest frame when full."""
        if not packet.meta.blur_applied:
            raise ValueError("refusing to upload a frame without the privacy blur")
        ok, buf = cv2.imencode(".jpg", packet.image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        with self._cv:
            if self._closing:
                raise RuntimeError("uploader is closed")
            while len(self._q) >= self.max_queue:
                self._q.popleft()
                self.stats.dropped_backpressure += 1
            self._q.append((packet.meta, buf.tobytes()))
            self.stats.queued += 1
            self._cv.notify()

    def pending(self) -> int:
        with self._cv:
            return len(self._q) + self._in_flight

    def flush(self, timeout_s: float = 10.0) -> bool:
        """Wait until everything queued has been sent or dropped."""
        end = time.monotonic() + timeout_s
        with self._cv:
            while self._q or self._in_flight:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(min(left, 0.05))
        return True

    def close(self, timeout_s: float = 10.0) -> None:
        self.flush(timeout_s)
        with self._cv:
            self._closing = True
            self._cv.notify_all()
        for t in self._threads:
            t.join(timeout=1.0)

    # ------------------------------------------------------------- internals

    def _run(self):
        while True:
            with self._cv:
                while not self._q and not self._closing:
                    self._cv.wait()
                if not self._q and self._closing:
                    return
                meta, body = self._q.popleft()
                self._in_flight += 1
            try:
                self._send(meta, body)
            finally:
                with self._cv:
                    self._in_flight -= 1
                    self._cv.notify_all()

    def _send(self, meta: FrameMeta, body: bytes) -> None:
        headers = {"Content-Type": "image/jpeg", META_HEADER: json.dumps(meta.model_dump(mode="json"), separators=(",", ":"))}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        attempt = 0
        while True:
            wait = self._busy_until - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            t0 = time.perf_counter()
            status, retry_after, err = None, None, ""
            try:
                req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    status = resp.status
                    resp.read()
            except urllib.error.HTTPError as e:
                status = e.code
                retry_after = e.headers.get("Retry-After") if e.headers else None
                err = f"HTTP {e.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                err = f"{type(e).__name__}: {e}"
            self.stats.last_latency_ms = (time.perf_counter() - t0) * 1000
            if status is not None:
                self.stats.status_counts[status] = self.stats.status_counts.get(status, 0) + 1
            if status is not None and 200 <= status < 300:
                self.stats.sent += 1
                return
            self.stats.last_error = err
            if status is not None and 400 <= status < 500 and status not in (408, 429):
                self.stats.dropped_rejected += 1
                log.warning("frame %s/%s rejected: %s", meta.camera_id, meta.frame_index, err)
                return
            if status in (429, 503):
                delay = self._parse_retry_after(retry_after)
                self._busy_until = max(self._busy_until, time.monotonic() + delay)
            if attempt >= self.max_retries:
                self.stats.dropped_after_retries += 1
                log.warning("frame %s/%s dropped after %d retries: %s", meta.camera_id, meta.frame_index, attempt, err)
                return
            attempt += 1
            self.stats.retries += 1
            time.sleep(min(self.max_backoff_s, self.backoff_s * (2 ** (attempt - 1))))

    def _parse_retry_after(self, value: str | None) -> float:
        try:
            return min(self.max_backoff_s, max(0.0, float(value)))
        except (TypeError, ValueError):
            return self.backoff_s
