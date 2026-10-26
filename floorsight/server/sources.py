"""Frame sources for the server runtime.

Every source is an iterable of ``dict[camera_id, FramePacket]`` (one synchronised
set of frames per tick) and exposes:

    site        SitePlan (or None for a bare live source)
    fps         nominal frames per second
    realtime    True if the source paces itself (live cameras); False means the
                runtime must pace it at --speed
    has_video   False when images are placeholders (FakeFeed)
    close()

Session time ``t`` always comes from the frames, never the wall clock (except
for live cameras, where capture time is the only clock there is).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Iterator

import cv2
import numpy as np

from floorsight.contracts import FrameMeta, SitePlan
from floorsight.interfaces import FramePacket
from floorsight.site import load_site

log = logging.getLogger("floorsight.server.sources")


class VideoFileSource:
    """Synchronised playback of ``{camera_id}.mp4`` files from a simulator run.

    Frame index ``i`` of every file is the same instant, ``t = i / fps``.
    Stops at the shortest file, or starts again from frame 0 when ``loop``.
    """

    realtime = False
    has_video = True

    def __init__(self, sim_dir: str, loop: bool = False, site: SitePlan | None = None):
        self.sim_dir = sim_dir
        self.loop = loop
        self.site = site or load_site(os.path.join(sim_dir, "site.json"))
        self.paths: dict[str, str] = {}
        missing = []
        for cam in self.site.cameras:
            path = os.path.join(sim_dir, f"{cam.camera_id}.mp4")
            if os.path.exists(path):
                self.paths[cam.camera_id] = path
            else:
                missing.append(cam.camera_id)
        if not self.paths:
            raise FileNotFoundError(f"no camera videos (cam*.mp4) found in {sim_dir}")
        if missing:
            log.warning("no video for %s in %s; those cameras will be absent", missing, sim_dir)
        self.fps = float(self.site.cameras[0].fps if self.site.cameras else 10.0)
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._open()

    def _open(self) -> None:
        self.close()
        for cam_id, path in self.paths.items():
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self.close()
                raise OSError(f"cannot decode {path} (is it still being rendered?)")
            self._caps[cam_id] = cap

    def __iter__(self) -> Iterator[dict[str, FramePacket]]:
        while True:
            index = 0
            while True:
                packets = {}
                for cam_id, cap in self._caps.items():
                    ok, img = cap.read()
                    if not ok or img is None:
                        packets = None
                        break
                    h, w = img.shape[:2]
                    meta = FrameMeta(
                        camera_id=cam_id,
                        frame_index=index,
                        t=round(index / self.fps, 3),
                        width=w,
                        height=h,
                        faces_blurred=0,
                        blur_applied=False,
                    )
                    packets[cam_id] = FramePacket(meta, img)
                if not packets:
                    break
                yield packets
                index += 1
            if not self.loop or index == 0:
                return
            self._open()

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps = {}


class FakeFeedSource:
    """Passthrough around ``floorsight.fake.FakeFeed`` (blank images, truth-driven)."""

    realtime = False
    has_video = False

    def __init__(self, feed, loop: bool = False):
        self.feed = feed
        self.site = feed.site
        self.loop = loop
        self.fps = float(self.site.cameras[0].fps if self.site.cameras else 10.0)

    def __iter__(self) -> Iterator[dict[str, FramePacket]]:
        while True:
            n = 0
            for packets in self.feed:
                n += 1
                yield packets
            if not self.loop or n == 0:
                return

    def close(self) -> None:
        pass


def parse_live_spec(spec: str) -> dict[str, int | str]:
    """``"cam1=0,cam2=rtsp://host/stream"`` -> ``{"cam1": 0, "cam2": "rtsp://host/stream"}``."""
    out: dict[str, int | str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"expected camera_id=index_or_url, got {part!r}")
        cam_id, value = part.split("=", 1)
        value = value.strip()
        out[cam_id.strip()] = int(value) if value.isdigit() else value
    return out


class _CameraReader(threading.Thread):
    """Keeps only the newest frame from one webcam / RTSP stream; reconnects on failure."""

    def __init__(self, camera_id: str, target: int | str, retry_s: float = 2.0):
        super().__init__(name=f"live-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.target = target
        self.retry_s = retry_s
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._stop = threading.Event()
        self.connected = False

    def latest(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._seq, self._frame

    def run(self) -> None:
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.target)
            if not cap.isOpened():
                self.connected = False
                log.warning("live camera %s (%s) not available; retrying", self.camera_id, self.target)
                self._stop.wait(self.retry_s)
                continue
            self.connected = True
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok or img is None:
                    break
                with self._lock:
                    self._frame = img
                    self._seq += 1
            cap.release()
            self.connected = False
            if not self._stop.is_set():
                self._stop.wait(self.retry_s)

    def stop(self) -> None:
        self._stop.set()


class LiveCameraSource:
    """Real webcams or RTSP streams: ``{camera_id: device_index_or_url}``.

    One reader thread per camera keeps the newest frame; this source emits a
    synchronised set at ``fps`` with ``t`` = seconds since the source started.
    Cameras with no frame yet are left out of that tick.
    """

    realtime = True
    has_video = True

    def __init__(self, spec: dict[str, int | str], fps: float = 10.0, site: SitePlan | None = None):
        if not spec:
            raise ValueError("live source needs at least one camera (camera_id=index_or_url)")
        self.spec = spec
        self.fps = fps
        self.site = site
        self._readers = {cam_id: _CameraReader(cam_id, target) for cam_id, target in spec.items()}
        self._started = False
        self._closed = threading.Event()

    def __iter__(self) -> Iterator[dict[str, FramePacket]]:
        if not self._started:
            for r in self._readers.values():
                r.start()
            self._started = True
        t0 = time.perf_counter()
        index = 0
        period = 1.0 / self.fps
        while not self._closed.is_set():
            due = t0 + index * period
            delay = due - time.perf_counter()
            if delay > 0:
                self._closed.wait(delay)
            t = round(time.perf_counter() - t0, 3)
            packets = {}
            for cam_id, reader in self._readers.items():
                _, img = reader.latest()
                if img is None:
                    continue
                h, w = img.shape[:2]
                meta = FrameMeta(
                    camera_id=cam_id, frame_index=index, t=t, width=w, height=h, faces_blurred=0, blur_applied=False
                )
                packets[cam_id] = FramePacket(meta, img.copy())
            index += 1
            if packets:
                yield packets

    def status(self) -> dict[str, bool]:
        return {cam_id: r.connected for cam_id, r in self._readers.items()}

    def close(self) -> None:
        self._closed.set()
        for r in self._readers.values():
            r.stop()
