"""Synchronised multi-camera video reader that yields FramePackets (stands in for the edge devices)."""

from __future__ import annotations

import os
from typing import Iterator

import cv2

from floorsight.contracts import FrameMeta
from floorsight.interfaces import FramePacket
from floorsight.site import load_site


class VideoFeed:
    """Reads <sim_dir>/<camera_id>.mp4 for every camera in site.json, frame-synchronised by index."""

    def __init__(self, sim_dir: str, max_frames: int | None = None):
        self.sim_dir = sim_dir
        self.site = load_site(os.path.join(sim_dir, "site.json"))
        self.max_frames = max_frames
        self.decode_s = 0.0

    def __iter__(self) -> Iterator[dict[str, FramePacket]]:
        import time

        caps = {c.camera_id: cv2.VideoCapture(os.path.join(self.sim_dir, f"{c.camera_id}.mp4")) for c in self.site.cameras}
        fps = {c.camera_id: c.fps for c in self.site.cameras}
        i = 0
        try:
            while self.max_frames is None or i < self.max_frames:
                t0 = time.perf_counter()
                frames = {}
                for cid, cap in caps.items():
                    ok, img = cap.read()
                    if not ok:
                        return
                    frames[cid] = img
                self.decode_s += time.perf_counter() - t0
                yield {
                    cid: FramePacket(
                        FrameMeta(camera_id=cid, frame_index=i, t=round(i / fps[cid], 3), width=img.shape[1], height=img.shape[0], faces_blurred=0),
                        img,
                    )
                    for cid, img in frames.items()
                }
                i += 1
        finally:
            for cap in caps.values():
                cap.release()
