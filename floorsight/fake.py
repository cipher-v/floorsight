"""Fake vision layer driven by simulator ground truth.

Lets the agent, server and dashboard streams run end-to-end before the real
vision pipeline exists. Positions get realistic noise; the cold room is a blind
spot; cam3 reports the scripted sun glare in Exit B as a static object (the
false alarm the agent must reject); cam4 reports "moved" after its knock.

    from floorsight.fake import FakeFeed, FakeVision
    feed = FakeFeed("data/sim/demo_001")
    vision = FakeVision(feed)
    for packets in feed:
        world = vision.process(packets)
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from floorsight.contracts import (
    CameraHealth,
    CameraStatus,
    CoverageMap,
    Evidence,
    FrameMeta,
    Track,
    TrackClass,
    TrackState,
    WorldState,
)
from floorsight.interfaces import FramePacket
from floorsight.site import load_site, point_in_polygon, zone_by_id


class FakeFeed:
    """Yields one dict of FramePackets per frame, with blank 64x36 images."""

    def __init__(self, sim_dir: str):
        self.sim_dir = sim_dir
        self.site = load_site(os.path.join(sim_dir, "site.json"))
        with open(os.path.join(sim_dir, "truth_events.json"), encoding="utf-8") as f:
            self.truth = json.load(f)
        self.frames = []
        with open(os.path.join(sim_dir, "truth_frames.jsonl"), encoding="utf-8") as f:
            for line in f:
                self.frames.append(json.loads(line))
        self._blank = np.zeros((36, 64, 3), np.uint8)

    def __iter__(self):
        for fr in self.frames:
            yield {
                c.camera_id: FramePacket(
                    FrameMeta(camera_id=c.camera_id, frame_index=fr["frame"], t=fr["t"], width=64, height=36, faces_blurred=0),
                    self._blank,
                )
                for c in self.site.cameras
            }

    def __len__(self):
        return len(self.frames)


class FakeVision:
    def __init__(self, feed: FakeFeed, seed: int = 3, noise_m: float = 0.08):
        self.feed = feed
        self.site = feed.site
        self.rng = np.random.default_rng(seed)
        self.noise_m = noise_m
        self._by_frame = {fr["frame"]: fr for fr in feed.frames}
        self._first_seen: dict[str, float] = {}
        ev = feed.truth["events"]
        self._glare = next((e for e in ev if e["type"] == "false_alarm_source"), None)
        self._bump = next((e for e in ev if e["type"] == "camera_moved"), None)
        self._recalibrated: set[str] = set()
        self._cold = zone_by_id(self.site, "cold_room")
        self._t = 0.0
        self._entities: list[dict] = []

    # ---- VisionPipeline -------------------------------------------------- #

    def process(self, packets: dict[str, FramePacket]) -> WorldState:
        meta = next(iter(packets.values())).meta
        t = meta.t
        self._t = t
        fr = self._by_frame.get(meta.frame_index, {"entities": []})
        self._entities = fr["entities"]
        tracks = []
        for e in fr["entities"]:
            hidden = self._cold is not None and point_in_polygon(e["x"], e["y"], self._cold.polygon)
            self._first_seen.setdefault(e["id"], t)
            nx, ny = self.rng.normal(0, self.noise_m, 2)
            tracks.append(
                Track(
                    track_id=f"T-{e['id']}",
                    cls=TrackClass(e["cls"]),
                    x=e["x"] + (0 if hidden else nx),
                    y=e["y"] + (0 if hidden else ny),
                    vx=e["vx"],
                    vy=e["vy"],
                    sigma_m=0.6 if hidden else 0.12,
                    camera_ids=[] if hidden else self._cams_for(e["x"], e["y"]),
                    first_seen_t=self._first_seen[e["id"]],
                    last_seen_t=t,
                    state=TrackState.unseen if hidden else TrackState.seen,
                )
            )
        if self._glare and self._glare["t_start"] <= t < self._glare["t_end"]:
            gx, gy = self._glare["location"]
            self._first_seen.setdefault("glare", t)
            tracks.append(
                Track(
                    track_id="T-glare",
                    cls=TrackClass.object,
                    x=gx,
                    y=gy,
                    vx=0.0,
                    vy=0.0,
                    sigma_m=0.2,
                    camera_ids=["cam3"],
                    first_seen_t=self._first_seen["glare"],
                    last_seen_t=t,
                    state=TrackState.seen,
                )
            )
        cams = []
        for c in self.site.cameras:
            moved = (
                self._bump is not None
                and c.camera_id == self._bump["camera_id"]
                and t >= self._bump["t_start"]
                and c.camera_id not in self._recalibrated
            )
            cams.append(
                CameraHealth(
                    camera_id=c.camera_id,
                    status=CameraStatus.moved if moved else CameraStatus.ok,
                    fps=c.fps,
                    last_frame_t=t,
                    markers_visible=6,
                    calib_error_m=0.35 if moved else 0.03,
                )
            )
        return WorldState(t=t, tracks=tracks, cameras=cams, coverage_version=1, pipeline_latency_ms=12.0)

    # ---- VisionTools ----------------------------------------------------- #

    def _cams_for(self, x, y):
        if x > 13.5 and 6.5 < y < 9.2:
            return ["cam2", "cam3"]
        if 4.0 < y < 6.0:
            return ["cam4", "cam1"]
        return ["cam1", "cam2"]

    def cameras_covering(self, xy):
        if self._cold is not None and point_in_polygon(xy[0], xy[1], self._cold.polygon):
            return []
        return self._cams_for(*xy)

    def second_view(self, xy, exclude, kind):
        out = []
        for cam in self.cameras_covering(xy):
            if cam in exclude:
                continue
            present = any(
                math.dist((e["x"], e["y"]), xy) < 0.8 and (e["cls"] == "object" if kind == "static" else e["cls"] != "object")
                for e in self._entities
            )
            out.append(
                Evidence(
                    kind="second_view",
                    camera_id=cam,
                    t=self._t,
                    supports=present,
                    score=0.9 if present else 0.05,
                    note=f"{cam} {'sees' if present else 'does not see'} a {kind} object at ({xy[0]:.1f}, {xy[1]:.1f})",
                )
            )
        return out

    def sharper_crop(self, camera_id, xy, radius_m=1.0):
        return Evidence(kind="crop", camera_id=camera_id, t=self._t, supports=None, score=None, note="fake crop")

    def recalibrate(self, camera_id):
        self._recalibrated.add(camera_id)
        return {"ok": True, "calib_error_m": 0.04, "markers_used": 5}

    def coverage(self):
        cell = 0.5
        cols, rows = int(self.site.width_m / cell), int(self.site.depth_m / cell)
        counts = []
        for r in range(rows):
            for c in range(cols):
                x, y = (c + 0.5) * cell, (r + 0.5) * cell
                counts.append(len(self.cameras_covering((x, y))))
        return CoverageMap(cell_m=cell, cols=cols, rows=rows, counts=counts, version=1)
