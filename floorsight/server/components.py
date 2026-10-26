"""Builds the swappable parts (vision, agent, face blur) with graceful fallbacks.

The real parts live in other streams and may not exist yet:
    floorsight.vision.pipeline.RealVision(site, evidence_store)
    floorsight.agent.runtime.FloorsightAgent(site, vision, notifier=None)
    floorsight.edge.privacy.blur_faces(image) -> (image, n_faces)
They are imported lazily. When missing, the server falls back (FakeVision /
NullVision, StubAgent, no blur) and reports a visible warning instead of failing.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
from collections.abc import Callable

import cv2
import numpy as np

from floorsight.contracts import (
    CameraHealth,
    CameraStatus,
    CoverageMap,
    LiveMessage,
    SitePlan,
    WorldState,
)
from floorsight.interfaces import FramePacket

log = logging.getLogger("floorsight.server.components")

BlurFn = Callable[[np.ndarray], tuple[np.ndarray, int]]


# --------------------------------------------------------------------------- #
# Fallbacks
# --------------------------------------------------------------------------- #


class StubAgent:
    """No-op AgentRuntime: keeps the app running before the real agent exists."""

    name = "stub"

    def __init__(self, site: SitePlan | None = None, vision=None, notifier=None):
        self.site = site

    def on_world(self, world: WorldState) -> list[LiveMessage]:
        return []

    def add_rule_from_text(self, text: str, t: float) -> list[LiveMessage]:
        return []

    def decide(self, approval_id: str, approved: bool, t: float) -> list[LiveMessage]:
        return []

    def snapshot(self) -> dict:
        return {"rules": [], "events": [], "approvals": [], "alerts": [], "trace": []}

    def shift_report(self) -> dict:
        return {"note": "Agent not connected (stub mode): no events were analysed."}


class NullVision:
    """VisionPipeline that sees nothing; used with live cameras when real vision is missing."""

    def __init__(self, site: SitePlan):
        self.site = site
        self._t = 0.0
        self._last_frame_t: dict[str, float] = {}

    def process(self, packets: dict[str, FramePacket]) -> WorldState:
        for camera_id, packet in packets.items():
            self._last_frame_t[camera_id] = max(packet.meta.t, self._last_frame_t.get(camera_id, packet.meta.t))
        t = max([self._t, *(p.meta.t for p in packets.values())])
        self._t = t
        cams = [
            CameraHealth(
                camera_id=c.camera_id,
                status=CameraStatus.needs_human if c.camera_id in packets else CameraStatus.offline,
                fps=0.0,  # No perception runs; configured capture rate is not measured throughput.
                last_frame_t=self._last_frame_t.get(c.camera_id, 0.0),
                markers_visible=0,
                calib_error_m=None,
            )
            for c in self.site.cameras
        ]
        return WorldState(t=t, tracks=[], cameras=cams, coverage_version=0, pipeline_latency_ms=0.0)

    def cameras_covering(self, xy):
        return []

    def second_view(self, xy, exclude, kind):
        return []

    def sharper_crop(self, camera_id, xy, radius_m=1.0):
        from floorsight.contracts import Evidence

        return Evidence(kind="crop", camera_id=camera_id, t=self._t, supports=None, note="vision not connected")

    def recalibrate(self, camera_id):
        return {"ok": False, "calib_error_m": None, "markers_used": 0}

    def coverage(self) -> CoverageMap:
        cell = 0.5
        cols, rows = int(math.ceil(self.site.width_m / cell)), int(math.ceil(self.site.depth_m / cell))
        return CoverageMap(cell_m=cell, cols=cols, rows=rows, counts=[0] * (cols * rows), version=0)


# --------------------------------------------------------------------------- #
# Lazy imports of the real parts
# --------------------------------------------------------------------------- #


def _try_import(module: str, attr: str):
    try:
        mod = importlib.import_module(module)
        return getattr(mod, attr), None
    except Exception as exc:  # ImportError, or the module failing on import
        return None, f"{module}.{attr} unavailable ({type(exc).__name__}: {exc})"


def load_blur() -> tuple[BlurFn | None, str | None]:
    """Returns (blur_faces, warning)."""
    return _try_import("floorsight.edge.privacy", "blur_faces")


def load_setup_plates(setup_dir: str | None, site: SitePlan) -> dict:
    """Empty-floor reference images recorded at installation (setup_empty_<camera_id>.png)."""
    plates = {}
    if not setup_dir:
        return plates
    for cam in site.cameras:
        path = os.path.join(setup_dir, f"setup_empty_{cam.camera_id}.png")
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                plates[cam.camera_id] = img
    return plates


def build_vision(kind: str, site: SitePlan, evidence_store, fake_feed=None, setup_dir: str | None = None, replay_dir: str | None = None):
    """Returns (vision, label, warnings). kind: fake | real | replay | null."""
    warnings: list[str] = []
    if kind == "replay":
        from floorsight.realclip import ReplayVision

        return ReplayVision(site, evidence_store, replay_dir), "replay", warnings
    if kind == "real":
        cls, err = _try_import("floorsight.vision.pipeline", "RealVision")
        if cls is not None:
            try:
                vision = cls(site, evidence_store)
                plates = load_setup_plates(setup_dir, site)
                if plates and hasattr(vision, "set_background_plates"):
                    vision.set_background_plates(plates)
                    log.info("empty-floor setup images loaded for %s", ", ".join(sorted(plates)))
                elif not plates:
                    warnings.append("No empty-floor setup images: people standing still at start-up may be missed")
                return vision, "real", warnings
            except Exception as exc:
                err = f"RealVision failed to start ({type(exc).__name__}: {exc})"
        warnings.append(f"Real vision not available, using fallback: {err}")
        log.warning(warnings[-1])
    if kind == "fake" and fake_feed is not None:
        from floorsight.fake import FakeVision

        return FakeVision(fake_feed), "fake", warnings
    if kind == "fake":
        warnings.append("Fake vision needs simulator ground truth (truth_events.json); seeing nothing instead")
    return NullVision(site), "none", warnings


def build_agent(kind: str, site: SitePlan, vision, truth: dict | None = None, notifier=None):
    """Returns (agent, label, warnings). kind: stub | real | demo."""
    warnings: list[str] = []
    if kind == "real":
        cls, err = _try_import("floorsight.agent.runtime", "FloorsightAgent")
        if cls is not None:
            try:
                return cls(site, vision, notifier=notifier), "real", warnings
            except Exception as exc:
                err = f"FloorsightAgent failed to start ({type(exc).__name__}: {exc})"
        warnings.append(f"Real agent not available, running without one: {err}")
        log.warning(warnings[-1])
    if kind == "demo":
        if truth is not None:
            from floorsight.server.demo_agent import DemoAgent

            return DemoAgent(site, vision, truth), "demo", warnings
        warnings.append("Demo agent needs simulator truth_events.json; running without an agent")
    return StubAgent(site, vision), "stub", warnings


def load_truth(sim_dir: str | None) -> dict | None:
    if not sim_dir:
        return None
    path = os.path.join(sim_dir, "truth_events.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Camera field-of-view wedges for the floor map
# --------------------------------------------------------------------------- #


def camera_views(site: SitePlan, truth: dict | None = None, vision=None) -> dict[str, dict]:
    """Per camera: {x, y, height_m, yaw_deg, hfov_deg, near_m, far_m, source}.

    The SitePlan has no camera orientation, so the direction comes from (in
    order): an optional ``vision.camera_views()`` hook, the simulator's camera
    poses in truth_events.json, or a guess (facing the middle of the floor).
    Angles: yaw 0 = +x on the plan, counter-clockwise.
    """
    hook = getattr(vision, "camera_views", None)
    if callable(hook):
        try:
            views = hook()
            if isinstance(views, dict) and views:
                return views
        except Exception as exc:
            log.warning("vision.camera_views() failed: %s", exc)
    poses = {c["camera_id"]: c for c in (truth or {}).get("cameras", []) if "R" in c}
    out = {}
    for cam in site.cameras:
        if cam.nominal_position is None:
            continue
        x, y, h = cam.nominal_position
        fx = cam.K[0][0] if cam.K and cam.K[0][0] else cam.width
        fy = cam.K[1][1] if cam.K and cam.K[1][1] else fx
        hfov = math.degrees(2 * math.atan(cam.width / (2 * fx)))
        vhalf = math.atan(cam.height / (2 * fy))
        pose = poses.get(cam.camera_id)
        if pose is not None:
            ax, ay, az = pose["R"][2]  # optical axis in world coordinates
            yaw = math.degrees(math.atan2(ay, ax))
            pitch = math.atan2(-az, math.hypot(ax, ay))  # below horizontal
            source = "sim_pose"
        else:
            cx, cy = site.width_m / 2, site.depth_m / 2
            yaw = math.degrees(math.atan2(cy - y, cx - x))
            pitch = math.radians(35.0)
            source = "guess"
        near = h / math.tan(min(pitch + vhalf, math.radians(89.0)))
        top = pitch - vhalf
        far = 40.0 if top <= math.radians(3.0) else min(40.0, h / math.tan(top))
        out[cam.camera_id] = {
            "x": x,
            "y": y,
            "height_m": h,
            "yaw_deg": round(yaw, 2),
            "hfov_deg": round(hfov, 2),
            "near_m": round(max(0.0, near), 2),
            "far_m": round(far, 2),
            "source": source,
        }
    return out
