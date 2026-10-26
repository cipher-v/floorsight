"""Boundaries between Floorsight components.

Each stream implements one side of these and can be swapped for a fake:
    vision   -> VisionPipeline   (real: floorsight.vision, fake: floorsight.fake.FakeVision)
    agent    -> AgentRuntime     (floorsight.agent)
    storage  -> EvidenceStore    (local disk now, S3 on AWS)
    alerts   -> Notifier         (dashboard now, SNS on AWS)
All times are session seconds taken from the frames, never the wall clock,
so a replay of the same footage produces the same decisions.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import numpy as np

from floorsight.contracts import XY, Alert, CoverageMap, Evidence, FrameMeta, LiveMessage, WorldState


class FramePacket(NamedTuple):
    meta: FrameMeta
    image: np.ndarray  # BGR, faces already blurred on the edge device


class EvidenceStore(Protocol):
    def put(self, image: np.ndarray, label: str) -> str:
        """Store a (blurred) evidence image, return its reference id."""

    def get(self, ref: str) -> bytes | None:
        """JPEG bytes for a reference id."""


class Notifier(Protocol):
    def send(self, alert: Alert) -> None: ...


class VisionTools(Protocol):
    """What the agent may ask the vision layer to do. Every call returns evidence, never a verdict."""

    def cameras_covering(self, xy: XY) -> list[str]:
        """Cameras that currently have an unobstructed view of this floor point."""

    def second_view(self, xy: XY, exclude: list[str], kind: str) -> list[Evidence]:
        """Look at the same floor point from every other covering camera.
        kind: "static" (is there an object lying here?) or "moving" (is someone here?).
        One Evidence per camera; supports=True if that camera also sees it."""

    def sharper_crop(self, camera_id: str, xy: XY, radius_m: float = 1.0) -> Evidence:
        """Full-resolution crop around a floor point from one camera, stored as evidence."""

    def recalibrate(self, camera_id: str) -> dict:
        """Re-fit the camera's floor alignment from visible markers.
        Returns {"ok": bool, "calib_error_m": float | None, "markers_used": int}.
        An unavailable measurement is None, never a fabricated zero."""

    def coverage(self) -> CoverageMap: ...


class VisionPipeline(VisionTools, Protocol):
    def process(self, packets: dict[str, FramePacket]) -> WorldState:
        """Consume one synchronised set of frames (one per camera) and return the fused world."""


class AgentRuntime(Protocol):
    def on_world(self, world: WorldState) -> list[LiveMessage]:
        """Run rules and any due agent work for this tick. Returns messages to broadcast."""

    def add_rule_from_text(self, text: str, t: float) -> list[LiveMessage]: ...

    def decide(self, approval_id: str, approved: bool, t: float) -> list[LiveMessage]: ...

    def snapshot(self) -> dict:
        """{"rules": [...], "events": [...], "approvals": [...], "alerts": [...], "trace": [...]}"""

    def shift_report(self) -> dict: ...
