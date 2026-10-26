"""Message contracts shared by every Floorsight component.

Every stream (edge, vision, cloud, agent, dashboard, evaluation) talks only in
these shapes. Units: metres on the floor plan, seconds since the start of the
session, pixels in camera images. Floor origin is the plan's bottom-left
corner, x to the right, y up the plan.

Export JSON Schemas with:  python -m floorsight.contracts
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.3"  # 1.3 (20 Oct 2026): optional real-camera floor hazards, floor_warning events, human verdicts

XY = tuple[float, float]


# --------------------------------------------------------------------------- #
# Site description (static)
# --------------------------------------------------------------------------- #


class ZoneKind(str, Enum):
    exit = "exit"
    lane = "lane"
    restricted = "restricted"
    area = "area"


class Zone(BaseModel):
    zone_id: str
    name: str
    kind: ZoneKind
    polygon: list[XY] = Field(description="Floor polygon, metres, counter-clockwise")


class Obstacle(BaseModel):
    """A static volume such as a rack. Blocks camera sight lines."""

    obstacle_id: str
    name: str
    polygon: list[XY]
    height_m: float


class FloorMarker(BaseModel):
    """A printed ArUco marker lying flat on the floor."""

    marker_id: int
    dictionary: str = "DICT_4X4_50"
    center: XY
    size_m: float
    rotation_deg: float = 0.0


class CameraInfo(BaseModel):
    camera_id: str
    width: int
    height: int
    fps: float
    K: list[list[float]] = Field(description="3x3 intrinsic matrix")
    dist: list[float] = Field(default_factory=lambda: [0.0] * 5)
    nominal_position: tuple[float, float, float] | None = Field(
        default=None, description="Rough mounting position (x, y, height) in metres"
    )


class SitePlan(BaseModel):
    schema_version: str = SCHEMA_VERSION
    site_id: str
    name: str
    width_m: float
    depth_m: float
    zones: list[Zone]
    obstacles: list[Obstacle]
    markers: list[FloorMarker]
    cameras: list[CameraInfo]
    floor_stage: bool = Field(
        default=False,
        description="True when the cameras come with the real-camera floor stage (zones, warnings) in WorldState.hazards",
    )


# --------------------------------------------------------------------------- #
# Edge -> vision
# --------------------------------------------------------------------------- #


class FrameMeta(BaseModel):
    """Sent alongside every JPEG frame leaving the edge device."""

    camera_id: str
    frame_index: int
    t: float = Field(description="Capture time, seconds since session start")
    width: int
    height: int
    faces_blurred: int = Field(description="Faces blurred on the device before upload")
    blur_applied: bool = True


# --------------------------------------------------------------------------- #
# Vision outputs
# --------------------------------------------------------------------------- #


class DetectionKind(str, Enum):
    moving = "moving"
    static = "static"


class Detection(BaseModel):
    """One object seen by one camera, already projected to the floor."""

    camera_id: str
    t: float
    bbox: tuple[int, int, int, int] = Field(description="x, y, w, h in pixels")
    foot_px: XY
    floor_xy: XY
    floor_sigma_m: float
    footprint_m: float = Field(description="Estimated width on the floor, metres")
    kind: DetectionKind
    score: float


class TrackClass(str, Enum):
    person = "person"
    forklift = "forklift"
    object = "object"
    unknown = "unknown"


class TrackState(str, Enum):
    seen = "seen"
    estimated = "estimated"  # image evidence exists, but floor position/contact is uncertain
    unseen = "unseen"  # predicted, currently outside any camera's view
    lost = "lost"


class Track(BaseModel):
    track_id: str
    cls: TrackClass
    x: float
    y: float
    vx: float
    vy: float
    sigma_m: float
    camera_ids: list[str]
    first_seen_t: float
    last_seen_t: float
    state: TrackState


class CameraStatus(str, Enum):
    ok = "ok"
    moved = "moved"
    repairing = "repairing"
    needs_human = "needs_human"
    offline = "offline"


class CameraHealth(BaseModel):
    camera_id: str
    status: CameraStatus
    fps: float
    last_frame_t: float
    markers_visible: int
    calib_error_m: float | None = Field(description="Median marker re-projection error on the floor; null when not measured")


class CoverageMap(BaseModel):
    """How many cameras can see each floor cell (0 = blind spot)."""

    cell_m: float
    cols: int
    rows: int
    counts: list[int] = Field(description="Row-major, row 0 at y=0")
    version: int


class FloorAlert(BaseModel):
    """One person's floor assessment against one forklift, from the real-camera floor stage."""

    person_id: str
    vehicle_id: str | None
    level: Literal["warning", "danger"] = Field(
        description="warning: will come within 0.8 m in the next 3 s if the forklift keeps its course; danger: in its stopping or tail-swing zone, in its blind zone while it drives load-first, or within 0.8 m"
    )
    confirmed: bool = Field(description="The condition held for 3 frames in a row")
    streak: int = 0
    clearance_m: dict | None = Field(default=None, description="{median, p5, p95} floor clearance now")
    pass_m: dict | None = Field(default=None, description="Closest the forklift will come in the next 3 s if it keeps its course")
    pass_t_s: float | None = None
    ttc_s: dict | None = Field(default=None, description="Time until within 0.8 m, {median, p5, p95}")
    p_warn: float = 0.0
    p_path: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    vehicle_moving: bool = False
    cut_by_frame: bool = False


class FloorHazards(BaseModel):
    """Per-tick output of the real-camera floor stage (empty for sites without it)."""

    source: str = ""
    zones: dict[str, list[list[XY]]] = Field(
        default_factory=dict, description="'Likely' outlines in floor metres: stop, tail, blind, blindTrailing, warn"
    )
    footprints: dict[str, list[XY]] = Field(default_factory=dict, description="vehicle track id -> floor footprint polygon")
    notes: dict[str, list[str]] = Field(default_factory=dict, description="vehicle track id -> labels (speed, stopping distance, load, turn)")
    alerts: list[FloorAlert] = Field(default_factory=list)


class WorldState(BaseModel):
    """Broadcast to the dashboard and the agent several times per second."""

    schema_version: str = SCHEMA_VERSION
    t: float
    tracks: list[Track]
    cameras: list[CameraHealth]
    coverage_version: int
    pipeline_latency_ms: float
    hazards: FloorHazards | None = None
    interlock: dict | None = Field(default=None, description="Forklift interlock: each vehicle's state (normal/slow/stop) and the latest automatic commands")


# --------------------------------------------------------------------------- #
# Rules, events, agent
# --------------------------------------------------------------------------- #


class RuleType(str, Enum):
    blocked = "blocked"  # static object inside a zone for longer than `seconds`
    dwell = "dwell"  # a tracked class inside a zone for longer than `seconds`
    proximity = "proximity"  # two classes closer than `distance_m` while closing
    unseen = "unseen"  # a track predicted inside a zone while no camera sees it
    floor_zone = "floor_zone"  # a confirmed floor warning/danger from the real-camera floor stage (WorldState.hazards)


class Rule(BaseModel):
    rule_id: str
    text: str = Field(description="Plain-language form shown to the user")
    type: RuleType
    zone_id: str | None = None
    classes: list[TrackClass] = Field(default_factory=list)
    seconds: float | None = None
    distance_m: float | None = None
    closing_mps: float | None = None
    enabled: bool = True
    source: Literal["default", "manual", "language", "agent"] = "default"
    confirmed: bool = True


class EventType(str, Enum):
    blocked_exit = "blocked_exit"
    near_miss = "near_miss"
    camera_moved = "camera_moved"
    unseen_in_zone = "unseen_in_zone"
    dwell = "dwell"
    floor_warning = "floor_warning"


class EventStatus(str, Enum):
    candidate = "candidate"
    verifying = "verifying"
    confirmed = "confirmed"
    rejected = "rejected"
    escalated = "escalated"
    resolved = "resolved"


class Evidence(BaseModel):
    kind: Literal["crop", "second_view", "history", "calibration", "trajectory", "replay", "redetect"]
    camera_id: str | None = None
    t: float
    supports: bool | None = Field(description="True supports the event, False contradicts it")
    score: float | None = None
    image_ref: str | None = Field(default=None, description="Blurred evidence image id")
    note: str


class Event(BaseModel):
    event_id: str
    type: EventType
    rule_id: str | None
    t_start: float
    t_detect: float
    location: XY
    zone_id: str | None = None
    track_ids: list[str] = Field(default_factory=list)
    camera_ids: list[str] = Field(default_factory=list)
    severity: Literal["info", "warning", "critical"]
    status: EventStatus
    summary: str
    evidence: list[Evidence] = Field(default_factory=list)


class AgentStep(BaseModel):
    """One line of the agent's decision trace. Judges read these."""

    event_id: str
    step: int
    t: float
    tool: str
    tool_input: dict
    tool_output: dict
    decision: str
    reason: str
    mode: Literal["local", "bedrock"]


class ApprovalStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    expired = "expired"


class ApprovalRequest(BaseModel):
    approval_id: str
    event_id: str | None
    action: Literal["escalate", "create_rule", "recalibrate", "dispatch_check", "confirm_rule", "confirm_incident"]
    summary: str
    payload: dict = Field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.pending
    created_t: float
    decided_t: float | None = None


class Alert(BaseModel):
    alert_id: str
    event_id: str
    t: float
    channel: Literal["dashboard", "sns"]
    audience: Literal["floor_lead", "manager"]
    message: str


# --------------------------------------------------------------------------- #
# Live channel envelope
# --------------------------------------------------------------------------- #


class LiveMessage(BaseModel):
    """Everything on the dashboard websocket is wrapped in this."""

    kind: Literal["world", "event", "agent_step", "approval", "alert", "rule", "coverage"]
    data: dict


ALL_MODELS = [
    SitePlan,
    FloorAlert,
    FloorHazards,
    FrameMeta,
    Detection,
    Track,
    CameraHealth,
    CoverageMap,
    WorldState,
    Rule,
    Event,
    AgentStep,
    ApprovalRequest,
    Alert,
    LiveMessage,
]


def export_schemas(out_dir: str) -> list[str]:
    import json
    import os

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for model in ALL_MODELS:
        path = os.path.join(out_dir, f"{model.__name__}.schema.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model.model_json_schema(), f, indent=2)
        written.append(path)
    return written


if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in export_schemas(os.path.join(here, "contracts", "schemas")):
        print(p)
