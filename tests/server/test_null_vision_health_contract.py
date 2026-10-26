"""Pending regression contract for missing perception. Run on GCP only.

These tests describe the prepared correction. They have not run on the changed
source; no validated fix or test pass is claimed.
"""
import json
from types import SimpleNamespace

from floorsight.contracts import CameraStatus
from floorsight.server.components import NullVision
from floorsight.server.runtime import to_jsonable
from floorsight.site import load_site
from floorsight.formatting import floor_error_text


def test_missing_vision_cannot_report_connected_camera_healthy():
    site = load_site("data/sim/demo_001/site.json")
    camera = site.cameras[0].camera_id
    vision = NullVision(site)
    world = vision.process({camera: SimpleNamespace(meta=SimpleNamespace(t=5.0))})
    health = next(h for h in world.cameras if h.camera_id == camera)
    assert health.status == CameraStatus.needs_human
    assert world.tracks == []  # No fabricated detections or truth fallback.


def test_unavailable_calibration_has_json_null_not_zero_or_nan():
    site = load_site("data/sim/demo_001/site.json")
    vision = NullVision(site)
    payload = to_jsonable(vision.process({}))
    json.dumps(payload, allow_nan=False)
    assert all(h["calib_error_m"] is None for h in payload["cameras"])
    result = vision.recalibrate(site.cameras[0].camera_id)
    assert result["ok"] is False and result["calib_error_m"] is None
    json.dumps(result, allow_nan=False)


def test_no_packets_cannot_report_nominal_camera_rate():
    vision = NullVision(load_site("data/sim/demo_001/site.json"))
    health = vision.process({}).cameras
    assert all(h.status == CameraStatus.offline and h.fps == 0 for h in health)


def test_last_seen_time_belongs_to_each_camera():
    site = load_site("data/sim/demo_001/site.json")
    first, second = [c.camera_id for c in site.cameras[:2]]
    vision = NullVision(site)
    world = vision.process({first: SimpleNamespace(meta=SimpleNamespace(t=5.0)),
                            second: SimpleNamespace(meta=SimpleNamespace(t=8.0))})
    by_id = {h.camera_id: h for h in world.cameras}
    assert by_id[first].last_frame_t == 5.0
    assert by_id[second].last_frame_t == 8.0
    later = vision.process({second: SimpleNamespace(meta=SimpleNamespace(t=9.0))})
    stale = next(h for h in later.cameras if h.camera_id == first)
    assert stale.status == CameraStatus.offline and stale.last_frame_t == 5.0


def test_unknown_floor_error_is_not_formatted_as_measured_zero():
    assert floor_error_text(None) == "unavailable"
    assert floor_error_text(float("nan")) == "unavailable"
    assert floor_error_text(0.0) == "0.00 m"


def test_camera_moved_summary_accepts_unknown_calibration():
    from floorsight.agent.rules import RulesEngine
    from floorsight.contracts import EventType
    site = load_site("data/sim/demo_001/site.json")
    world = NullVision(site).process({})
    world.cameras[0].status = CameraStatus.moved
    new, continuing = RulesEngine(site).update(world)
    incident = next(c for c in new + continuing if c.type == EventType.camera_moved)
    assert "unavailable" in incident.summary
    assert incident.data["calib_error_m"] is None
