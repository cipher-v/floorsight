"""Real clips inside the app: replay vision -> rules -> agent second look -> human verdict -> trace on disk.

Uses a tiny synthetic replay folder (downward-looking camera, 12 frames) so it runs anywhere; the real replay folder is
produced on GCP by scripts/export_app_replay.py from the perception stage.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from floorsight.agent.planner import Action, guard
from floorsight.agent.rules import default_rules
from floorsight.agent.runtime import FloorsightAgent
from floorsight.contracts import EventType, FrameMeta
from floorsight.interfaces import FramePacket
from floorsight.realclip import FloorCamera, ReplayVision
from floorsight.site import load_site
from floorsight.storage import LocalEvidenceStore

K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
R = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]  # looking straight down
H = 4.0
OFF = [-5.0, -5.0]  # site = floor - OFF
N = 12


def _alert(confirmed: bool, streak: int, level="warning", cut=False, ttc=1.0):
    return {"person": "T1:P1", "vehicle": "T1:F1", "level": level, "confirmed": confirmed, "streak": streak,
            "clearance": {"median": 1.1, "p5": 0.9, "p95": 1.3}, "pass": {"median": 0.3, "p5": 0.1, "p95": 0.5}, "passT": 1.2,
            "ttc": {"median": ttc, "p5": max(0.0, ttc - 0.2), "p95": ttc + 0.2}, "pWarn": 0.9, "pPath": 0.1, "reasons": [] if level == "warning" else ["closer than 0.8 m"],
            "moving": True, "cut": cut}


def _person_box(xy_site):
    cam = FloorCamera(K, R, H, OFF)
    u, v = cam.project([xy_site[0], xy_site[1], 0.0])[0]
    return [float(u - 30), float(v - 120), float(u + 30), float(v)], [float(u), float(v)]


def make_replay(root: Path, alerts_from: int = 4, jump: bool = False, ttc: float = 1.0, restart: bool = False) -> Path:
    d = root / "replay"
    d.mkdir()
    site = {"schema_version": "1.3", "site_id": "test-real", "name": "test", "width_m": 10.0, "depth_m": 10.0, "zones": [], "obstacles": [], "markers": [],
            "cameras": [{"camera_id": "camA", "width": 1920, "height": 1080, "fps": 25.0, "K": K, "dist": [0.0] * 5, "nominal_position": [5.0, 5.0, H]}],
            "floor_stage": True}
    (d / "site.json").write_text(json.dumps(site))
    rows = []
    for f in range(N):
        px, py = 5.5, 5.0
        fx = 3.0 + 0.08 * f
        heading = 0.0 if not (jump and f == alerts_from) else math.radians(80)
        if restart:  # stopped, then pulls away at 80 deg; the floor stage marks the frame it takes the new direction
            heading = 0.0 if f < alerts_from else math.radians(80)
        box, foot = _person_box((px, py))
        alerts = []
        if f >= alerts_from:
            streak = f - alerts_from + 1
            alerts.append(_alert(streak >= 3, streak, ttc=ttc))
        rows.append({"f": f, "t": round(f / 25.0, 3), "persons": [{"id": "T1:P1", "xy": [px, py], "sigma": 0.08, "seated": True, "box": box, "foot": foot}],
                     "vehicles": [{"id": "T1:F1", "footprint": [[fx - 1.2, 4.4], [fx + 1.2, 4.4], [fx + 1.2, 5.5], [fx - 1.2, 5.5]], "heading": heading,
                                   "speed": [1.8, 2.0, 2.2], "accel": 0.0, "turn": 0.0, "turnSE": 0.05, "cut": False, "est": False,
                                   "box": [700.0, 500.0, 900.0, 640.0], "notes": ["2.0 m/s, stops in 3.0 m (2.4-3.9)"],
                                   "restart": restart and f == alerts_from}],
                     "zones": {"warn": [[[4.0, 4.0], [7.0, 4.0], [7.0, 6.0], [4.0, 6.0]]], "stop": [[[4.5, 4.4], [6.0, 4.4], [6.0, 5.5], [4.5, 5.5]]]},
                     "alerts": alerts})
    (d / "T1.frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    vw = cv2.VideoWriter(str(d / "T1-filtered.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (1920, 1080))
    for f in range(N):
        img = np.full((1080, 1920, 3), 60, np.uint8)
        cv2.putText(img, f"frame {f}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        vw.write(img)
    vw.release()
    playlist = {"camera_id": "camA", "calibration": {"K": K, "R": R, "h": H}, "offset_m": OFF, "frame_size": [1920, 1080],
                "clips": [{"name": "T1", "code": "T1", "fps": 25.0, "frames": N, "offset_frames": 0, "offset_t": 0.0, "video": "T1-filtered.mp4", "records": "T1.frames.jsonl"}]}
    (d / "playlist.json").write_text(json.dumps(playlist))
    return d


def _packets(f: int):
    img = np.full((1080, 1920, 3), 60, np.uint8)
    meta = FrameMeta(camera_id="camA", frame_index=f, t=round(f / 25.0, 3), width=1920, height=1080, faces_blurred=0, blur_applied=True)
    return {"camA": FramePacket(meta, img)}


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOORSIGHT_REDETECT_CMD", raising=False)
    monkeypatch.delenv("FLOORSIGHT_LLM", raising=False)
    d = make_replay(tmp_path)
    site = load_site(str(d / "site.json"))
    vision = ReplayVision(site, LocalEvidenceStore(str(tmp_path / "evidence"), retention_s=3600), str(d))
    agent = FloorsightAgent(site, vision)
    return site, vision, agent


def run_all(vision, agent):
    msgs = []
    for f in range(N):
        msgs += agent.on_world(vision.process(_packets(f)))
    for _ in range(3):  # a few more ticks so an asynchronous second detector pass reports back
        msgs += agent.on_world(vision.process(_packets(N - 1)))
    return msgs


def test_floor_stage_site_uses_floor_rule(rig):
    site, _, _ = rig
    ids = [r.rule_id for r in default_rules(site)]
    assert "R-floor-warning" in ids and "R-near-miss" not in ids


def test_replay_world_has_tracks_and_hazards(rig):
    _, vision, _ = rig
    w = vision.process(_packets(6))
    cls = {t.track_id: t.cls.value for t in w.tracks}
    assert cls == {"T1:P1": "person", "T1:F1": "forklift"}
    assert w.hazards is not None and "warn" in w.hazards.zones and "T1:F1" in w.hazards.footprints
    assert [a.confirmed for a in w.hazards.alerts] == [True]
    views = vision.camera_views()["camA"]
    assert abs(views["x"] - 5.0) < 1e-6 and abs(views["y"] - 5.0) < 1e-6


def test_agent_second_look_then_waits_for_human(rig):
    _, vision, agent = rig
    run_all(vision, agent)
    assert len(agent.cases) == 1
    case = next(iter(agent.cases.values()))
    assert case.event.type == EventType.floor_warning
    tools = [s.tool for s in agent.trace if s.event_id == case.event.event_id]
    assert tools[:6] == ["short_replay", "sharper_crop", "redetect", "redetect_result", "recommend", "request_approval"]
    kinds = [e.kind for e in case.event.evidence]
    assert "replay" in kinds and "crop" in kinds and "redetect" in kinds
    rep = next(e for e in case.event.evidence if e.kind == "replay")
    assert rep.supports is True and rep.image_ref
    assert case.event.status.value == "verifying"
    ap = next(iter(agent.approvals.values()))
    assert ap.action == "confirm_incident" and ap.status.value == "pending"
    assert "Agent recommends CONFIRM" in ap.summary
    assert not agent.alerts  # nothing is sent before a human decides


def test_human_approve_confirms_and_alerts(rig):
    _, vision, agent = rig
    run_all(vision, agent)
    ap = next(iter(agent.approvals.values()))
    msgs = agent.decide(ap.approval_id, True, 1.0, reviewer="pytest reviewer")
    case = next(iter(agent.cases.values()))
    assert case.event.status.value == "confirmed"
    assert [a.audience for a in agent.alerts] == ["floor_lead"]
    hd = [s for s in agent.trace if s.tool == "human_decision"]
    assert hd and hd[0].tool_input["reviewer"] == "pytest reviewer" and hd[0].decision == "approved"
    assert any(m.kind == "alert" for m in msgs)


def test_human_reject_closes_as_false_alarm(rig):
    _, vision, agent = rig
    run_all(vision, agent)
    ap = next(iter(agent.approvals.values()))
    agent.decide(ap.approval_id, False, 1.0, reviewer="pytest reviewer")
    case = next(iter(agent.cases.values()))
    assert case.event.status.value == "rejected"
    assert not agent.alerts


def test_guard_blocks_confirm_and_alert_without_human(rig):
    _, vision, agent = rig
    run_all(vision, agent)
    case = next(iter(agent.cases.values()))
    view = agent._view(case, "new", [])
    ok, why = guard(view, Action("confirm_event"), ["camA"])
    assert not ok and ("human" in why or "evidence" in why)
    ok, why = guard(view, Action("send_alert", {"audience": "floor_lead", "message": "x"}), ["camA"])
    assert not ok


def test_tracking_jump_is_flagged_by_replay(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOORSIGHT_REDETECT_CMD", raising=False)
    d = make_replay(tmp_path, jump=True)
    site = load_site(str(d / "site.json"))
    vision = ReplayVision(site, LocalEvidenceStore(str(tmp_path / "ev"), retention_s=3600), str(d))
    agent = FloorsightAgent(site, vision)
    run_all(vision, agent)
    case = next(iter(agent.cases.values()))
    rep = next(e for e in case.event.evidence if e.kind == "replay")
    assert rep.supports is False and "heading jumped" in rep.note
    ap = next(iter(agent.approvals.values()))
    assert "Agent recommends REJECT" in ap.summary  # the human still decides


def test_a_forklift_pulling_away_in_a_new_direction_is_not_a_tracking_glitch(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOORSIGHT_REDETECT_CMD", raising=False)
    d = make_replay(tmp_path, restart=True)
    site = load_site(str(d / "site.json"))
    vision = ReplayVision(site, LocalEvidenceStore(str(tmp_path / "ev"), retention_s=3600), str(d))
    agent = FloorsightAgent(site, vision)
    run_all(vision, agent)
    case = next(iter(agent.cases.values()))
    rep = next(e for e in case.event.evidence if e.kind == "replay")
    assert "heading jumped" not in rep.note and "pulled away in a new direction" in rep.note
    assert rep.supports is not False


def test_runtime_writes_trace_and_human_decision(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOORSIGHT_REDETECT_CMD", raising=False)
    from floorsight.server.runtime import Runtime, RuntimeConfig

    d = make_replay(tmp_path)
    cfg = RuntimeConfig(source="realclip", sim_dir=str(d), vision="replay", agent="real", runtime_dir=str(tmp_path / "rt"), speed=50.0, max_lag_s=float("inf"))
    rt = Runtime(cfg)
    assert rt.blur_state == "prefiltered"
    rt._run()
    assert rt.mode == "ended"
    lines = [json.loads(x) for x in Path(rt.trace_path).read_text().splitlines()]
    assert any(x["kind"] == "agent_step" and x["data"]["tool"] == "short_replay" for x in lines)
    ap = next(x["data"] for x in lines if x["kind"] == "approval")
    rt.decide(ap["approval_id"], True, reviewer="pytest stand-in")
    lines = [json.loads(x) for x in Path(rt.trace_path).read_text().splitlines()]
    assert any(x["kind"] == "agent_step" and x["data"]["tool"] == "human_decision" and x["data"]["tool_input"]["reviewer"] == "pytest stand-in" for x in lines)
    snap = json.loads(Path(rt.trace_path[: -len(".jsonl")] + "-snapshot.json").read_text())
    assert snap["privacy"]["blur"] == "prefiltered" and snap["privacy"]["positions_only"] is False
    assert any(a["audience"] == "floor_lead" for a in snap["agent"]["alerts"])


def test_decision_card_has_verdict_line_and_three_numbers(rig):
    _, vision, agent = rig
    run_all(vision, agent)
    ap = next(iter(agent.approvals.values()))
    card = ap.payload["card"]
    assert card["verdict"] == "confirm" and card["headline"].startswith("Agent recommends CONFIRM")
    assert [n["label"] for n in card["numbers"]] == ["Floor gap at the warning", "Within 0.8 m", "Closest predicted pass"]
    assert [n["value"] for n in card["numbers"]] == ["1.1 m", "in 1.0 s", "0.30 m (0.10-0.50)"]


def test_rejected_card_names_the_glitch(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOORSIGHT_REDETECT_CMD", raising=False)
    d = make_replay(tmp_path, jump=True)
    site = load_site(str(d / "site.json"))
    agent = FloorsightAgent(site, ReplayVision(site, LocalEvidenceStore(str(tmp_path / "ev"), retention_s=3600), str(d)))
    run_all(agent.vision, agent)
    card = next(iter(agent.approvals.values())).payload["card"]
    assert card["verdict"] == "reject" and "tracking glitch (heading jumped" in card["headline"]


def test_zero_time_to_contact_reads_now(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOORSIGHT_REDETECT_CMD", raising=False)
    d = make_replay(tmp_path, ttc=0.0)
    site = load_site(str(d / "site.json"))
    agent = FloorsightAgent(site, ReplayVision(site, LocalEvidenceStore(str(tmp_path / "ev"), retention_s=3600), str(d)))
    run_all(agent.vision, agent)
    case = next(iter(agent.cases.values()))
    assert "within 0.8 m of T1:P1 now if it keeps its course" in case.event.summary and "0.0 s" not in case.event.summary
    card = next(iter(agent.approvals.values())).payload["card"]
    assert card["numbers"][1]["value"] == "now"


def test_second_detector_label_is_not_repeated():
    from floorsight.agent.planner import floor_recommendation

    note = "second detector pass (raw frame inside the private perception stage) inconclusive: forklift not re-detected"
    view = {"event": {"evidence": [{"kind": "replay", "supports": True, "score": 3.0, "note": "condition held 3 frame(s) in a row (0.12 s)"},
                                   {"kind": "redetect", "supports": None, "note": note}]}}
    verdict, why = floor_recommendation(view)
    assert verdict == "confirm" and why.count("second detector pass") == 1 and why.count("inconclusive") == 1
    view["event"]["evidence"][1] = {"kind": "redetect", "supports": True, "note": "re-detected 1 person(s), 1 forklift(s); re-measured floor gap 1.05 m"}
    assert "second detector pass agrees: re-detected" in floor_recommendation(view)[1]


def test_camera_tile_reports_upstream_blur(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOORSIGHT_REDETECT_CMD", raising=False)
    from floorsight.server.runtime import Runtime, RuntimeConfig

    d = make_replay(tmp_path)
    cfg = RuntimeConfig(source="realclip", sim_dir=str(d), vision="replay", agent="real", runtime_dir=str(tmp_path / "rt"), speed=50.0, max_lag_s=float("inf"))
    rt = Runtime(cfg)
    rt._run()
    assert rt.cam_extra["camA"]["blur_mode"] == "prefiltered"


def test_warning_on_the_last_frames_still_gets_an_approval_card(tmp_path, monkeypatch):
    """The stream ends right after the warning: the runtime drains the agent so the second look still finishes."""
    monkeypatch.delenv("FLOORSIGHT_REDETECT_CMD", raising=False)
    from floorsight.server.runtime import Runtime, RuntimeConfig

    d = make_replay(tmp_path, alerts_from=N - 3)  # confirmed (3 frames in a row) only on the very last frame
    cfg = RuntimeConfig(source="realclip", sim_dir=str(d), vision="replay", agent="real", runtime_dir=str(tmp_path / "rt"), speed=50.0, max_lag_s=float("inf"))
    rt = Runtime(cfg)
    rt._run()
    assert rt.mode == "ended"
    aps = list(rt.agent.approvals.values())
    assert len(aps) == 1 and aps[0].action == "confirm_incident" and aps[0].payload.get("card")
    assert not rt.agent.pending_work()
