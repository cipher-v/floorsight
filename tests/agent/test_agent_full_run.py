"""Full 180 s shift over FakeFeed/FakeVision: the scripted incidents end the right way."""

import json
import os

from floorsight.agent.demo import run


def _events(agent, etype):
    return [c for c in agent.cases.values() if c.event.type.value == etype]


def _steps(agent, event_id):
    return [s for s in agent.trace if s.event_id == event_id]


def _alerts(agent, event_id):
    return [a for a in agent.alerts if a.event_id == event_id]


def _approvals(agent, event_id, action=None):
    return [a for a in agent.approvals.values() if a.event_id == event_id and (action is None or a.action == action)]


def test_glare_is_rejected_not_alerted(full_run):
    agent, _ = full_run
    glare = [c for c in _events(agent, "blocked_exit") if 28 <= c.event.t_detect <= 56]
    assert len(glare) == 1
    ev = glare[0].event
    assert ev.status.value == "rejected"
    assert ev.camera_ids == ["cam3"]
    assert not _alerts(agent, ev.event_id)
    tools = [s.tool for s in _steps(agent, ev.event_id)]
    assert tools == ["sharper_crop", "second_view", "reject_event"]
    sv = [e for e in ev.evidence if e.kind == "second_view"]
    assert sv and all(e.supports is False for e in sv)
    # the spot is remembered: cam3-only detections there now need 30 s
    assert len(agent.engine.spots) == 1
    assert agent.engine.spot_penalty("exit_b", ev.location, ["cam3"]) == 10.0
    assert agent.engine.spot_penalty("exit_b", ev.location, ["cam2", "cam3"]) == 0.0


def test_exit_b_confirmed_alerted_escalated_resolved(full_run):
    agent, _ = full_run
    real = [c for c in _events(agent, "blocked_exit") if c.event.t_detect > 70]
    assert len(real) == 1
    case = real[0]
    ev = case.event
    assert 94.0 <= ev.t_detect <= 97.0
    assert ev.t_start == 75.0
    confirm = next(s for s in _steps(agent, ev.event_id) if s.tool == "confirm_event")
    assert 94.0 <= confirm.t <= 97.0
    # a second camera supported it before confirming
    order = [s.tool for s in _steps(agent, ev.event_id)]
    assert order.index("second_view") < order.index("confirm_event")
    alerts = _alerts(agent, ev.event_id)
    assert [a.audience for a in alerts] == ["floor_lead", "manager"]
    assert 94.0 <= alerts[0].t <= 97.0
    esc = _approvals(agent, ev.event_id, "escalate")
    assert len(esc) == 1 and 124.0 <= esc[0].created_t <= 127.0
    assert esc[0].status.value == "approved"
    assert alerts[1].t >= esc[0].created_t
    assert any(s.decision.endswith("escalated") for s in _steps(agent, ev.event_id))
    assert ev.status.value == "resolved"
    resolve = next(s for s in _steps(agent, ev.event_id) if s.tool == "resolve_event")
    assert 150.0 <= resolve.t <= 157.0


def test_near_misses_confirmed_and_slow_zone_proposed_on_third(full_run, sim_dir):
    agent, _ = full_run
    with open(os.path.join(sim_dir, "truth_events.json"), encoding="utf-8") as f:
        truth = [e for e in json.load(f)["events"] if e["type"] == "near_miss"]
    nm = [c for c in _events(agent, "near_miss") if c.rule and c.rule.rule_id == "R-near-miss"]
    assert len(nm) == len(truth) == 3
    for c, tr in zip(sorted(nm, key=lambda c: c.event.t_detect), sorted(truth, key=lambda e: e["t_start"])):
        assert abs(c.event.t_detect - tr["t_start"]) <= 1.5
        assert c.was_confirmed
        assert [a.audience for a in _alerts(agent, c.event.event_id)] == ["floor_lead"]
        assert any(e.kind == "trajectory" and e.supports for e in c.event.evidence)
    first, second, third = sorted(nm, key=lambda c: c.event.t_detect)
    assert not _approvals(agent, first.event.event_id) and not _approvals(agent, second.event.event_id)
    props = _approvals(agent, third.event.event_id, "create_rule")
    assert len(props) == 1
    p = props[0]
    assert p.created_t == third.event.t_detect
    assert p.payload["zone_id"] == "aisle4_crossing"
    assert "Slow zone at Aisle 4 crossing" in p.summary
    rule = agent.engine.get_rule(p.payload["rule"]["rule_id"])
    assert rule is not None and rule.source == "agent" and rule.enabled and rule.zone_id == "aisle4_crossing"
    assert rule.distance_m == 5.0


def test_cam4_moved_resolved_by_recalibration(full_run):
    agent, _ = full_run
    cams = _events(agent, "camera_moved")
    assert len(cams) == 1
    ev = cams[0].event
    assert ev.camera_ids == ["cam4"] and 100.0 <= ev.t_detect <= 101.0
    steps = _steps(agent, ev.event_id)
    assert [s.tool for s in steps] == ["recalibrate", "resolve_event"]
    assert steps[0].tool_output["ok"] is True
    assert steps[0].tool_output["calib_error_before_m"] > steps[0].tool_output["calib_error_m"]
    assert ev.status.value == "resolved"
    assert not _approvals(agent, ev.event_id)


def test_cold_room_asks_for_dispatch_check(full_run):
    agent, _ = full_run
    un = _events(agent, "unseen_in_zone")
    assert len(un) == 1
    ev = un[0].event
    assert ev.zone_id == "cold_room" and 27.0 <= ev.t_detect <= 30.0
    steps = _steps(agent, ev.event_id)
    assert steps[0].tool == "cameras_covering" and steps[0].tool_output["cameras"] == []
    aps = _approvals(agent, ev.event_id, "dispatch_check")
    assert len(aps) == 1 and "check the cold room" in aps[0].summary
    alerts = _alerts(agent, ev.event_id)
    assert [a.audience for a in alerts] == ["floor_lead"]
    assert ev.status.value == "resolved"  # p2 walked out ~62 s


def test_every_step_is_local_and_messages_cover_everything(full_run):
    agent, log = full_run
    assert agent.trace and all(s.mode == "local" for s in agent.trace)
    kinds = {m.kind for m in log}
    assert {"event", "agent_step", "approval", "alert", "rule"} <= kinds
    assert sum(m.kind == "agent_step" for m in log) == len(agent.trace)
    assert sum(m.kind == "alert" for m in log) == len(agent.alerts)
    snap = agent.snapshot()
    assert set(snap) >= {"rules", "events", "approvals", "alerts", "trace"}
    json.dumps(snap)  # JSON-safe


def test_replay_is_deterministic(sim_dir, full_run):
    agent, _ = full_run
    again, _ = run(sim_dir, auto_approve=True)
    assert [s.model_dump() for s in again.trace] == [s.model_dump() for s in agent.trace]


def test_shift_report(full_run):
    agent, _ = full_run
    rep = agent.shift_report()
    assert rep["counts_by_type"]["blocked_exit"] == {"total": 2, "confirmed": 1, "rejected": 1, "resolved": 1, "open": 0}
    assert rep["counts_by_type"]["near_miss"]["confirmed"] == 3
    assert rep["exit_b_blocked_minutes"] == 1.25
    assert len(rep["false_alarms_rejected"]) == 1 and rep["false_alarms_rejected"][0]["camera_ids"] == ["cam3"]
    hot = rep["near_miss_hotspots"][0]
    assert hot["zone_id"] == "aisle4_crossing" and hot["count"] == 3 and hot["fix_added"]
    assert rep["approvals"]["asked"] == rep["approvals"]["answered"] == 3
    assert rep["open_items"] == []
    for item in rep["false_alarms_rejected"] + rep["blocked_incidents"]:
        assert "evidence_refs" in item and item["evidence"]
    json.dumps(rep)


def test_without_approvals_nothing_is_escalated(sim_dir):
    agent, _ = run(sim_dir, auto_approve=False)
    assert not any(a.audience == "manager" for a in agent.alerts)
    assert not any(r.source == "agent" for r in agent.engine.rules)
    pending = {a.action for a in agent.approvals.values() if a.status.value == "pending"}
    assert pending == {"create_rule"}
    expired = {a.action for a in agent.approvals.values() if a.status.value == "expired"}
    assert expired == {"dispatch_check", "escalate"}
    cold = next(c for c in agent.cases.values() if c.event.type.value == "unseen_in_zone")
    assert not [a for a in agent.alerts if a.event_id == cold.event.event_id]
    # a human approves the slow zone after the shift: the rule is still added
    ap = next(a for a in agent.approvals.values() if a.action == "create_rule")
    msgs = agent.decide(ap.approval_id, True, 180.0)
    assert any(m.kind == "rule" and m.data["source"] == "agent" for m in msgs)
    rep = agent.shift_report()
    assert rep["approvals"]["pending"] == 0
    assert not any(o["kind"] == "approval" for o in rep["open_items"])
    assert rep["approvals"]["expired"] == 2
    assert rep["approvals"]["answered"] == 1


def test_local_policy_never_trips_the_guard(full_run):
    agent, _ = full_run
    assert not [s for s in agent.trace if s.decision == "refused by guard"]
