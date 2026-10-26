"""Plain-language rules: LocalRuleParser and the confirm-rule flow."""

import pytest

from floorsight.agent.language import LocalRuleParser
from floorsight.agent.planner import LocalPlanner
from floorsight.agent.runtime import FloorsightAgent
from floorsight.contracts import Track, TrackClass, TrackState, WorldState


@pytest.fixture()
def parser(site):
    return LocalRuleParser(site)


@pytest.mark.parametrize(
    "text, rtype, zone, seconds, distance",
    [
        ("alert me if anyone stands in the forklift lane for more than 5 seconds", "dwell", "lane_main", 5.0, None),
        ("tell me when Exit B is blocked for 30 seconds", "blocked", "exit_b", 30.0, None),
        ("warn when a person gets within 1 metre of a forklift", "proximity", None, None, 1.0),
        ("alert if someone is in the cold room and we can't see them", "unseen", "cold_room", 10.0, None),
        ("Tell me if exit-b is obstructed for half a minute", "blocked", "exit_b", 30.0, None),
        ("alert if anybody waits in the loop lane for 10s", "dwell", "lane_loop", 10.0, None),
        ("warn if anyone is within 50cm of a fork lift in the aisle 4 crossing", "proximity", "aisle4_crossing", None, 0.5),
        ("let me know when someone is in the coldroom and we cannot see them for 2 minutes", "unseen", "cold_room", 120.0, None),
        ("alert when somebody stays on picking face A for 20 seconds", "dwell", "picking_a", 20.0, None),
    ],
)
def test_parses(parser, text, rtype, zone, seconds, distance):
    r = parser.parse(text)
    assert r.ok, r.message
    rule = r.rule
    assert rule.type.value == rtype
    assert rule.zone_id == zone
    assert rule.seconds == seconds
    assert rule.distance_m == distance
    assert rule.confirmed is False and rule.enabled is False and rule.source == "language"
    assert rule.text == text


def test_fuzzy_zone_names(parser):
    assert parser.parse("tell me when exit b is blokced... I mean blocked for 10 seconds").rule.zone_id == "exit_b"
    assert parser.parse("alert if anyone stands in the forklfit lane (main aisle) for 3 seconds").rule.zone_id == "lane_main"
    # "forklift lane" fits both lanes: best guess first, the other offered as an alternative
    r = parser.parse("alert me if anyone stands in the forklift lane for more than 5 seconds")
    assert r.alternatives == ["lane_loop"]


def test_proximity_class_words_do_not_pick_a_lane(parser):
    assert parser.parse("warn when a person gets within 1 metre of a forklift").rule.zone_id is None


@pytest.mark.parametrize("text", ["make me a sandwich", "", "what's the weather", "alert me when the loading dock is blocked"])
def test_unparseable_gives_clear_message(parser, text):
    r = parser.parse(text)
    assert not r.ok
    assert len(r.message) > 10


def test_add_rule_from_text_needs_confirmation(site):
    agent = FloorsightAgent(site, vision=None, planner=LocalPlanner(), rule_parser=LocalRuleParser(site))
    msgs = agent.add_rule_from_text("alert me if anyone stands in the forklift lane for more than 5 seconds", 3.0)
    kinds = [m.kind for m in msgs]
    assert kinds == ["agent_step", "rule", "approval"]
    rule_msg = msgs[1].data
    assert rule_msg["enabled"] is False and rule_msg["confirmed"] is False and rule_msg["source"] == "language"
    ap = msgs[2].data
    assert ap["action"] == "confirm_rule" and ap["payload"]["zone_id"] == "lane_main"
    assert ap["payload"]["question"].startswith("Is this the lane you mean?")
    assert ap["created_t"] == 3.0
    out = agent.decide(ap["approval_id"], True, 4.0)
    rule = agent.engine.get_rule(rule_msg["rule_id"])
    assert rule.enabled and rule.confirmed
    assert any(m.kind == "rule" and m.data["enabled"] for m in out)


def test_rejected_rule_stays_disabled(site):
    agent = FloorsightAgent(site, vision=None, planner=LocalPlanner(), rule_parser=LocalRuleParser(site))
    msgs = agent.add_rule_from_text("tell me when Exit B is blocked for 30 seconds", 0.0)
    ap = msgs[-1].data
    agent.decide(ap["approval_id"], False, 1.0)
    assert not agent.engine.get_rule(msgs[1].data["rule_id"]).enabled


def test_unparseable_text_creates_no_rule(site):
    agent = FloorsightAgent(site, vision=None, planner=LocalPlanner(), rule_parser=LocalRuleParser(site))
    n = len(agent.engine.rules)
    msgs = agent.add_rule_from_text("make me a sandwich", 0.0)
    assert len(agent.engine.rules) == n
    assert not agent.approvals
    assert msgs[-1].kind == "rule" and msgs[-1].data["ok"] is False and msgs[-1].data["message"]


class _StubVision:
    def cameras_covering(self, xy):
        return ["cam1", "cam4"]

    def second_view(self, xy, exclude, kind):
        from floorsight.contracts import Evidence

        return [Evidence(kind="second_view", camera_id="cam4", t=0.0, supports=True, score=0.9, note="cam4 sees a person")]

    def sharper_crop(self, camera_id, xy, radius_m=1.0):
        from floorsight.contracts import Evidence

        return Evidence(kind="crop", camera_id=camera_id, t=0.0, supports=None, image_ref="crop-1", note="crop")

    def recalibrate(self, camera_id):
        return {"ok": True, "calib_error_m": 0.03, "markers_used": 4}

    def coverage(self):
        return None


def test_confirmed_language_rule_fires_and_is_verified(site):
    agent = FloorsightAgent(site, vision=_StubVision(), planner=LocalPlanner(), rule_parser=LocalRuleParser(site))
    msgs = agent.add_rule_from_text("alert me if anyone stands in the forklift lane for more than 5 seconds", 0.0)
    agent.decide(msgs[-1].data["approval_id"], True, 0.0)
    out = []
    for i in range(0, 80):
        t = i * 0.1
        tr = Track(track_id="T-x", cls=TrackClass.person, x=6.0, y=5.0, vx=0, vy=0, sigma_m=0.1, camera_ids=["cam1"], first_seen_t=0.0, last_seen_t=t, state=TrackState.seen)
        out += agent.on_world(WorldState(t=t, tracks=[tr], cameras=[], coverage_version=1, pipeline_latency_ms=1))
    dwell = [c for c in agent.cases.values() if c.event.type.value == "dwell"]
    assert len(dwell) == 1
    ev = dwell[0].event
    assert abs(ev.t_detect - 5.0) < 0.11 and ev.status.value == "confirmed"
    assert [s.tool for s in agent.trace if s.event_id == ev.event_id] == ["check_trajectory", "second_view", "sharper_crop", "confirm_event", "send_alert"]
