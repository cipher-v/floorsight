"""Bedrock planner / rule parser with a stubbed client. No network, ever."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from floorsight.agent.demo import run
from floorsight.agent.planner import LocalPlanner
from floorsight.llm import select_planner, select_rule_parser
from floorsight.llm.bedrock import MODEL_ID, PLANNER_TOOLS, BedrockPlanner, BedrockRuleParser


class StubClient:
    """Looks like AnthropicBedrockMantle for client.beta.messages.create(**params)."""

    def __init__(self, responder):
        self.requests = []
        stub = self

        class _Messages:
            def create(self, **params):
                stub.requests.append(params)
                return responder(params)

        self.beta = SimpleNamespace(messages=_Messages())


def tool_use(name, inp, stop="tool_use"):
    return SimpleNamespace(
        stop_reason=stop,
        stop_details=None,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="tool_use", id="toolu_1", name=name, input=inp)],
    )


def view_of(params):
    text = params["messages"][-1]["content"]
    return json.loads(text.split("\n", 1)[1])


def policy_model(params):
    """A 'model' that happens to follow the local policy exactly."""
    a = LocalPlanner().next_action(view_of(params))
    return tool_use(a.tool, {**a.input, "reason": "model: " + a.reason})


def reckless_model(params):
    """A 'model' that always wants to confirm (or alert) straight away."""
    v = view_of(params)
    if v["event"]["status"] in ("candidate", "verifying"):
        return tool_use("confirm_event", {"reason": "looks real to me"})
    return tool_use("send_alert", {"audience": "manager", "message": "!!!", "reason": "be safe"})


def _outcomes(agent):
    return sorted((c.event.type.value, round(c.event.t_detect), c.event.status.value) for c in agent.cases.values())


def test_request_shape_follows_the_opus_5_5_rules():
    stub = StubClient(lambda p: tool_use("wait", {"reason": "x"}))
    planner = BedrockPlanner(client=stub)
    view = {"t": 1.0, "trigger": "new", "event": {"event_id": "EV-1", "type": "near_miss", "status": "candidate", "evidence": [], "location": [1, 2], "track_ids": [], "camera_ids": [], "summary": "", "t_detect": 1.0, "t_start": 1.0}, "facts": {}, "approvals": [], "done": [], "done_this_trigger": [], "cameras": ["cam1"]}
    planner.next_action(view)
    req = stub.requests[0]
    assert req["model"] == MODEL_ID == "anthropic.claude-opus-5-5"
    assert req["tool_choice"] == {"type": "auto"}
    assert req["output_config"] == {"effort": "low"}
    assert "thinking" not in req and "budget_tokens" not in json.dumps(req)
    assert "temperature" not in req
    names = [t["name"] for t in req["tools"]]
    assert len(names) == len(set(names))
    for t in req["tools"]:
        assert t["strict"] is True
        sch = t["input_schema"]
        assert sch["additionalProperties"] is False and sorted(sch["required"]) == sorted(sch["properties"])
        assert "reason" in sch["properties"]
    # the view rides along as deterministic JSON
    assert view_of(req)["event"]["event_id"] == "EV-1"


def test_tool_set_matches_the_local_policy():
    from floorsight.agent.planner import ALL_TOOLS

    assert sorted(t["name"] for t in PLANNER_TOOLS) == sorted(ALL_TOOLS)


def test_full_run_with_policy_following_model(sim_dir, full_run):
    stub = StubClient(policy_model)
    agent, _ = run(sim_dir, auto_approve=True, planner=BedrockPlanner(client=stub))
    local_agent, _ = full_run
    assert _outcomes(agent) == _outcomes(local_agent)
    assert stub.requests  # the model was consulted ...
    # ... and its choices stood (a human's verdict is recorded as its own step and is not a model choice)
    assert all(s.mode == "bedrock" for s in agent.trace if not s.event_id.startswith("rule:") and s.tool != "human_decision")
    assert all(s.reason.startswith("model: ") for s in agent.trace if s.mode == "bedrock")


def test_reckless_model_is_overridden_by_the_guard(sim_dir, full_run):
    """The model can't confirm the glare (no supporting evidence) or alert the manager unapproved."""
    agent, _ = run(sim_dir, auto_approve=True, planner=BedrockPlanner(client=StubClient(reckless_model)))
    local_agent, _ = full_run
    assert _outcomes(agent) == _outcomes(local_agent)
    glare = next(c for c in agent.cases.values() if c.event.type.value == "blocked_exit" and c.event.t_detect < 60)
    assert glare.event.status.value == "rejected"
    assert not [a for a in agent.alerts if a.event_id == glare.event.event_id]
    overridden = [s for s in agent.trace if "guard overrode the model's confirm_event" in s.reason]
    assert overridden and all(s.mode == "local" for s in overridden)
    manager = [a for a in agent.alerts if a.audience == "manager"]
    esc = next(a for a in agent.approvals.values() if a.action == "escalate")
    assert len(manager) == 1 and manager[0].t >= esc.decided_t


def _blocked_view(**facts):
    return {
        "t": 95.0,
        "trigger": "new",
        "event": {"event_id": "EV-9", "type": "blocked_exit", "status": "verifying", "evidence": [], "location": [15.0, 7.9], "track_ids": ["T-x"], "camera_ids": ["cam2"], "summary": "", "t_detect": 95.0, "t_start": 75.0, "zone_id": "exit_b"},
        "facts": {"detect_cameras": ["cam2"], "zone_name": "Exit B", "alerted": [], **facts},
        "approvals": [],
        "done": [],
        "done_this_trigger": [],
        "cameras": ["cam1", "cam2", "cam3", "cam4"],
    }


def test_model_choice_is_used_when_allowed():
    planner = BedrockPlanner(client=StubClient(lambda p: tool_use("sharper_crop", {"camera_id": "cam2", "x": 15.0, "y": 7.9, "radius_m": 1.0, "reason": "look closer"})))
    a = planner.next_action(_blocked_view())
    assert (a.tool, a.mode, a.reason) == ("sharper_crop", "bedrock", "look closer")
    assert "reason" not in a.input


def test_unknown_camera_is_overridden():
    planner = BedrockPlanner(client=StubClient(lambda p: tool_use("sharper_crop", {"camera_id": "cam9", "x": 1, "y": 1, "radius_m": 1, "reason": "?"})))
    a = planner.next_action(_blocked_view())
    assert a.mode == "local" and "unknown camera" in a.reason and a.tool == "sharper_crop" and a.input["camera_id"] == "cam2"


def test_refusal_falls_back_to_local():
    resp = SimpleNamespace(stop_reason="refusal", stop_details=SimpleNamespace(category="cyber", explanation=None), content=[])
    planner = BedrockPlanner(client=StubClient(lambda p: resp))
    a = planner.next_action(_blocked_view())
    assert a.mode == "local" and "declined (category cyber)" in a.reason and a.tool == "sharper_crop"
    assert planner.fallbacks == 1


@pytest.mark.parametrize(
    "responder, note",
    [
        (lambda p: (_ for _ in ()).throw(ConnectionError("no network")), "Bedrock call failed (ConnectionError)"),
        (lambda p: SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="hmm")]), "no tool call"),
        (lambda p: tool_use("sharper_crop", {"camera_id": "cam2"}, stop="max_tokens"), "max_tokens"),
        (lambda p: tool_use("teleport", {"reason": "?"}), "unknown tool"),
    ],
)
def test_failures_fall_back_to_local(responder, note):
    a = BedrockPlanner(client=StubClient(responder)).next_action(_blocked_view())
    assert a.mode == "local" and note in a.reason and a.tool == "sharper_crop"


def test_model_cannot_sit_on_a_confirmed_event():
    view = _blocked_view()
    view["event"]["status"] = "confirmed"
    a = BedrockPlanner(client=StubClient(lambda p: tool_use("wait", {"reason": "later"}))).next_action(view)
    assert a.tool == "send_alert" and a.input["audience"] == "floor_lead" and a.mode == "local"


def test_bedrock_rule_parser(site):
    def model(params):
        assert params["tool_choice"] == {"type": "auto"} and all(t["strict"] for t in params["tools"])
        assert "exit_b" in params["system"]
        return tool_use("make_rule", {"type": "dwell", "zone_id": "lane_main", "classes": ["person"], "seconds": 5, "distance_m": None, "closing_mps": None, "reason": "dwell"})

    p = BedrockRuleParser(site, client=StubClient(model))
    r = p.parse("if anyone hangs about in the main forklift lane for 5 s, ping me")
    assert r.ok and r.rule.type.value == "dwell" and r.rule.zone_id == "lane_main" and r.rule.seconds == 5
    assert r.rule.confirmed is False and r.rule.enabled is False and r.rule.source == "language"


def test_bedrock_rule_parser_falls_back(site):
    bad_zone = BedrockRuleParser(site, client=StubClient(lambda p: tool_use("make_rule", {"type": "blocked", "zone_id": "dock_9", "classes": [], "seconds": 30, "distance_m": None, "closing_mps": None, "reason": ""})))
    r = bad_zone.parse("tell me when Exit B is blocked for 30 seconds")
    assert r.ok and r.rule.zone_id == "exit_b"  # local parser result
    nope = BedrockRuleParser(site, client=StubClient(lambda p: tool_use("cannot_parse", {"message": "That is not an alert.", "reason": ""})))
    r = nope.parse("make me a sandwich")
    assert not r.ok and r.message == "That is not an alert."


def test_env_selects_bedrock_without_creating_a_client(monkeypatch, site):
    monkeypatch.setenv("FLOORSIGHT_LLM", "bedrock")
    planner = select_planner()
    parser = select_rule_parser(site)
    assert isinstance(planner, BedrockPlanner) and planner._client is None
    assert isinstance(parser, BedrockRuleParser) and parser._client is None
    monkeypatch.setenv("FLOORSIGHT_LLM", "local")
    assert isinstance(select_planner(), LocalPlanner)


def test_sdk_names_exist_without_instantiating():
    import inspect

    from anthropic import AnthropicBedrockMantle, BetaFallbackState, BetaRefusalFallbackMiddleware

    params = inspect.signature(AnthropicBedrockMantle.__init__).parameters
    assert "aws_region" in params and "middleware" in params
    assert BetaFallbackState and BetaRefusalFallbackMiddleware


def test_local_mode_never_imports_anthropic(sim_dir):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    code = (
        "import sys; from floorsight.agent.demo import run; "
        f"agent, _ = run({sim_dir!r}); "
        "assert 'anthropic' not in sys.modules, 'anthropic imported in local mode'; print(len(agent.trace))"
    )
    env = {k: v for k, v in os.environ.items() if k != "FLOORSIGHT_LLM"}
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
