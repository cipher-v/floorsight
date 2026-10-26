"""Claude on Amazon Bedrock as the agent's planner and rule parser.

PREPARED, NOT CALLED: used only when FLOORSIGHT_LLM=bedrock (or when a client
is injected, as the tests do with a stub). Nothing here runs on import and the
anthropic SDK is imported lazily.

    planner = BedrockPlanner(region="us-east-1")
    action = planner.next_action(view)   # same view the LocalPlanner gets

Design:
- Same tools the local policy uses, exposed as strict Claude tool definitions.
  The model picks the next tool from the evidence so far (one stateless call
  per step; the case view carries the history).
- The local policy is the guard: `floorsight.agent.planner.guard` vets every
  model choice (no confirming without independent supporting evidence, no
  manager alert without an approved escalation, ...). Rejected choices, model
  refusals, missing tool calls and API errors all fall back to LocalPlanner,
  and the trace step records why (mode "local").
- API notes for anthropic.claude-opus-5-5 on Bedrock: tool_choice must be
  "auto" (forced tool choice is a 400 on this model), `thinking` is omitted
  (disabled/budget_tokens are a 400; it is always adaptive), effort is set via
  output_config for this latency-sensitive loop, and stop_reason is checked
  before reading content. Server-side `fallbacks` is unavailable on Bedrock,
  so refusals are retried client-side by BetaRefusalFallbackMiddleware, which
  only handles `client.beta.messages` - hence the beta surface below.
"""

from __future__ import annotations

import json
import os
from typing import Any

from floorsight.agent.language import LocalRuleParser, ParseResult, describe, match_zones
from floorsight.agent.planner import Action, LocalPlanner, guard
from floorsight.contracts import Rule, RuleType, SitePlan, TrackClass

MODEL_ID = "anthropic.claude-opus-5-5"
# Refusal fallback target (client-side middleware). Permitted targets are not
# listed by Bedrock; the docs expect Claude Opus 5 / Opus 4.8 for Opus 5.5.
FALLBACK_MODEL_ID = "anthropic.claude-opus-5"
DEFAULT_REGION = "us-east-1"
SAFETY_TOOLS = ("send_alert", "request_approval", "confirm_event", "reject_event", "resolve_event", "add_rule", "schedule_follow_up")


def make_bedrock_client(region: str | None = None):
    """AnthropicBedrockMantle client with client-side refusal fallback. Creating it makes no request."""
    from anthropic import AnthropicBedrockMantle, BetaRefusalFallbackMiddleware

    return AnthropicBedrockMantle(
        aws_region=region or os.environ.get("AWS_REGION") or DEFAULT_REGION,
        middleware=[BetaRefusalFallbackMiddleware([{"model": FALLBACK_MODEL_ID}])],
    )


# --------------------------------------------------------------------------- #
# Tool definitions (strict: every object closed, every property required)
# --------------------------------------------------------------------------- #


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


_REASON = {"type": "string", "description": "One sentence for the audit trail: why this is the right next step, citing the evidence so far."}
_X = {"type": "number", "description": "Floor x in metres (use the event location unless you have a reason not to)"}
_Y = {"type": "number", "description": "Floor y in metres"}


def _tool(name: str, description: str, props: dict) -> dict:
    return {"name": name, "description": description, "strict": True, "input_schema": _obj({**props, "reason": _REASON})}


PLANNER_TOOLS: list[dict] = [
    _tool("sharper_crop", "Get a full-resolution, face-blurred close-up around a floor point from one camera and store it as evidence. Call it to look closer at what the detecting camera reported before trusting it.", {"camera_id": {"type": "string"}, "x": _X, "y": _Y, "radius_m": {"type": "number"}}),
    _tool("second_view", "Ask every other camera covering a floor point whether it also sees the thing. kind 'static' = object lying there, 'moving' = person present. Returns one evidence item per camera (supports true/false). Call it before confirming any blocked exit.", {"x": _X, "y": _Y, "exclude": {"type": "array", "items": {"type": "string"}, "description": "Cameras to leave out, normally the detecting camera"}, "kind": {"type": "string", "enum": ["static", "moving"]}}),
    _tool("cameras_covering", "List cameras that currently have an unobstructed view of a floor point. Call it when a person may be in a blind spot.", {"x": _X, "y": _Y}),
    _tool("recalibrate", "Re-fit a moved camera's floor alignment from the floor markers it can still see. Only for camera_moved events.", {"camera_id": {"type": "string"}}),
    _tool("check_trajectory", "Check the tracks behind an event are well established (seen >= 1 s, low position uncertainty) and report the measured gap/closing speed. Call it before confirming a near-miss or dwell.", {"track_ids": {"type": "array", "items": {"type": "string"}}}),
    _tool("incident_history", "Count confirmed near-misses within a radius of a point this shift. Call it after confirming a near-miss to find hotspots.", {"x": _X, "y": _Y, "radius_m": {"type": "number"}}),
    _tool("confirm_event", "Mark the event confirmed. Only allowed when independent evidence supports it (another camera for blocked exits, a stable trajectory for near-misses).", {}),
    _tool("reject_event", "Mark the event a false alarm. Only when the evidence contradicts it and nothing supports it.", {}),
    _tool("send_alert", "Alert a person on the floor. floor_lead for confirmed incidents or approved dispatches; manager only after a human approved escalation.", {"audience": {"type": "string", "enum": ["floor_lead", "manager"]}, "message": {"type": "string", "description": "Plain-language message for the person"}}),
    _tool("request_approval", "Ask a human to approve an action: escalate (still blocked at follow-up), dispatch_check (no camera can verify), recalibrate (self-repair failed), create_rule (3rd confirmed near-miss at one spot), confirm_incident (the human's verdict on a floor warning: real hazard or false alarm).", {"action": {"type": "string", "enum": ["escalate", "dispatch_check", "recalibrate", "create_rule", "confirm_incident"]}, "summary": {"type": "string", "description": "The question shown to the human"}}),
    _tool("short_replay", "Floor warnings only: look back over the last seconds of floor measurements for this person/forklift pair - how long the condition held, gap and predicted pass, tracking gaps, heading or speed jumps. supports=false means the onset looks like a tracking artefact.", {"track_ids": {"type": "array", "items": {"type": "string"}}, "seconds": {"type": "number"}}),
    _tool("redetect", "Floor warnings only: run the detectors again on a sharper crop and re-measure the floor gap; supports=true when it agrees with the tracker.", {"track_ids": {"type": "array", "items": {"type": "string"}}}),
    _tool("recommend", "Floor warnings only: record your verdict from the second look (confirm, reject or uncertain) before asking the human. It changes nothing by itself.", {"verdict": {"type": "string", "enum": ["confirm", "reject", "uncertain"]}}),
    _tool("schedule_follow_up", "Come back to a confirmed event after this many seconds of sim time to check it was handled.", {"seconds": {"type": "number"}}),
    _tool("resolve_event", "Close the event because the facts show it is over (object gone, pair separated, camera re-aligned, person visible again).", {}),
    _tool("add_rule", "Add the standing rule from an approved create_rule approval.", {"approval_id": {"type": "string"}}),
    _tool("wait", "Nothing more to do until something changes (a timer, a human decision, the world).", {}),
]

PLANNER_SYSTEM = """You are the safety agent of Floorsight, a live floor map of a warehouse built from fixed ceiling cameras.
You verify before you act. Each request gives you one incident (an event, the evidence gathered so far, facts the runtime knows,
approvals asked, tools already used, and the trigger that woke you). Choose exactly one next tool.

Policy:
- Never trust a single camera. Blocked exit: sharper_crop from the detecting camera, then second_view(kind=static) excluding it.
  Confirm only if another camera supports it; reject as a false alarm if other cameras contradict and none support;
  if no other camera covers the spot, request_approval(dispatch_check).
- Confirmed blocked exit: alert the floor_lead, schedule a 30 s follow-up; at the follow-up re-check with second_view and,
  if still blocked, request_approval(escalate). Alert the manager only after escalation is approved. Resolve when the object is gone.
- Near-miss: check_trajectory, sharper_crop, then confirm or reject; alert the floor_lead; incident_history within 3 m;
  on the third confirmed near-miss at one spot request_approval(create_rule) for a slow zone; add_rule once approved.
- Camera moved: recalibrate; resolve if it worked, otherwise request_approval(recalibrate).
- Person unseen in a restricted zone: cameras_covering; if none, request_approval(dispatch_check); alert the floor_lead once approved.
- Use wait when nothing should happen until a timer, a human or the world changes. Keep reasons to one factual sentence.
A deterministic guard checks every choice; choices without supporting evidence are overridden."""


def _model_view(view: dict) -> str:
    """The case view as compact, deterministic JSON (stable key order keeps prompts cacheable)."""
    return json.dumps(view, sort_keys=True, separators=(",", ":"), default=str)


def _block_type(block: Any) -> str | None:
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def _tool_use(resp: Any) -> Any | None:
    return next((b for b in (getattr(resp, "content", None) or []) if _block_type(b) == "tool_use"), None)


def _field(block: Any, name: str) -> Any:
    return block.get(name) if isinstance(block, dict) else getattr(block, name, None)


def _refusal_note(resp: Any) -> str:
    details = getattr(resp, "stop_details", None)
    cat = getattr(details, "category", None) if details is not None else None
    return f"model declined (category {cat})" if cat else "model declined"


class _BedrockBase:
    def __init__(self, client=None, model: str = MODEL_ID, region: str | None = None, effort: str = "low", max_tokens: int = 4096):
        self._client = client
        self._own_client = client is None
        self.model = model
        self.region = region
        self.effort = effort
        self.max_tokens = max_tokens
        self.calls = 0
        self.fallbacks = 0
        self._states: dict[str, Any] = {}

    @property
    def client(self):
        if self._client is None:
            self._client = make_bedrock_client(self.region)
        return self._client

    def _create(self, params: dict, scope: str):
        """One Messages call on the beta surface (so the refusal-fallback middleware applies)."""
        self.calls += 1
        if self._own_client:
            from anthropic import BetaFallbackState

            state = self._states.setdefault(scope, BetaFallbackState())  # pins one incident to the model that accepted
            with state:
                return self.client.beta.messages.create(**params)
        return self.client.beta.messages.create(**params)


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #


class BedrockPlanner(_BedrockBase):
    """Claude picks the next tool; the local policy guards and backs it up."""

    mode = "bedrock"

    def __init__(self, client=None, model: str = MODEL_ID, region: str | None = None, effort: str = "low", max_tokens: int = 4096, local: LocalPlanner | None = None):
        super().__init__(client, model, region, effort, max_tokens)
        self.local = local or LocalPlanner()

    def build_request(self, view: dict) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": PLANNER_SYSTEM,
            "tools": PLANNER_TOOLS,
            "tool_choice": {"type": "auto"},
            "output_config": {"effort": self.effort},
            "messages": [{"role": "user", "content": "Incident case view (JSON). Choose the next tool.\n" + _model_view(view)}],
        }

    def next_action(self, view: dict) -> Action:
        try:
            resp = self._create(self.build_request(view), view["event"]["event_id"])
        except Exception as e:  # auth, network, HTTP errors: the local policy keeps the floor safe
            return self._local(view, f"Bedrock call failed ({type(e).__name__})")
        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            return self._local(view, _refusal_note(resp))
        if stop == "max_tokens":
            return self._local(view, "model output cut off at max_tokens")
        block = _tool_use(resp)
        if block is None:
            return self._local(view, "model made no tool call")
        inp = dict(_field(block, "input") or {})
        reason = str(inp.pop("reason", "") or "")
        action = Action(tool=str(_field(block, "name")), input=inp, reason=reason, mode="bedrock")
        ok, why = guard(view, action, view.get("cameras"))
        if not ok:
            return self._local(view, f"guard overrode the model's {action.tool}: {why}")
        if action.tool == "wait":
            local = self.local.next_action(view)
            if local.tool in SAFETY_TOOLS:
                return self._local(view, f"model chose to wait but the policy requires {local.tool}", local)
        return action

    def _local(self, view: dict, note: str, action: Action | None = None) -> Action:
        self.fallbacks += 1
        a = action or self.local.next_action(view)
        return Action(a.tool, a.input, f"[{note}; local policy used] {a.reason}".strip(), mode="local")


# --------------------------------------------------------------------------- #
# Rule parser
# --------------------------------------------------------------------------- #

RULE_SYSTEM = """You turn a warehouse supervisor's plain-language alert request into one Floorsight rule.
Rule types: blocked (static object inside a zone longer than `seconds`), dwell (a class inside a zone longer than `seconds`),
proximity (classes[0] within `distance_m` of classes[1], optionally only inside a zone, optionally only when closing faster than
`closing_mps`), unseen (someone predicted inside a zone while no camera can see them, longer than `seconds`).
Use only the zone ids listed. Use null for fields that do not apply. If the request is not an alert rule, call cannot_parse."""


class BedrockRuleParser(_BedrockBase):
    """Claude parses the rule; anything invalid falls back to LocalRuleParser."""

    mode = "bedrock"

    def __init__(self, site: SitePlan, client=None, model: str = MODEL_ID, region: str | None = None, effort: str = "low", max_tokens: int = 2048):
        super().__init__(client, model, region, effort, max_tokens)
        self.site = site
        self.local = LocalRuleParser(site)
        zone_ids = [z.zone_id for z in site.zones]
        num_or_null = {"anyOf": [{"type": "number"}, {"type": "null"}]}
        self.tools = [
            _tool(
                "make_rule",
                "Create the rule the supervisor asked for.",
                {
                    "type": {"type": "string", "enum": [t.value for t in RuleType]},
                    "zone_id": {"anyOf": [{"type": "string", "enum": zone_ids}, {"type": "null"}]},
                    "classes": {"type": "array", "items": {"type": "string", "enum": [c.value for c in TrackClass]}},
                    "seconds": num_or_null,
                    "distance_m": num_or_null,
                    "closing_mps": num_or_null,
                },
            ),
            _tool("cannot_parse", "The text is not an alert rule Floorsight can watch for. Explain what is missing.", {"message": {"type": "string"}}),
        ]
        self.zone_list = "\n".join(f"- {z.zone_id}: {z.name} ({z.kind.value})" for z in site.zones)

    def build_request(self, text: str) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": RULE_SYSTEM + "\n\nZones:\n" + self.zone_list,
            "tools": self.tools,
            "tool_choice": {"type": "auto"},
            "output_config": {"effort": self.effort},
            "messages": [{"role": "user", "content": text}],
        }

    def parse(self, text: str) -> ParseResult:
        try:
            resp = self._create(self.build_request(text), "rules")
        except Exception:
            return self.local.parse(text)
        if getattr(resp, "stop_reason", None) in ("refusal", "max_tokens"):
            return self.local.parse(text)
        block = _tool_use(resp)
        if block is None:
            return self.local.parse(text)
        inp = dict(_field(block, "input") or {})
        if _field(block, "name") == "cannot_parse":
            local = self.local.parse(text)
            return local if local.ok else ParseResult(None, str(inp.get("message") or local.message))
        try:
            rtype = RuleType(inp["type"])
            zone_id = inp.get("zone_id")
            zone = next((z for z in self.site.zones if z.zone_id == zone_id), None)
            if zone_id is not None and zone is None:
                raise ValueError("unknown zone")
            if rtype != RuleType.proximity and zone is None:
                raise ValueError("zone required")
            classes = [TrackClass(c) for c in inp.get("classes") or []]
            if rtype == RuleType.proximity and len(classes) < 2:
                classes = [TrackClass.person, TrackClass.forklift]
            rule = Rule(
                rule_id="",
                text=text.strip(),
                type=rtype,
                zone_id=zone_id,
                classes=classes,
                seconds=inp.get("seconds"),
                distance_m=inp.get("distance_m"),
                closing_mps=inp.get("closing_mps"),
                source="language",
                confirmed=False,
                enabled=False,
            )
            if rtype == RuleType.proximity and rule.distance_m is None:
                raise ValueError("distance required")
            if rtype != RuleType.proximity and rule.seconds is None:
                rule.seconds = {RuleType.blocked: 20.0, RuleType.unseen: 10.0}.get(rtype, 0.0)
        except (KeyError, ValueError):
            return self.local.parse(text)
        ranked = match_zones(self.site, text)
        alternatives = [z.zone_id for s, z in ranked if z.zone_id != zone_id and ranked and s == ranked[0][0]]
        return ParseResult(rule, describe(rule, self.site), zone=zone, alternatives=alternatives)
