"""Verify-then-act policy.

A planner looks at a *case view* (the event, the evidence gathered so far, a
few facts the runtime knows, and why it was woken up) and picks the next tool.
The runtime executes it, records an AgentStep and asks again until the planner
says "wait". `LocalPlanner` is the deterministic policy; the Bedrock planner
(floorsight.llm.bedrock) lets Claude choose instead, but every choice it makes
goes through `guard`, which mirrors the local policy's hard rules - most
importantly that an event can't be confirmed without supporting evidence.

View layout (plain JSON so it can be sent to a model unchanged):
    {"t", "trigger", "event": Event dict, "facts": {...}, "approvals": [...],
     "done": [tool names used on this event], "done_this_trigger": [...],
     "cameras": [site camera ids]}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
from floorsight.formatting import floor_error_text

VISION_TOOLS = ("sharper_crop", "second_view", "cameras_covering", "recalibrate", "short_replay", "redetect")
ANALYSIS_TOOLS = ("check_trajectory", "incident_history")
DECISION_TOOLS = (
    "recommend",
    "confirm_event",
    "reject_event",
    "send_alert",
    "request_approval",
    "schedule_follow_up",
    "resolve_event",
    "add_rule",
    "wait",
)
ALL_TOOLS = VISION_TOOLS + ANALYSIS_TOOLS + DECISION_TOOLS

CLOSED = ("resolved", "rejected")
NEAR_MISS_MIN_VIEWS = 2  # independent camera views needed when the tracks themselves are too new
OPEN_VERIFY = ("candidate", "verifying")
ACTIVE = ("confirmed", "escalated")

# Which approvals each event type may ask for.
APPROVALS_BY_TYPE = {
    "blocked_exit": ("escalate", "dispatch_check"),
    "near_miss": ("create_rule",),
    "camera_moved": ("recalibrate",),
    "unseen_in_zone": ("dispatch_check",),
    "dwell": ("dispatch_check",),
    "floor_warning": ("confirm_incident",),
}


@dataclass
class Action:
    tool: str
    input: dict = field(default_factory=dict)
    reason: str = ""
    mode: str = "local"


WAIT = "wait"


class Planner(Protocol):
    mode: str

    def next_action(self, view: dict) -> Action: ...


# --------------------------------------------------------------------------- #
# View helpers
# --------------------------------------------------------------------------- #


def evidence(view: dict, kind: str) -> list[dict]:
    return [e for e in view["event"]["evidence"] if e["kind"] == kind]


def approval(view: dict, action: str) -> dict | None:
    """Most recent approval of this kind for the event."""
    found = [a for a in view["approvals"] if a["action"] == action]
    return found[-1] if found else None


def approved(view: dict, action: str) -> bool:
    a = approval(view, action)
    return a is not None and a["status"] == "approved"


def alerted(view: dict, audience: str) -> bool:
    return audience in view["facts"].get("alerted", [])


def has_support(view: dict) -> bool:
    """True if the gathered evidence supports the event (needed before confirming)."""
    etype = view["event"]["type"]
    detect = set(view["facts"].get("detect_cameras", []))
    sv = evidence(view, "second_view")
    traj = evidence(view, "trajectory")
    if etype == "blocked_exit":
        return any(e["supports"] is True and e["camera_id"] not in detect for e in sv)
    if etype == "near_miss":
        # Stable tracks, or (when the tracker just re-numbered someone) two cameras seeing a person on the spot.
        return view["facts"].get("geometry_supported", False) and (any(e["supports"] is True for e in traj) or sum(1 for e in sv if e["supports"] is True) >= NEAR_MISS_MIN_VIEWS)
    if etype == "dwell":
        return any(e["supports"] is True for e in traj) and (not sv or any(e["supports"] is True for e in sv))
    if etype == "unseen_in_zone":
        return any(e["supports"] is True for e in sv)
    if etype == "camera_moved":
        return any(e["supports"] is True for e in evidence(view, "calibration"))
    if etype == "floor_warning":
        # One camera only: the agent's second look informs the human, but only a human verdict confirms.
        return approved(view, "confirm_incident")
    return False


def has_contradiction(view: dict) -> bool:
    etype = view["event"]["type"]
    if etype == "floor_warning":
        a = approval(view, "confirm_incident")
        return a is not None and a["status"] == "rejected"
    kinds = {"blocked_exit": ("second_view",), "near_miss": ("trajectory",), "dwell": ("trajectory", "second_view"), "unseen_in_zone": ("second_view",)}
    return any(e["supports"] is False for k in kinds.get(etype, ()) for e in evidence(view, k))


def _detector_text(d: dict, verdict: str) -> str:
    """'second detector pass <verdict>: <note>', without repeating the label when the note already starts with it."""
    note = d["note"]
    if note.startswith("second detector pass"):
        return note
    return f"second detector pass {verdict}: {note}" if verdict else f"second detector pass: {note}"


# Replay findings that mark the onset as a likely tracking artefact (wording from ReplayVision.short_replay).
_ARTEFACT_MARKS = ("jumped", "not measured", "cut by the frame")


def floor_recommendation(view: dict) -> tuple[str, str]:
    """The agent's own verdict for a floor warning from its second look: (confirm | reject | uncertain, why)."""
    rep = evidence(view, "replay")
    red = evidence(view, "redetect")
    r = rep[-1] if rep else None
    d = red[-1] if red else None
    # A tracking glitch does not establish safety. A second measurement may
    # disagree because the person is CLOSER than the tracker said. Reject only
    # when both checks contradict the warning and the independent gap is larger.
    tracked = view.get("facts", {}).get("min_clearance_m")
    measured = d.get("score") if d else None
    larger_gap = (isinstance(tracked, (int, float)) and isinstance(measured, (int, float))
                  and measured > max(.8, tracked + max(.4, .3 * tracked)))
    if r is not None and d is not None:
        if r["supports"] is False and d["supports"] is False and larger_gap:
            return "reject", "short replay: " + r["note"] + "; " + _detector_text(d, "disagrees with a larger gap")
        if r["supports"] is True and d["supports"] is True:
            return "confirm", "short replay: " + r["note"] + "; " + _detector_text(d, "agrees")
    parts = ["short replay: " + r["note"] if r else "no replay", _detector_text(d, "") if d else "no second detector pass"]
    return "uncertain", "Evidence is incomplete or conflicting; keep for human review. " + "; ".join(parts)


def floor_headline(view: dict, verdict: str) -> str:
    """One plain line for the decision card: the verdict and the single reason that drives it."""
    rep = evidence(view, "replay")
    red = evidence(view, "redetect")
    r = rep[-1] if rep else None
    d = red[-1] if red else None
    if verdict == "reject":
        if r is not None and r["supports"] is False:
            marks = [p.strip() for p in r["note"].split(";") if any(m in p for m in _ARTEFACT_MARKS)]
            return "Agent recommends REJECT: likely a tracking glitch" + (f" ({marks[0]})" if marks else "")
        return "Agent recommends REJECT: a second detector re-measured a different gap"
    if verdict == "confirm":
        held = f"held for {int(r['score'])} frames " if r is not None and r.get("score") else ""
        tail = ", and a second detector agrees" if d is not None and d["supports"] is True else ""
        return f"Agent recommends CONFIRM: the warning {held}with no tracking glitch{tail}"
    return "Agent is UNCERTAIN: the second look could not settle it"


def _xy(view: dict) -> dict:
    x, y = view["event"]["location"]
    return {"x": x, "y": y}


# --------------------------------------------------------------------------- #
# Local policy
# --------------------------------------------------------------------------- #


class LocalPlanner:
    """Deterministic verify-then-act policy, one branch per event type."""

    mode = "local"

    def next_action(self, view: dict) -> Action:
        fn = getattr(self, "_" + view["event"]["type"], None)
        return fn(view) if fn else Action(WAIT, reason="no policy for this event type")

    # ---- blocked exit ---------------------------------------------------- #

    def _blocked_exit(self, v: dict) -> Action:
        ev, f = v["event"], v["facts"]
        st = ev["status"]
        zone = f.get("zone_name") or ev.get("zone_id") or "the exit"
        if st in CLOSED:
            return Action(WAIT)
        if v["trigger"] == "gone":
            return Action("resolve_event", {}, f"No object seen at {zone} for {f.get('absent_s', 0):.0f} s - the exit is clear")
        if st in OPEN_VERIFY:
            detect = f.get("detect_cameras") or []
            if not evidence(v, "crop") and detect:
                return Action("sharper_crop", {"camera_id": detect[0], **_xy(v), "radius_m": 1.0}, f"Look closer at what {detect[0]} reported before trusting it")
            if "second_view" not in v["done"]:
                return Action("second_view", {**_xy(v), "exclude": detect, "kind": "static"}, "Ask every other camera that covers this spot whether it sees the same object")
            if has_support(v):
                cams = sorted({e["camera_id"] for e in evidence(v, "second_view") if e["supports"]})
                return Action("confirm_event", {}, f"{_names(cams)} independently {_verb(cams, 'sees', 'see')} an object in {zone}")
            if has_contradiction(v):
                cams = sorted({e["camera_id"] for e in evidence(v, "second_view") if e["supports"] is False})
                return Action("reject_event", {}, f"{_names(cams)} {_verb(cams, 'sees', 'see')} nothing at this spot; only {_names(detect)} reported it - treating it as a false alarm")
            if approval(v, "dispatch_check") is None:
                return Action(
                    "request_approval",
                    {"action": "dispatch_check", "summary": f"Only {', '.join(detect) or 'one camera'} can see {zone}. Ask the nearest worker to check whether it is blocked?"},
                    "No other camera covers this spot, so a person has to check",
                )
            if approved(v, "dispatch_check") and not alerted(v, "floor_lead"):
                return Action("send_alert", {"audience": "floor_lead", "message": f"Please check {zone}: a camera reports it may be blocked."}, "Dispatch approved")
            return Action(WAIT)
        # confirmed / escalated
        if not alerted(v, "floor_lead"):
            return Action("send_alert", {"audience": "floor_lead", "message": f"{zone} is blocked by an object at ({ev['location'][0]:.1f}, {ev['location'][1]:.1f}). Please clear it."}, "Blocked exit confirmed by a second camera")
        if approved(v, "escalate") and not alerted(v, "manager"):
            return Action("send_alert", {"audience": "manager", "message": f"{zone} has been blocked since t={ev['t_start']:.0f} s and is still blocked after a reminder to the floor lead."}, "Escalation approved by a human")
        if f.get("follow_up_t") is None:
            return Action("schedule_follow_up", {"seconds": f.get("follow_up_s", 30.0)}, "Check again later whether the floor lead cleared it")
        if v["trigger"] == "follow_up" and approval(v, "escalate") is None:
            if "second_view" not in v["done_this_trigger"]:
                return Action("second_view", {**_xy(v), "exclude": [], "kind": "static"}, "Follow-up: is the object still there?")
            recheck = [e for e in evidence(v, "second_view") if e["t"] >= f["follow_up_t"]]
            if f.get("still_present") or any(e["supports"] for e in recheck):
                return Action(
                    "request_approval",
                    {"action": "escalate", "summary": f"{zone} is still blocked {v['t'] - ev['t_detect']:.0f} s after the floor lead was told. Escalate to the manager?"},
                    "Still blocked at follow-up",
                )
        return Action(WAIT)

    # ---- near-miss ------------------------------------------------------- #

    def _near_miss(self, v: dict) -> Action:
        ev, f = v["event"], v["facts"]
        st = ev["status"]
        if approved(v, "create_rule") and not f.get("rule_added"):
            a = approval(v, "create_rule")
            return Action("add_rule", {"approval_id": a["approval_id"]}, "A human approved the proposed standing fix")
        if st in CLOSED:
            return Action(WAIT)
        if st in OPEN_VERIFY:
            if not evidence(v, "trajectory"):
                return Action("check_trajectory", {"track_ids": ev["track_ids"]}, "Check both tracks are well established before trusting the distance")
            if not has_support(v) and not evidence(v, "second_view"):
                return Action(
                    "second_view",
                    {**_xy(v), "exclude": [], "kind": "moving"},
                    "A track is too new to trust on its own - look at the spot from every camera that covers it",
                )
            if not evidence(v, "crop"):
                cam = (f.get("detect_cameras") or ev["camera_ids"] or [None])[0]
                if cam:
                    return Action("sharper_crop", {"camera_id": cam, **_xy(v), "radius_m": 1.5}, "Keep a blurred close-up for the record")
            if has_support(v):
                return Action("confirm_event", {}, "Both tracks are stable and the measured gap and closing speed are real")
            return Action("reject_event", {}, "Reliable pair-clearance geometry and observation support are not both available; this does not establish safe clearance")
        if v["trigger"] == "over":
            return Action("resolve_event", {}, f"The pair separated {f.get('since_last_hit_s', 0):.0f} s ago; incident logged")
        if not alerted(v, "floor_lead"):
            return Action("send_alert", {"audience": "floor_lead", "message": ev["summary"]}, "Confirmed near-miss")
        if not evidence(v, "history"):
            return Action("incident_history", {**_xy(v), "radius_m": 3.0}, "Has this happened here before?")
        if f.get("history_count", 0) >= 3 and not f.get("hotspot_proposed") and f.get("counts_for_history", True):
            return Action(
                "request_approval",
                {"action": "create_rule", "summary": f.get("proposal_text", "Add a slow zone here?")},
                f"{f['history_count']} confirmed near-misses within 3 m - propose a standing fix",
            )
        return Action(WAIT)

    # ---- floor warning (real-camera floor stage) -------------------------- #

    def _floor_warning(self, v: dict) -> Action:
        """Second look (replay, close-up, second detector pass) -> recommendation -> human verdict -> act."""
        ev, f = v["event"], v["facts"]
        st = ev["status"]
        if st in CLOSED:
            return Action(WAIT)
        if st in OPEN_VERIFY:
            if not evidence(v, "replay") and "short_replay" not in v["done"]:
                return Action("short_replay", {"track_ids": ev["track_ids"], "seconds": 1.5},
                              "Was it sustained, or a one-frame dip or a tracking jump? Look back over the last 1.5 s of floor measurements")
            if not evidence(v, "crop") and "sharper_crop" not in v["done"]:
                cam = (f.get("detect_cameras") or ev["camera_ids"] or v["cameras"] or [None])[0]
                if cam:
                    return Action("sharper_crop", {"camera_id": cam, **_xy(v), "radius_m": 1.5}, "Full-resolution close-up (already privacy-filtered) for the human reviewer")
            if not evidence(v, "redetect") and "redetect" not in v["done"]:
                return Action("redetect", {"track_ids": ev["track_ids"]},
                              "Run the detectors again on a sharper crop and re-measure the floor gap independently of the tracker")
            if f.get("redetect_pending"):
                return Action(WAIT, reason="waiting for the second detector pass")
            if f.get("recommendation") is None:
                verdict, why = floor_recommendation(v)
                return Action("recommend", {"verdict": verdict, "headline": floor_headline(v, verdict)}, why)
            a = approval(v, "confirm_incident")
            if a is None:
                rec = f["recommendation"]
                head = {"confirm": "Agent recommends CONFIRM", "reject": "Agent recommends REJECT (false alarm)", "uncertain": "Agent is UNCERTAIN"}[rec["verdict"]]
                return Action(
                    "request_approval",
                    {"action": "confirm_incident", "summary": f"{head}: {ev['summary']}. {rec['reason']}. Approve = real hazard, alert the floor lead; reject = false alarm."},
                    "Only one camera sees this spot: a human makes the final call",
                )
            if a["status"] == "approved":
                return Action("confirm_event", {}, "A human reviewer confirmed the hazard from the agent's evidence")
            if a["status"] == "rejected":
                return Action("reject_event", {}, "A human reviewer judged it a false alarm")
            return Action(WAIT)
        if not alerted(v, "floor_lead"):
            return Action("send_alert", {"audience": "floor_lead", "message": ev["summary"]}, "Hazard confirmed by a human reviewer")
        if v["trigger"] == "over":
            return Action("resolve_event", {}, f"The person and the forklift separated {f.get('since_last_hit_s', 0):.0f} s ago; incident logged")
        return Action(WAIT)

    # ---- camera moved ---------------------------------------------------- #

    def _camera_moved(self, v: dict) -> Action:
        ev, f = v["event"], v["facts"]
        st = ev["status"]
        cam = ev["camera_ids"][0]
        if st in CLOSED:
            return Action(WAIT)
        if v["trigger"] == "camera_ok" and "recalibrate" in v["done"]:
            return Action("resolve_event", {}, f"{cam} reports healthy alignment again")
        rec = f.get("recalibrate")
        if rec is None:
            return Action("recalibrate", {"camera_id": cam}, f"Try to re-fit {cam}'s floor alignment from the markers it can still see")
        if rec.get("ok"):
            return Action("resolve_event", {}, f"Recalibrated from {rec.get('markers_used')} markers: floor error {floor_error_text(f.get('calib_error_before_m'))} -> {floor_error_text(rec.get('calib_error_m'))}")
        if approval(v, "recalibrate") is None:
            return Action(
                "request_approval",
                {"action": "recalibrate", "summary": f"{cam} could not re-align itself ({rec.get('markers_used', 0)} markers). It needs a person to fix the mount."},
                "Self-repair failed",
            )
        if approved(v, "recalibrate") and not alerted(v, "floor_lead"):
            return Action("send_alert", {"audience": "floor_lead", "message": f"Please re-seat the mount of {cam}; it has moved and cannot re-align itself."}, "Repair visit approved")
        return Action(WAIT)

    # ---- unseen in zone -------------------------------------------------- #

    def _unseen_in_zone(self, v: dict) -> Action:
        ev, f = v["event"], v["facts"]
        st = ev["status"]
        zone = f.get("zone_name") or ev.get("zone_id")
        if st in CLOSED:
            return Action(WAIT)
        if v["trigger"] == "cleared":
            return Action("resolve_event", {}, f"The unseen-track condition ended in {zone}; the track may have expired. This does not verify that the zone is empty.")
        if st in OPEN_VERIFY:
            if "cameras_covering" not in v["done"]:
                return Action("cameras_covering", _xy(v), f"Can any camera see inside {zone}?")
            if f.get("covering"):
                if "second_view" not in v["done"]:
                    return Action("second_view", {**_xy(v), "exclude": [], "kind": "moving"}, "Some camera covers the spot - look for the person")
                if has_support(v):
                    return Action("confirm_event", {}, f"A camera sees someone in {zone}")
                if has_contradiction(v):
                    return Action("reject_event", {}, "Covering cameras see nobody there - the predicted position was stale")
            if approval(v, "dispatch_check") is None:
                return Action(
                    "request_approval",
                    {"action": "dispatch_check", "summary": f"Someone is probably inside {zone} and no camera can see them. Ask the nearest worker to check the {_short(zone)}?"},
                    f"{zone} is a blind spot; only a person can verify",
                )
            if approved(v, "dispatch_check") and not alerted(v, "floor_lead"):
                return Action("send_alert", {"audience": "floor_lead", "message": f"Please check the {_short(zone)}: someone went in and no camera can see them."}, "Dispatch approved")
            return Action(WAIT)
        if not alerted(v, "floor_lead"):
            return Action("send_alert", {"audience": "floor_lead", "message": ev["summary"]}, "Person confirmed in restricted zone")
        return Action(WAIT)

    # ---- dwell ------------------------------------------------------------ #

    def _dwell(self, v: dict) -> Action:
        ev, f = v["event"], v["facts"]
        st = ev["status"]
        if st in CLOSED:
            return Action(WAIT)
        if v["trigger"] == "cleared":
            return Action("resolve_event", {}, "Left the zone")
        if st in OPEN_VERIFY:
            if not evidence(v, "trajectory"):
                return Action("check_trajectory", {"track_ids": ev["track_ids"]}, "Is the track stable and still inside the zone?")
            detect = f.get("detect_cameras") or []
            if "second_view" not in v["done"]:
                return Action("second_view", {**_xy(v), "exclude": detect[:1], "kind": "moving"}, "Ask another camera whether someone is there")
            if not evidence(v, "crop") and detect:
                return Action("sharper_crop", {"camera_id": detect[0], **_xy(v), "radius_m": 1.0}, "Keep a blurred close-up for the record")
            if has_support(v):
                return Action("confirm_event", {}, "Track is stable and other views agree")
            return Action("reject_event", {}, "Evidence does not support the dwell")
        if not alerted(v, "floor_lead"):
            return Action("send_alert", {"audience": "floor_lead", "message": ev["summary"]}, "Dwell rule confirmed")
        return Action(WAIT)


def _names(cams: list[str]) -> str:
    return ", ".join(cams) if cams else "no camera"


def _verb(items: list, one: str, many: str) -> str:
    return one if len(items) == 1 else many


def _short(zone_name: str | None) -> str:
    """'Cold room (restricted)' -> 'cold room'."""
    return (zone_name or "zone").split(" (")[0].lower()


# --------------------------------------------------------------------------- #
# Guard (applies to any planner, mirrors the local policy's hard rules)
# --------------------------------------------------------------------------- #


def guard(view: dict, action: Action, known_cameras: list[str] | None = None) -> tuple[bool, str]:
    """Is this action allowed given the evidence so far? Returns (ok, why_not)."""
    ev, f = view["event"], view["facts"]
    st, etype = ev["status"], ev["type"]
    tool, inp = action.tool, action.input
    if tool not in ALL_TOOLS:
        return False, f"unknown tool {tool!r}"
    if tool == "wait":
        return True, ""
    if st in CLOSED and tool not in ("add_rule",):
        return False, f"event is already {st}"
    if tool in VISION_TOOLS + ANALYSIS_TOOLS and view["done_this_trigger"].count(tool) >= 2:
        return False, f"{tool} already called twice for this trigger"
    known_cameras = known_cameras if known_cameras is not None else view.get("cameras")
    if known_cameras is not None and tool in ("sharper_crop", "recalibrate") and inp.get("camera_id") not in known_cameras:
        return False, f"unknown camera {inp.get('camera_id')!r}"
    if tool == "second_view" and inp.get("kind") not in ("static", "moving"):
        return False, "second_view kind must be 'static' or 'moving'"
    if tool == "recalibrate" and etype != "camera_moved":
        return False, "recalibrate is only for camera_moved events"
    if tool in ("short_replay", "redetect", "recommend") and etype != "floor_warning":
        return False, f"{tool} is only for floor warnings"
    if tool == "recommend" and inp.get("verdict") not in ("confirm", "reject", "uncertain"):
        return False, "recommend verdict must be confirm, reject or uncertain"
    if tool == "recommend" and inp.get("verdict") != "uncertain":
        supported, _ = floor_recommendation(view)
        if inp["verdict"] != supported:
            return False, "recommendation is not supported by both checks; ask for human review"
    if tool == "confirm_event":
        if st not in OPEN_VERIFY:
            return False, f"cannot confirm an event that is {st}"
        if not has_support(view):
            return False, "no supporting evidence from an independent source - cannot confirm"
    if tool == "reject_event":
        if st not in OPEN_VERIFY:
            return False, f"cannot reject an event that is {st}"
        if has_support(view):
            return False, "evidence supports the event - cannot reject it"
        if etype != "near_miss" and not has_contradiction(view):
            return False, "no contradicting evidence - ask a human instead of rejecting"
    if tool == "send_alert":
        audience = inp.get("audience")
        if audience not in ("floor_lead", "manager"):
            return False, "audience must be floor_lead or manager"
        dispatch_ok = approved(view, "dispatch_check") or approved(view, "recalibrate")
        if st not in ACTIVE and not dispatch_ok:
            return False, "alerts need a confirmed event or an approved dispatch"
        if audience == "manager" and not approved(view, "escalate"):
            return False, "the manager is only alerted after a human approves escalation"
        if alerted(view, audience):
            return False, f"{audience} already alerted for this event"
    if tool == "request_approval":
        act = inp.get("action")
        if act not in APPROVALS_BY_TYPE.get(etype, ()):
            return False, f"approval {act!r} does not apply to {etype}"
        prev = approval(view, act)
        if prev is not None and prev["status"] == "pending":
            return False, f"an approval for {act} is already pending"
        if act == "escalate" and (st not in ACTIVE or not (f.get("still_present") or view["trigger"] == "follow_up")):
            return False, "escalation needs a confirmed event that is still present"
        if act == "create_rule" and (f.get("history_count", 0) < 3 or f.get("hotspot_proposed")):
            return False, "a standing fix needs 3 confirmed incidents at a spot without an existing proposal"
    if tool == "confirm_event" and etype == "floor_warning" and not approved(view, "confirm_incident"):
        return False, "floor warnings are confirmed only by a human verdict"
    if tool == "reject_event" and etype == "floor_warning" and not has_contradiction(view):
        return False, "floor warnings are rejected only by a human verdict"
    if tool == "resolve_event":
        ok = (
            view["trigger"] in ("gone", "over", "cleared", "camera_ok")
            or (etype == "camera_moved" and (f.get("recalibrate") or {}).get("ok"))
        )
        if not ok:
            return False, "nothing shows the incident is over"
    if tool == "add_rule" and not approved(view, "create_rule"):
        return False, "rules proposed by the agent need human approval"
    if tool == "schedule_follow_up":
        if st not in ACTIVE:
            return False, "follow-ups are for confirmed events"
        s = inp.get("seconds")
        if not isinstance(s, (int, float)) or not 5 <= s <= 600:
            return False, "follow-up must be 5..600 s"
    return True, ""
