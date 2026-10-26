"""FloorsightAgent: rules -> events -> verify-then-act, with human approvals.

    agent = FloorsightAgent(site, vision)
    for packets in feed:
        msgs = agent.on_world(vision.process(packets))

Every tool call and decision is recorded as an AgentStep. Time comes only from
WorldState.t (and the `t` passed to add_rule_from_text / decide), never the
wall clock, so replaying the same footage gives the same trace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from floorsight.contracts import (
    XY,
    AgentStep,
    Alert,
    ApprovalRequest,
    ApprovalStatus,
    CameraStatus,
    Event,
    EventStatus,
    EventType,
    Evidence,
    LiveMessage,
    Rule,
    RuleType,
    SitePlan,
    TrackClass,
    TrackState,
    WorldState,
    ZoneKind,
)
from floorsight.interfaces import Notifier, VisionTools
from floorsight.formatting import floor_error_text
from floorsight.site import point_in_polygon, zone_by_id

from floorsight.agent.language import ParseResult, describe, zone_question
from floorsight.agent.planner import Action, Planner, floor_headline, guard
from floorsight.agent.rules import BLOCKING_CLASSES, NEAR_MISS_MERGE_S, Candidate, RulesEngine

MAX_STEPS_PER_TRIGGER = 10
FOLLOW_UP_S = 30.0
GONE_S = 5.0  # a blocked exit is clear after the object has been gone this long
CLEARED_S = 2.0  # unseen / dwell incidents end after the condition has been false this long
HOTSPOT_RADIUS_M = 3.0
MIN_TRACK_AGE_S = 1.0
MAX_TRACK_SIGMA_M = 0.35
SLOW_ZONE_DISTANCE_M = 5.0


@dataclass
class Case:
    """The agent's working state for one event."""

    event: Event
    key: tuple
    rule: Rule | None
    detect_cams: list[str]
    data: dict
    last_hit: float
    step_n: int = 0
    done: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    alerted: set[str] = field(default_factory=set)
    follow_up_t: float | None = None
    follow_up_fired: bool = False
    absent_since: float | None = None
    cleared_since: float | None = None
    t_end: float | None = None
    was_confirmed: bool = False


def _dump(model) -> dict:
    return model.model_dump(mode="json")


def _point_polygon_distance(xy: XY, poly: list[XY]) -> float:
    if point_in_polygon(xy[0], xy[1], poly):
        return 0.0
    best = math.inf
    px, py = xy
    for i in range(len(poly)):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % len(poly)]
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        a = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / L2))
        best = min(best, math.hypot(px - (x0 + a * dx), py - (y0 + a * dy)))
    return best


class FloorsightAgent:
    """Implements floorsight.interfaces.AgentRuntime."""

    def __init__(
        self,
        site: SitePlan,
        vision: VisionTools,
        notifier: Notifier | None = None,
        planner: Planner | None = None,
        rule_parser=None,
        follow_up_s: float = FOLLOW_UP_S,
        gone_s: float = GONE_S,
    ):
        from floorsight.llm import select_planner, select_rule_parser

        self.site = site
        self.vision = vision
        self.notifier = notifier
        self.planner = planner if planner is not None else select_planner()
        self.rule_parser = rule_parser if rule_parser is not None else select_rule_parser(site)
        self.follow_up_s = follow_up_s
        self.gone_s = gone_s
        self.engine = RulesEngine(site)
        self.cases: dict[str, Case] = {}
        self.case_by_key: dict[tuple, str] = {}
        self.approvals: dict[str, ApprovalRequest] = {}
        self.alerts: list[Alert] = []
        self.trace: list[AgentStep] = []
        self.rule_log: list[dict] = []
        self.proposals: list[dict] = []  # hotspots the agent already proposed a fix for
        self.t = 0.0
        self.t_first: float | None = None
        self.world: WorldState | None = None
        self._n = {"event": 0, "approval": 0, "alert": 0, "lang": 0, "agent_rule": 0}
        self._rule_steps: dict[str, int] = {}
        self._camera_ids = [c.camera_id for c in site.cameras]

    # ------------------------------------------------------------------ #
    # AgentRuntime
    # ------------------------------------------------------------------ #

    def on_world(self, world: WorldState) -> list[LiveMessage]:
        self.world = world
        self.t = world.t
        if self.t_first is None:
            self.t_first = world.t
        out: list[LiveMessage] = []
        new, cont = self.engine.update(world)
        for c in cont:
            eid = self.case_by_key.get(c.key)
            if eid:
                case = self.cases[eid]
                case.last_hit = world.t
                if c.type == EventType.near_miss:
                    d = c.data.get("distance_m")
                    if d is not None and d < case.data.get("min_distance_m", math.inf):
                        case.data["min_distance_m"] = d
                if c.type == EventType.floor_warning:
                    d = c.data.get("clearance_m")
                    if d is not None and d < case.data.get("min_clearance_m", math.inf):
                        case.data["min_clearance_m"] = d
                    if c.data.get("level") == "danger":
                        case.data["max_level"] = "danger"
        for c in new:
            out += self._open_case(c)
        for case in list(self.cases.values()):
            out += self._monitor(case)
        return out

    def add_rule_from_text(self, text: str, t: float) -> list[LiveMessage]:
        self.t = max(self.t, t)
        res: ParseResult = self.rule_parser.parse(text)
        if res.rule is None:
            step = self._rule_step("rule:-", "parse_rule", {"text": text}, {"ok": False, "message": res.message}, "no rule created", res.message)
            return [step, LiveMessage(kind="rule", data={"ok": False, "text": text, "message": res.message})]
        self._n["lang"] += 1
        rule = res.rule.model_copy(update={"rule_id": f"R-L{self._n['lang']:03d}", "source": "language", "enabled": False, "confirmed": False})
        self.engine.add_rule(rule)
        zone = res.zone
        ap = self._new_approval(
            event_id=None,
            action="confirm_rule",
            summary=f"New rule: {describe(rule, self.site)}. {zone_question(zone)}",
            payload={
                "rule_id": rule.rule_id,
                "zone_id": rule.zone_id,
                "zone_name": zone.name if zone else None,
                "question": zone_question(zone),
                "alternatives": res.alternatives,
                "assumptions": res.assumptions,
                "rule": _dump(rule),
            },
        )
        step = self._rule_step(
            f"rule:{rule.rule_id}",
            "parse_rule",
            {"text": text, "parser": getattr(self.rule_parser, "mode", "local")},
            {"ok": True, "rule": _dump(rule), "zone_id": rule.zone_id, "alternatives": res.alternatives},
            f"created {rule.rule_id} (disabled until confirmed), asked {ap.approval_id}",
            res.message,
        )
        return [step, self._rule_msg(rule), self._approval_msg(ap)]

    def decide(self, approval_id: str, approved: bool, t: float, reviewer: str | None = None) -> list[LiveMessage]:
        self.t = max(self.t, t)
        ap = self.approvals.get(approval_id)
        if ap is None or ap.status != ApprovalStatus.pending:
            return []
        ap.status = ApprovalStatus.approved if approved else ApprovalStatus.rejected
        ap.decided_t = t
        out = [self._approval_msg(ap)]
        if ap.event_id and ap.event_id in self.cases:
            # The human verdict is part of the trace judges read: who decided, what, and when.
            out.append(
                self._step(
                    self.cases[ap.event_id],
                    Action("human_decision", {"approval_id": approval_id, "action": ap.action, "approved": approved, "reviewer": reviewer or "unknown"}),
                    {"status": ap.status.value, "decided_t": t},
                    "approved" if approved else "rejected",
                    f"decided by {reviewer or 'a human'} in the dashboard",
                )
            )
        if ap.action == "confirm_rule":
            rule = self.engine.get_rule(ap.payload.get("rule_id"))
            if rule is not None:
                rule.enabled = approved
                rule.confirmed = approved
                out.append(
                    self._rule_step(
                        f"rule:{rule.rule_id}",
                        "decide_rule",
                        {"approval_id": approval_id, "approved": approved},
                        {"enabled": rule.enabled},
                        "rule enabled" if approved else "rule left disabled",
                        "Human confirmed the rule and its zone" if approved else "Human rejected the rule",
                    )
                )
                out.append(self._rule_msg(rule))
            return out
        if ap.event_id and ap.event_id in self.cases:
            out += self._run(self.cases[ap.event_id], "approval")
        return out

    def snapshot(self) -> dict:
        return {
            "t": self.t,
            "mode": getattr(self.planner, "mode", "local"),
            "rules": [_dump(r) for r in self.engine.rules],
            "events": [_dump(c.event) for c in self.cases.values()],
            "approvals": [_dump(a) for a in self.approvals.values()],
            "alerts": [_dump(a) for a in self.alerts],
            "trace": [_dump(s) for s in self.trace],
            "remembered_spots": [vars(s) | {"xy": list(s.xy)} for s in self.engine.spots],
        }

    def shift_report(self) -> dict:
        from floorsight.agent.report import build_shift_report

        return build_shift_report(self)

    # ------------------------------------------------------------------ #
    # Cases
    # ------------------------------------------------------------------ #

    def _open_case(self, c: Candidate) -> list[LiveMessage]:
        self._n["event"] += 1
        eid = f"EV-{self._n['event']:04d}"
        ev = Event(
            event_id=eid,
            type=c.type,
            rule_id=c.rule.rule_id if c.rule else None,
            t_start=c.t_start,
            t_detect=c.t,
            location=c.location,
            zone_id=c.zone_id,
            track_ids=c.track_ids,
            camera_ids=c.camera_ids,
            severity=c.severity,
            status=EventStatus.candidate,
            summary=c.summary,
        )
        case = Case(event=ev, key=c.key, rule=c.rule, detect_cams=c.camera_ids[:1], data=dict(c.data), last_hit=c.t)
        if c.type == EventType.floor_warning:
            case.data["min_clearance_m"] = c.data.get("clearance_m")
            case.data["max_level"] = c.data.get("level")
        if c.type == EventType.near_miss:
            case.data["min_distance_m"] = c.data.get("distance_m")
            case.detect_cams = self._track_cams(c.track_ids[:1]) or c.camera_ids[:1]
        if c.type == EventType.camera_moved:
            case.facts["calib_error_before_m"] = c.data.get("calib_error_m")
            ev.evidence.append(
                Evidence(
                    kind="calibration",
                    camera_id=c.camera_ids[0],
                    t=c.t,
                    supports=True,
                    score=c.data.get("calib_error_m"),
                    note=f"Health report: status moved, floor error {floor_error_text(c.data.get('calib_error_m'))}, {c.data.get('markers_visible')} markers visible",
                )
            )
        self.cases[eid] = case
        self.case_by_key[c.key] = eid
        return [self._event_msg(case)] + self._run(case, "new")

    def pending_work(self) -> bool:
        """True while a second detector pass started by the agent has not reported back."""
        return any(c.facts.get("redetect_pending") for c in self.cases.values())

    def drain(self) -> list[LiveMessage]:
        """Collect finished second detector passes without a new frame (the stream has ended)."""
        out: list[LiveMessage] = []
        for case in list(self.cases.values()):
            if case.facts.get("redetect_pending"):
                out += self._monitor(case)
        return out

    def _monitor(self, case: Case) -> list[LiveMessage]:
        ev = case.event
        st = ev.status.value
        if st in ("resolved", "rejected"):
            return []
        t = self.t
        token = case.facts.get("_redetect_token")
        if token and case.facts.get("redetect_pending"):
            e = self.vision.redetect_poll(token)
            if e is None:
                return []
            case.facts["redetect_pending"] = False
            ev.evidence.append(e)
            verdict = {True: "agrees", False: "disagrees", None: "inconclusive"}[e.supports]
            out = [self._step(case, Action("redetect_result", {"token": token}), {"evidence": _dump(e)}, f"second detector pass {verdict}",
                              "result of the second detector pass that ran beside the stream"), self._event_msg(case)]
            return out + self._run(case, "tool_done")
        if ev.type == EventType.blocked_exit:
            if self._object_present(case):
                case.absent_since = None
            else:
                case.absent_since = case.absent_since if case.absent_since is not None else t
                if t - case.absent_since >= self.gone_s:
                    case.t_end = case.absent_since
                    return self._run(case, "gone")
            if case.follow_up_t is not None and not case.follow_up_fired and t >= case.follow_up_t:
                case.follow_up_fired = True
                return self._run(case, "follow_up")
        elif ev.type in (EventType.near_miss, EventType.floor_warning):
            if t - case.last_hit > NEAR_MISS_MERGE_S and st in ("confirmed", "escalated"):
                case.t_end = case.last_hit
                return self._run(case, "over")
        elif ev.type == EventType.camera_moved:
            h = next((h for h in (self.world.cameras if self.world else []) if h.camera_id == ev.camera_ids[0]), None)
            if h is not None and h.status == CameraStatus.ok and "recalibrate" in case.done:
                return self._run(case, "camera_ok")
        elif ev.type in (EventType.unseen_in_zone, EventType.dwell):
            if self._condition_holds(case):
                case.cleared_since = None
            else:
                case.cleared_since = case.cleared_since if case.cleared_since is not None else t
                if t - case.cleared_since >= CLEARED_S:
                    case.t_end = case.cleared_since
                    return self._run(case, "cleared")
        return []

    def _run(self, case: Case, trigger: str) -> list[LiveMessage]:
        out: list[LiveMessage] = []
        done_this: list[str] = []
        for _ in range(MAX_STEPS_PER_TRIGGER):
            view = self._view(case, trigger, done_this)
            action = self.planner.next_action(view)
            ok, why = guard(view, action, self._camera_ids)
            if not ok:
                # Planners already guard themselves; this is the last line of defence.
                out.append(self._step(case, action, {"blocked": why}, "refused by guard", why))
                break
            if action.tool == "wait":
                break
            out += self._execute(case, action)
            done_this.append(action.tool)
        return out

    def _view(self, case: Case, trigger: str, done_this: list[str]) -> dict:
        ev = case.event
        zone = zone_by_id(self.site, ev.zone_id) if ev.zone_id else None
        facts = {
            "detect_cameras": case.detect_cams,
            "zone_name": zone.name if zone else None,
            "alerted": sorted(case.alerted),
            "follow_up_s": self.follow_up_s,
            "follow_up_t": case.follow_up_t,
            **{k: v for k, v in case.facts.items() if not k.startswith("_")},
            **{k: v for k, v in case.data.items() if k in ("distance_m", "min_distance_m", "closing_mps", "static_s", "threshold_s", "unseen_s", "dwell_s")},
        }
        if ev.type == EventType.blocked_exit:
            facts["still_present"] = self._object_present(case)
            facts["absent_s"] = round(self.t - case.absent_since, 2) if case.absent_since is not None else 0.0
        if ev.type == EventType.floor_warning:
            facts["level"] = case.data.get("max_level") or case.data.get("level")
            facts["min_clearance_m"] = case.data.get("min_clearance_m")
            facts["cut_by_frame"] = bool(case.data.get("cut_by_frame"))
            facts["since_last_hit_s"] = round(self.t - case.last_hit, 2)
            facts["recommendation"] = case.facts.get("recommendation")
        if ev.type == EventType.near_miss:
            facts["geometry_supported"] = case.data.get("geometry_supported", False)
            facts["since_last_hit_s"] = round(self.t - case.last_hit, 2)
            facts["hotspot_proposed"] = self._hotspot_proposed(ev.location)
            facts["counts_for_history"] = self._counts_for_history(case)
            facts["rule_added"] = bool(case.facts.get("rule_added"))
            if case.facts.get("history_count", 0) >= 3:
                facts["proposal_text"] = self._proposed_rule(case).text
        return {
            "t": self.t,
            "trigger": trigger,
            "event": _dump(ev),
            "facts": facts,
            "approvals": [_dump(a) for a in self.approvals.values() if a.event_id == ev.event_id],
            "done": list(case.done),
            "done_this_trigger": list(done_this),
            "cameras": list(self._camera_ids),
        }

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #

    def _execute(self, case: Case, a: Action) -> list[LiveMessage]:
        ev = case.event
        before = (ev.status, len(ev.evidence))
        extra: list[LiveMessage] = []
        if ev.status == EventStatus.candidate and a.tool not in ("wait",):
            ev.status = EventStatus.verifying
        fn = getattr(self, "_do_" + a.tool)
        output, decision = fn(case, a, extra)
        case.done.append(a.tool)
        out = [self._step(case, a, output, decision, a.reason)]
        if (ev.status, len(ev.evidence)) != before:
            out.append(self._event_msg(case))
        return out + extra

    def _xy(self, case: Case, a: Action) -> XY:
        loc = case.event.location
        return (float(a.input.get("x", loc[0])), float(a.input.get("y", loc[1])))

    def _do_sharper_crop(self, case, a, extra):
        cam = a.input.get("camera_id") or (case.detect_cams or self._camera_ids)[0]
        e = self.vision.sharper_crop(cam, self._xy(case, a), float(a.input.get("radius_m", 1.0)))
        case.event.evidence.append(e)
        return {"evidence": _dump(e)}, f"close-up from {cam} stored" + (f" ({e.image_ref})" if e.image_ref else "")

    def _do_second_view(self, case, a, extra):
        xy = self._xy(case, a)
        exclude = list(a.input.get("exclude") or [])
        kind = a.input.get("kind", "static")
        views = self.vision.second_view(xy, exclude, kind)
        case.event.evidence.extend(views)
        sup = [e.camera_id for e in views if e.supports]
        con = [e.camera_id for e in views if e.supports is False]
        if not views:
            decision = "no other camera covers this spot"
        else:
            decision = f"supports: {', '.join(sup) or 'none'}; contradicts: {', '.join(con) or 'none'}"
        return {"views": [_dump(e) for e in views], "supporting": sup, "contradicting": con}, decision

    def _do_short_replay(self, case, a, extra):
        fn = getattr(self.vision, "short_replay", None)
        ids = list(a.input.get("track_ids") or case.event.track_ids)
        if callable(fn):
            e = fn(ids, float(a.input.get("seconds", 1.5)))
        else:
            e = Evidence(kind="replay", t=self.t, supports=None, note="short replay not available on this vision layer")
        case.event.evidence.append(e)
        verdict = {True: "sustained", False: "not sustained / artefact", None: "inconclusive"}[e.supports]
        return {"evidence": _dump(e)}, f"replay: {verdict}"

    def _do_redetect(self, case, a, extra):
        ids = list(a.input.get("track_ids") or case.event.track_ids)
        start = getattr(self.vision, "redetect_start", None)
        if callable(start):  # runs beside the live stream; the result arrives through _monitor (step redetect_result)
            case.facts["_redetect_token"] = start(ids)
            case.facts["redetect_pending"] = True
            return {"started": True}, "second detector pass started (result in a few seconds)"
        fn = getattr(self.vision, "redetect", None)
        if callable(fn):
            e = fn(ids)
        else:
            e = Evidence(kind="redetect", t=self.t, supports=None, note="second detector pass not available on this vision layer")
        case.event.evidence.append(e)
        verdict = {True: "agrees", False: "disagrees", None: "inconclusive"}[e.supports]
        return {"evidence": _dump(e)}, f"second detector pass {verdict}" + (f" ({e.image_ref})" if e.image_ref else "")

    def _do_recommend(self, case, a, extra):
        verdict = a.input.get("verdict", "uncertain")
        headline = a.input.get("headline")
        if not headline and case.event.type == EventType.floor_warning:  # e.g. the Bedrock planner sends only the verdict
            headline = floor_headline({"event": {"evidence": [_dump(e) for e in case.event.evidence]}}, verdict)
        case.facts["recommendation"] = {"verdict": verdict, "reason": a.reason, "headline": headline, "t": self.t}
        return {"verdict": verdict}, f"recommends {verdict}"

    def _floor_card(self, case) -> dict:
        """Decision card for a floor warning: one-line verdict plus the three numbers a reviewer needs."""
        rec = case.facts.get("recommendation") or {}
        verdict = rec.get("verdict")
        al = case.data.get("alert") or {}
        clear, ttc, pas = (al.get("clearance_m") or {}).get("median"), (al.get("ttc_s") or {}).get("median"), al.get("pass_m") or {}
        if ttc is not None:
            when = "now" if ttc < 0.05 else f"in {ttc:.1f} s"
        else:
            when = "already in its danger zone" if al.get("level") == "danger" else "n/a"
        closest = "n/a"
        if pas.get("median") is not None:
            closest = f"{pas['median']:.2f} m"
            if pas.get("p5") is not None and pas.get("p95") is not None:
                closest += f" ({pas['p5']:.2f}-{pas['p95']:.2f})"
        default = {"confirm": "Agent recommends CONFIRM", "reject": "Agent recommends REJECT (false alarm)"}.get(verdict, "Agent is UNCERTAIN")
        return {
            "verdict": verdict or "uncertain",
            "headline": rec.get("headline") or default,
            "numbers": [
                {"label": "Floor gap at the warning", "value": f"{clear:.1f} m" if clear is not None else "n/a"},
                {"label": "Within 0.8 m", "value": when},
                {"label": "Closest predicted pass", "value": closest},
            ],
        }

    def _do_cameras_covering(self, case, a, extra):
        cams = list(self.vision.cameras_covering(self._xy(case, a)))
        case.facts["covering"] = cams
        return {"cameras": cams}, f"covered by {', '.join(cams)}" if cams else "blind spot: no camera covers this point"

    def _do_recalibrate(self, case, a, extra):
        cam = a.input.get("camera_id") or case.event.camera_ids[0]
        res = dict(self.vision.recalibrate(cam))
        case.facts["recalibrate"] = res
        before = case.facts.get("calib_error_before_m")
        case.event.evidence.append(
            Evidence(
                kind="calibration",
                camera_id=cam,
                t=self.t,
                supports=True,
                score=res.get("calib_error_m"),
                note=f"Recalibration {'succeeded' if res.get('ok') else 'failed'}: floor error {floor_error_text(before)} -> {floor_error_text(res.get('calib_error_m'))} using {res.get('markers_used', 0)} markers",
            )
        )
        return {**res, "calib_error_before_m": before}, "re-aligned" if res.get("ok") else "could not re-align"

    def _do_check_trajectory(self, case, a, extra):
        ev = case.event
        ids = list(a.input.get("track_ids") or ev.track_ids)
        tracks = {tr.track_id: tr for tr in (self.world.tracks if self.world else [])}
        rows, ok = [], True
        for tid in ids:
            tr = tracks.get(tid)
            if tr is None:
                rows.append({"track_id": tid, "found": False})
                ok = False
                continue
            age = round(self.t - tr.first_seen_t, 2)
            good = tr.state == TrackState.seen and age >= MIN_TRACK_AGE_S and tr.sigma_m <= MAX_TRACK_SIGMA_M
            ok = ok and good
            rows.append({"track_id": tid, "found": True, "state": tr.state.value, "age_s": age, "sigma_m": tr.sigma_m, "ok": good})
        if ev.type == EventType.dwell:
            zone = zone_by_id(self.site, ev.zone_id or "")
            inside = zone is not None and all(t.get("found") and point_in_polygon(tracks[t["track_id"]].x, tracks[t["track_id"]].y, zone.polygon) for t in rows)
            ok = ok and inside
        d, c = case.data.get("distance_m"), case.data.get("closing_mps")
        if ev.type == EventType.near_miss:
            ok = ok and case.data.get("geometry_supported", False)
        parts = [f"{r['track_id']}: " + (f"seen {r['age_s']:.1f} s, sigma {r['sigma_m']:.2f} m" if r.get("found") else "not found") for r in rows]
        if d is not None:
            parts.append(f"gap {d:.2f} m, closing {c:.1f} m/s")
        if ev.type == EventType.near_miss:
            parts.append(f"observed heading baseline {case.data.get('heading_baseline_m')} m; temporal clearance supports={case.data.get('geometry_supported', False)}")
        e = Evidence(kind="trajectory", t=self.t, supports=ok, score=d, note="; ".join(parts))
        ev.evidence.append(e)
        return {"tracks": rows, "distance_m": d, "closing_mps": c, "supports": ok}, "trajectory supports the event" if ok else "trajectory too weak"

    def _do_incident_history(self, case, a, extra):
        xy = self._xy(case, a)
        radius = float(a.input.get("radius_m", HOTSPOT_RADIUS_M))
        ids = [
            c.event.event_id
            for c in self.cases.values()
            if c.event.type == EventType.near_miss and c.was_confirmed and self._counts_for_history(c) and math.dist(c.event.location, xy) <= radius
        ]
        case.facts["history_count"] = len(ids)
        case.facts["history_event_ids"] = ids
        e = Evidence(kind="history", t=self.t, supports=len(ids) >= 3, score=float(len(ids)), note=f"{len(ids)} confirmed near-misses within {radius:g} m: {', '.join(ids)}")
        case.event.evidence.append(e)
        return {"count": len(ids), "event_ids": ids, "radius_m": radius}, f"{len(ids)} confirmed near-miss(es) here this shift"

    def _do_confirm_event(self, case, a, extra):
        case.event.status = EventStatus.confirmed
        case.was_confirmed = True
        return {"status": "confirmed"}, "confirmed"

    def _do_reject_event(self, case, a, extra):
        ev = case.event
        ev.status = EventStatus.rejected
        case.t_end = self.t
        out: dict = {"status": "rejected"}
        decision = "rejected as false alarm"
        if ev.type == EventType.near_miss:
            out["reason_category"] = "insufficient_evidence"
            out["safety_conclusion"] = "unknown"
            decision = "not confirmed: insufficient trajectory or clearance evidence; safety remains unknown"
        if ev.type == EventType.floor_warning:
            out["reason_category"] = "human_verdict_false_alarm"
            decision = "rejected by the human reviewer as a false alarm"
        if ev.type == EventType.blocked_exit and ev.zone_id:
            old = case.data.get("threshold_s")
            spot = self.engine.remember_false_alarm(ev.zone_id, ev.location, case.detect_cams, self.t, ev.event_id, a.reason)
            new = self.engine.spot_penalty(ev.zone_id, ev.location, case.detect_cams) + (case.rule.seconds if case.rule and case.rule.seconds else 0.0)
            out["remembered_spot"] = {"zone_id": spot.zone_id, "xy": list(spot.xy), "camera_ids": spot.camera_ids, "extra_s": spot.extra_s}
            out["threshold_s"] = {"before": old, "after": new}
            decision += f"; remembered spot ({spot.xy[0]:.2f}, {spot.xy[1]:.2f}) for {', '.join(spot.camera_ids)}: threshold {old:g} s -> {new:g} s"
        return out, decision

    def _do_send_alert(self, case, a, extra):
        audience = a.input.get("audience", "floor_lead")
        self._n["alert"] += 1
        alert = Alert(
            alert_id=f"AL-{self._n['alert']:04d}",
            event_id=case.event.event_id,
            t=self.t,
            channel="dashboard",
            audience=audience,
            message=a.input.get("message") or case.event.summary,
        )
        self.alerts.append(alert)
        case.alerted.add(audience)
        if self.notifier is not None:
            self.notifier.send(alert)
        if audience == "manager":
            case.event.status = EventStatus.escalated
        extra.append(LiveMessage(kind="alert", data=_dump(alert)))
        return {"alert": _dump(alert)}, f"alerted {audience.replace('_', ' ')}" + (" - escalated" if audience == "manager" else "")

    def _do_request_approval(self, case, a, extra):
        act = a.input["action"]
        payload: dict = {"zone_id": case.event.zone_id, "location": list(case.event.location)}
        if act == "create_rule":
            rule = self._proposed_rule(case)
            payload.update({"rule": _dump(rule), "zone_id": rule.zone_id, "hotspot": list(case.event.location), "event_ids": case.facts.get("history_event_ids", [])})
            self.proposals.append({"xy": case.event.location, "event_id": case.event.event_id, "rule_id": rule.rule_id})
        if act == "recalibrate":
            payload["camera_id"] = case.event.camera_ids[0]
        if act == "confirm_incident" and case.event.type == EventType.floor_warning:
            payload["card"] = self._floor_card(case)
        ap = self._new_approval(case.event.event_id, act, a.input.get("summary") or a.reason, payload)
        extra.append(self._approval_msg(ap))
        return {"approval": _dump(ap)}, f"asked a human: {act} ({ap.approval_id})"

    def _do_schedule_follow_up(self, case, a, extra):
        case.follow_up_t = round(self.t + float(a.input.get("seconds", self.follow_up_s)), 3)
        case.follow_up_fired = False
        return {"follow_up_t": case.follow_up_t}, f"follow-up at t={case.follow_up_t:.1f} s"

    def _do_resolve_event(self, case, a, extra):
        case.event.status = EventStatus.resolved
        for ap in self.approvals.values():
            if ap.event_id == case.event.event_id and ap.status == ApprovalStatus.pending and ap.action != "create_rule":
                ap.status = ApprovalStatus.expired
                ap.decided_t = self.t
                extra.append(self._approval_msg(ap))
        if case.t_end is None:
            case.t_end = self.t
        return {"status": "resolved", "t_end": case.t_end}, "resolved"

    def _do_add_rule(self, case, a, extra):
        ap = self.approvals.get(a.input.get("approval_id", ""))
        if ap is None:
            ap = next(x for x in self.approvals.values() if x.event_id == case.event.event_id and x.action == "create_rule")
        rule = Rule.model_validate(ap.payload["rule"])
        self.engine.add_rule(rule)
        case.facts["rule_added"] = rule.rule_id
        extra.append(self._rule_msg(rule))
        return {"rule": _dump(rule)}, f"standing rule {rule.rule_id} added"

    def _do_wait(self, case, a, extra):
        return {}, "wait"

    # ------------------------------------------------------------------ #
    # World queries
    # ------------------------------------------------------------------ #

    def _track_cams(self, ids: list[str]) -> list[str]:
        for tr in self.world.tracks if self.world else []:
            if tr.track_id in ids:
                return list(tr.camera_ids)
        return []

    def _object_present(self, case: Case) -> bool:
        ev = case.event
        zone = zone_by_id(self.site, ev.zone_id or "")
        for tr in self.world.tracks if self.world else []:
            classes = set(case.rule.classes) if case.rule and case.rule.classes else set(BLOCKING_CLASSES)
            if tr.cls not in classes or tr.state == TrackState.lost:
                continue
            if math.hypot(tr.vx, tr.vy) > 0.3 or math.dist((tr.x, tr.y), ev.location) > 1.5:
                continue
            if zone is None or point_in_polygon(tr.x, tr.y, zone.polygon):
                return True
        return False

    def _condition_holds(self, case: Case) -> bool:
        ev = case.event
        zone = zone_by_id(self.site, ev.zone_id or "")
        if ev.type == EventType.unseen_in_zone and zone is not None:
            # The tracker may re-number a person it cannot see; the incident is about the zone, not the id.
            return any(
                tr.state == TrackState.unseen
                and tr.cls in (TrackClass.person, TrackClass.unknown)
                and point_in_polygon(tr.x, tr.y, zone.polygon)
                for tr in (self.world.tracks if self.world else [])
            )
        tr = next((tr for tr in (self.world.tracks if self.world else []) if tr.track_id in ev.track_ids), None)
        if tr is None or zone is None or not point_in_polygon(tr.x, tr.y, zone.polygon):
            return False
        if ev.type == EventType.unseen_in_zone:
            return tr.state == TrackState.unseen
        return tr.state == TrackState.seen

    def _counts_for_history(self, case: Case) -> bool:
        """Only free-floor near-miss rules count toward hotspots (not the slow zones they create)."""
        return case.event.type == EventType.near_miss and (case.rule is None or case.rule.zone_id is None)

    def _hotspot_proposed(self, xy: XY) -> bool:
        return any(math.dist(p["xy"], xy) <= HOTSPOT_RADIUS_M for p in self.proposals)

    def _hotspot_zone(self, xy: XY):
        cands = [z for z in self.site.zones if z.kind in (ZoneKind.area, ZoneKind.lane)]
        if not cands:
            return None
        scored = sorted(((round(_point_polygon_distance(xy, z.polygon), 2), 0 if z.kind == ZoneKind.area else 1, i, z) for i, z in enumerate(cands)), key=lambda s: s[:3])
        best = scored[0][0]
        # prefer a crossing/area over a long lane when both are about as close
        near = [s for s in scored if s[0] <= best + 0.5]
        return sorted(near, key=lambda s: (s[1], s[0], s[2]))[0][3]

    def _proposed_rule(self, case: Case) -> Rule:
        zone = self._hotspot_zone(case.event.location)
        rid = case.facts.get("_proposed_rule_id")
        if rid is None:
            self._n["agent_rule"] += 1
            rid = case.facts["_proposed_rule_id"] = f"R-A{self._n['agent_rule']:03d}"
        if zone is None:
            text = f"Slow zone around ({case.event.location[0]:.1f}, {case.event.location[1]:.1f}): alert when a person is within {SLOW_ZONE_DISTANCE_M:g} m of a forklift"
            return Rule(rule_id=rid, text=text, type=RuleType.proximity, classes=[TrackClass.person, TrackClass.forklift], distance_m=SLOW_ZONE_DISTANCE_M, source="agent")
        short = "crossing" if "crossing" in zone.name.lower() else zone.name.split(" (")[0].lower()
        text = f"Slow zone at {zone.name}: alert when a person is in the {short} while a forklift is within {SLOW_ZONE_DISTANCE_M:g} m"
        return Rule(
            rule_id=rid,
            text=text,
            type=RuleType.proximity,
            zone_id=zone.zone_id,
            classes=[TrackClass.person, TrackClass.forklift],
            distance_m=SLOW_ZONE_DISTANCE_M,
            closing_mps=None,
            source="agent",
            confirmed=True,
            enabled=True,
        )

    # ------------------------------------------------------------------ #
    # Records and messages
    # ------------------------------------------------------------------ #

    def _new_approval(self, event_id: str | None, action: str, summary: str, payload: dict) -> ApprovalRequest:
        self._n["approval"] += 1
        ap = ApprovalRequest(approval_id=f"AP-{self._n['approval']:04d}", event_id=event_id, action=action, summary=summary, payload=payload, created_t=self.t)
        self.approvals[ap.approval_id] = ap
        return ap

    def _step(self, case: Case, a: Action, output: dict, decision: str, reason: str) -> LiveMessage:
        case.step_n += 1
        step = AgentStep(
            event_id=case.event.event_id,
            step=case.step_n,
            t=self.t,
            tool=a.tool,
            tool_input=dict(a.input),
            tool_output=output,
            decision=decision,
            reason=reason,
            mode="bedrock" if a.mode == "bedrock" else "local",
        )
        self.trace.append(step)
        return LiveMessage(kind="agent_step", data=_dump(step))

    def _rule_step(self, ref: str, tool: str, inp: dict, out: dict, decision: str, reason: str) -> LiveMessage:
        n = self._rule_steps[ref] = self._rule_steps.get(ref, 0) + 1
        mode = "bedrock" if getattr(self.rule_parser, "mode", "local") == "bedrock" and tool == "parse_rule" else "local"
        step = AgentStep(event_id=ref, step=n, t=self.t, tool=tool, tool_input=inp, tool_output=out, decision=decision, reason=reason, mode=mode)
        self.trace.append(step)
        return LiveMessage(kind="agent_step", data=_dump(step))

    def _event_msg(self, case: Case) -> LiveMessage:
        return LiveMessage(kind="event", data=_dump(case.event))

    def _approval_msg(self, ap: ApprovalRequest) -> LiveMessage:
        return LiveMessage(kind="approval", data=_dump(ap))

    def _rule_msg(self, rule: Rule) -> LiveMessage:
        return LiveMessage(kind="rule", data=_dump(rule))
