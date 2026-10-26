"""Scripted stand-in agent for dashboard development (``--agent demo``).

NOT the real agent (that is floorsight.agent.runtime.FloorsightAgent). It replays
the simulator's scripted incidents from truth_events.json as plausible events,
trace steps, approvals and alerts, so every dashboard panel can be exercised
before the real agent exists. It does call the vision tools (second_view,
recalibrate) so the trace shows real tool outputs from whichever vision runs.
"""

from __future__ import annotations

import re

from floorsight.contracts import (
    AgentStep,
    Alert,
    ApprovalRequest,
    ApprovalStatus,
    Event,
    EventStatus,
    EventType,
    Evidence,
    LiveMessage,
    Rule,
    RuleType,
    SitePlan,
    TrackClass,
    WorldState,
)
from floorsight.site import polygon_centroid, zone_by_id


def _msg(kind: str, model) -> LiveMessage:
    return LiveMessage(kind=kind, data=model.model_dump(mode="json"))


class DemoAgent:
    name = "demo"

    def __init__(self, site: SitePlan, vision, truth: dict):
        self.site = site
        self.vision = vision
        self.truth = truth
        self.rules: dict[str, Rule] = {}
        self.events: dict[str, Event] = {}
        self.approvals: dict[str, ApprovalRequest] = {}
        self.alerts: list[Alert] = []
        self.trace: list[AgentStep] = []
        self._queue: list[tuple[float, int, object]] = []
        self._fired: set[int] = set()
        self._n = 0
        self._t = 0.0
        for r in [
            Rule(rule_id="R-exit-clear", text="Exit B must stay clear (nothing parked for more than 20 s)", type=RuleType.blocked, zone_id="exit_b", seconds=20),
            Rule(rule_id="R-near-miss", text="Warn when a person and a forklift get closer than 0.8 m while closing", type=RuleType.proximity, classes=[TrackClass.person, TrackClass.forklift], distance_m=0.8, closing_mps=0.2),
            Rule(rule_id="R-cold-room", text="Tell me when someone is in the cold room (no camera covers it)", type=RuleType.unseen, zone_id="cold_room", classes=[TrackClass.person]),
        ]:
            self.rules[r.rule_id] = r

    # ---- helpers --------------------------------------------------------- #

    def _id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}-{self._n:04d}"

    def _later(self, t: float, fn) -> None:
        self._n += 1
        self._queue.append((t, self._n, fn))
        self._queue.sort(key=lambda q: (q[0], q[1]))

    def _step(self, ev: Event, tool: str, tool_input: dict, tool_output: dict, decision: str, reason: str) -> LiveMessage:
        n = sum(1 for s in self.trace if s.event_id == ev.event_id) + 1
        step = AgentStep(
            event_id=ev.event_id, step=n, t=self._t, tool=tool, tool_input=tool_input,
            tool_output=tool_output, decision=decision, reason=reason, mode="local",
        )
        self.trace.append(step)
        return _msg("agent_step", step)

    def _update(self, ev: Event, **changes) -> LiveMessage:
        ev = ev.model_copy(update=changes)
        self.events[ev.event_id] = ev
        return _msg("event", ev)

    def _alert(self, ev: Event, audience: str, message: str) -> LiveMessage:
        a = Alert(alert_id=self._id("A"), event_id=ev.event_id, t=self._t, channel="dashboard", audience=audience, message=message)
        self.alerts.append(a)
        return _msg("alert", a)

    def _ask(self, ev: Event | None, action: str, summary: str, payload: dict | None = None) -> LiveMessage:
        ap = ApprovalRequest(
            approval_id=self._id("AP"), event_id=ev.event_id if ev else None, action=action,
            summary=summary, payload=payload or {}, created_t=self._t,
        )
        self.approvals[ap.approval_id] = ap
        return _msg("approval", ap)

    def _second_view(self, xy, exclude, kind) -> list[Evidence]:
        try:
            return list(self.vision.second_view(tuple(xy), exclude, kind))
        except Exception:
            return []

    # ---- scripted incidents --------------------------------------------- #

    def _start(self, te: dict) -> list[LiveMessage]:
        kind = te.get("type")
        t0 = float(te.get("t_start", self._t))
        if kind == "false_alarm_source":
            return self._glare(te, t0)
        if kind == "blocked_exit":
            return self._blocked(te, t0)
        if kind == "near_miss":
            return self._near_miss(te, t0)
        if kind == "camera_moved":
            return self._camera_moved(te, t0)
        if kind == "person_in_restricted":
            return self._restricted(te, t0)
        return []

    def _glare(self, te, t0):
        xy = tuple(te.get("location", (15.0, 7.6)))
        cam = te.get("camera_id", "cam3")
        ev = Event(
            event_id=self._id("E"), type=EventType.blocked_exit, rule_id="R-exit-clear", t_start=t0, t_detect=self._t,
            location=xy, zone_id="exit_b", camera_ids=[cam], severity="critical", status=EventStatus.candidate,
            summary=f"Possible object blocking Exit B (only {cam} sees it)",
        )
        self.events[ev.event_id] = ev
        eid = ev.event_id

        def verify():
            e = self.events[eid]
            out = [self._update(e, status=EventStatus.verifying)]
            out.append(self._step(self.events[eid], "sharper_crop", {"camera_id": cam, "xy": list(xy)},
                                  {"note": "bright patch, no edges or shadow"}, "look from another camera",
                                  "One camera is not enough to block an exit alarm."))
            return out

        def decide():
            e = self.events[eid]
            ev_list = self._second_view(xy, [cam], "static")
            supports = [x for x in ev_list if x.supports]
            out = [self._step(e, "second_view", {"xy": list(xy), "exclude": [cam], "kind": "static"},
                              {"views": [x.model_dump(mode="json") for x in ev_list]},
                              "reject" if not supports else "confirm",
                              "No other camera sees an object here; it looks like sun glare on the floor."
                              if not supports else "A second camera agrees.")]
            status = EventStatus.rejected if not supports else EventStatus.confirmed
            out.append(self._update(e, status=status, evidence=e.evidence + ev_list,
                                    summary="False alarm: sun glare in Exit B (second camera sees the exit clear)"
                                    if not supports else e.summary))
            return out

        self._later(t0 + 3, verify)
        self._later(t0 + 6, decide)
        return [_msg("event", ev)]

    def _blocked(self, te, t0):
        xy = tuple(te.get("location", (15.0, 7.9)))
        ev = Event(
            event_id=self._id("E"), type=EventType.blocked_exit, rule_id="R-exit-clear", t_start=t0, t_detect=self._t,
            location=xy, zone_id=te.get("zone_id", "exit_b"), camera_ids=["cam2", "cam3"], severity="critical",
            status=EventStatus.candidate, summary="Something is parked in Exit B",
        )
        self.events[ev.event_id] = ev
        eid = ev.event_id

        def verify():
            e = self.events[eid]
            ev_list = self._second_view(xy, ["cam3"], "static")
            out = [self._update(e, status=EventStatus.verifying)]
            out.append(self._step(self.events[eid], "second_view", {"xy": list(xy), "exclude": ["cam3"], "kind": "static"},
                                  {"views": [x.model_dump(mode="json") for x in ev_list]}, "confirm",
                                  "Two cameras see the same object at the same spot for over 20 s."))
            out.append(self._update(self.events[eid], status=EventStatus.confirmed, evidence=ev_list,
                                    summary="Exit B is blocked by a pallet (seen by 2 cameras)"))
            out.append(self._alert(self.events[eid], "floor_lead", "Exit B is blocked by a pallet. Please clear it."))
            return out

        def ask():
            e = self.events[eid]
            if e.status != EventStatus.confirmed:
                return []
            return [self._ask(e, "escalate", "Exit B has been blocked for over a minute and nobody has cleared it. "
                                             "Tell the shift manager?", {"audience": "manager", "zone_id": "exit_b"})]

        def resolve():
            e = self.events[eid]
            return [self._update(e, status=EventStatus.resolved, summary="Exit B is clear again (pallet removed)")]

        self._later(t0 + 4, verify)
        self._later(t0 + 60, ask)
        if te.get("t_end") is not None:
            self._later(float(te["t_end"]) + 1, resolve)
        return [_msg("event", ev)]

    def _near_miss(self, te, t0):
        xy = tuple(te.get("location", (11.8, 6.3)))
        d = te.get("min_distance_m")
        ev = Event(
            event_id=self._id("E"), type=EventType.near_miss, rule_id="R-near-miss", t_start=t0, t_detect=self._t,
            location=xy, zone_id="aisle4_crossing", track_ids=[f"T-{te.get('track', '?')}", "T-forklift1"],
            camera_ids=["cam2", "cam3"], severity="warning", status=EventStatus.candidate,
            summary="Person and forklift came very close at the aisle 4 crossing",
        )
        self.events[ev.event_id] = ev
        eid = ev.event_id

        def confirm():
            e = self.events[eid]
            dist = f"{d:.2f} m" if isinstance(d, (int, float)) else "under 0.8 m"
            out = [self._step(e, "trajectory", {"track_ids": e.track_ids}, {"min_distance_m": d, "closing": True},
                              "confirm", f"Both tracks were seen by 2 cameras; closest gap {dist} while closing.")]
            out.append(self._update(e, status=EventStatus.confirmed,
                                    evidence=[Evidence(kind="trajectory", t=self._t, supports=True, score=0.9,
                                                       note=f"closest gap {dist}")]))
            out.append(self._alert(self.events[eid], "floor_lead", "Near miss at the aisle 4 crossing."))
            return out

        self._later(t0 + 1.5, confirm)
        return [_msg("event", ev)]

    def _camera_moved(self, te, t0):
        cam = te.get("camera_id", "cam4")
        ev = Event(
            event_id=self._id("E"), type=EventType.camera_moved, rule_id=None, t_start=t0, t_detect=self._t,
            location=(1.0, 5.0), camera_ids=[cam], severity="warning", status=EventStatus.confirmed,
            summary=f"{cam} was knocked; its floor positions are off",
        )
        cam_pos = next((c.nominal_position for c in self.site.cameras if c.camera_id == cam and c.nominal_position), None)
        if cam_pos:
            ev = ev.model_copy(update={"location": (cam_pos[0], cam_pos[1])})
        self.events[ev.event_id] = ev
        eid = ev.event_id

        def repair():
            e = self.events[eid]
            try:
                res = dict(self.vision.recalibrate(cam))
            except Exception as exc:
                res = {"ok": False, "error": str(exc)}
            ok = bool(res.get("ok"))
            out = [self._step(e, "recalibrate", {"camera_id": cam}, res, "resolved" if ok else "ask human",
                              "Re-fitted the camera to the floor markers." if ok else "Not enough markers visible.")]
            out.append(self._update(e, status=EventStatus.resolved if ok else EventStatus.escalated,
                                    summary=f"{cam} re-aligned itself to the floor markers" if ok else e.summary))
            if not ok:
                out.append(self._ask(e, "recalibrate", f"{cam} could not re-align itself. Send someone to check it?",
                                     {"camera_id": cam}))
            return out

        self._later(t0 + 4, repair)
        return [_msg("event", ev)]

    def _restricted(self, te, t0):
        zone = zone_by_id(self.site, te.get("zone_id", "cold_room"))
        xy = polygon_centroid(zone.polygon) if zone else (1.7, 8.7)
        ev = Event(
            event_id=self._id("E"), type=EventType.unseen_in_zone, rule_id="R-cold-room", t_start=t0, t_detect=self._t,
            location=xy, zone_id=zone.zone_id if zone else None, track_ids=[f"T-{te.get('track', '?')}"],
            severity="warning", status=EventStatus.candidate,
            summary="Someone walked into the cold room, where no camera can see",
        )
        self.events[ev.event_id] = ev
        eid = ev.event_id

        def confirm():
            e = self.events[eid]
            out = [self._step(e, "cameras_covering", {"xy": list(xy)}, {"cameras": []}, "confirm",
                              "The person was last seen at the cold room door and has not come out.")]
            out.append(self._update(e, status=EventStatus.confirmed,
                                    evidence=[Evidence(kind="history", t=self._t, supports=True,
                                                       note="last seen entering the cold room")]))
            out.append(self._ask(self.events[eid], "dispatch_check",
                                 "A person has been in the cold room for a while and no camera can see them. "
                                 "Ask the floor lead to check on them?", {"zone_id": e.zone_id}))
            return out

        def resolve():
            e = self.events[eid]
            return [self._update(e, status=EventStatus.resolved, summary="The person left the cold room")]

        self._later(t0 + 15, confirm)
        if te.get("t_end") is not None:
            self._later(float(te["t_end"]) + 1, resolve)
        return [_msg("event", ev)]

    # ---- AgentRuntime ---------------------------------------------------- #

    def on_world(self, world: WorldState) -> list[LiveMessage]:
        self._t = world.t
        out: list[LiveMessage] = []
        for i, te in enumerate(self.truth.get("events", [])):
            if i not in self._fired and float(te.get("t_start", 1e18)) <= world.t:
                self._fired.add(i)
                out += self._start(te)
        while self._queue and self._queue[0][0] <= world.t:
            _, _, fn = self._queue.pop(0)
            out += fn()
        return out

    def add_rule_from_text(self, text: str, t: float) -> list[LiveMessage]:
        self._t = t
        low = text.lower()
        zone = next((z for z in self.site.zones if z.name.lower().split(" (")[0] in low or z.zone_id.replace("_", " ") in low), None)
        num = re.search(r"(\d+(?:\.\d+)?)\s*(m\b|metre|meter|s\b|sec|second|min)", low)
        value, unit = (float(num.group(1)), num.group(2)) if num else (None, "")
        if any(w in low for w in ("close", "within", "near")) and "forklift" in low:
            rtype, desc = RuleType.proximity, "warn when a person and a forklift get too close"
        elif any(w in low for w in ("clear", "block", "parked")):
            rtype, desc = RuleType.blocked, "alert when something is left in the zone"
        else:
            rtype, desc = RuleType.dwell, "alert when someone stays in the zone too long"
        seconds = distance = None
        if value is not None:
            if unit.startswith("min"):
                seconds = value * 60
            elif unit.startswith("s"):
                seconds = value
            else:
                distance = value
        rule = Rule(
            rule_id=self._id("R"), text=text, type=rtype, zone_id=zone.zone_id if zone else None,
            classes=[TrackClass.person, TrackClass.forklift] if rtype == RuleType.proximity else [TrackClass.person],
            seconds=seconds, distance_m=distance or (1.0 if rtype == RuleType.proximity else None),
            enabled=False, source="language", confirmed=False,
        )
        self.rules[rule.rule_id] = rule
        where = f" in {zone.name}" if zone else ""
        limit = f" after {seconds:g} s" if seconds else (f" closer than {rule.distance_m:g} m" if rule.distance_m else "")
        summary = f"I read this as: {desc}{where}{limit}. Turn this rule on?"
        return [
            _msg("rule", rule),
            self._ask(None, "confirm_rule", summary, {"rule": rule.model_dump(mode="json"), "zone_id": rule.zone_id}),
        ]

    def decide(self, approval_id: str, approved: bool, t: float) -> list[LiveMessage]:
        self._t = t
        ap = self.approvals[approval_id]  # KeyError -> 404 in the API
        if ap.status != ApprovalStatus.pending:
            return [_msg("approval", ap)]
        ap = ap.model_copy(update={"status": ApprovalStatus.approved if approved else ApprovalStatus.rejected, "decided_t": t})
        self.approvals[approval_id] = ap
        out = [_msg("approval", ap)]
        ev = self.events.get(ap.event_id) if ap.event_id else None
        if ap.action == "confirm_rule":
            rid = (ap.payload.get("rule") or {}).get("rule_id")
            if rid in self.rules:
                r = self.rules[rid].model_copy(update={"confirmed": approved, "enabled": approved})
                self.rules[rid] = r
                out.append(_msg("rule", r))
        elif ap.action == "escalate" and ev is not None and approved:
            out.append(self._update(ev, status=EventStatus.escalated))
            out.append(self._alert(self.events[ev.event_id], "manager", f"Escalated: {ev.summary}"))
        elif ap.action == "dispatch_check" and ev is not None and approved:
            out.append(self._alert(ev, "floor_lead", "Please check on the person in the cold room."))
        if ev is not None:
            out.append(self._step(self.events[ev.event_id], "human_decision", {"approval_id": approval_id},
                                  {"approved": approved}, "approved" if approved else "rejected",
                                  "A person decided; the agent never acts beyond an alert on its own."))
        return out

    def snapshot(self) -> dict:
        return {
            "rules": [r.model_dump(mode="json") for r in self.rules.values()],
            "events": [e.model_dump(mode="json") for e in self.events.values()],
            "approvals": [a.model_dump(mode="json") for a in self.approvals.values()],
            "alerts": [a.model_dump(mode="json") for a in self.alerts],
            "trace": [s.model_dump(mode="json") for s in self.trace],
        }

    def shift_report(self) -> dict:
        evs = list(self.events.values())
        confirmed = [e for e in evs if e.status in (EventStatus.confirmed, EventStatus.escalated, EventStatus.resolved)]
        rejected = [e for e in evs if e.status == EventStatus.rejected]
        return {
            "headline": f"{len(confirmed)} confirmed warning{'s' * (len(confirmed) != 1)}, {len(rejected)} false "
                        f"alarm{'s' * (len(rejected) != 1)} caught before anyone was bothered.",
            "highlights": [e.summary for e in confirmed][:8],
            "false_alarms": [e.summary for e in rejected],
            "decisions_by_people": sum(1 for a in self.approvals.values() if a.status != ApprovalStatus.pending),
            "note": "Demo script, not the real agent.",
        }
