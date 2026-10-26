"""Rules engine: default rules, per-tick evaluation and incident dedup.

Each WorldState tick produces raw rule hits; `RulesEngine.update` folds them
into incidents (one per object / pair / camera episode) so the agent opens one
event per real-world incident instead of one per frame.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from floorsight.contracts import (
    XY,
    CameraStatus,
    EventType,
    Rule,
    RuleType,
    SitePlan,
    Track,
    TrackClass,
    TrackState,
    WorldState,
    ZoneKind,
)
from floorsight.site import FORKLIFT_LENGTH_M, FORKLIFT_WIDTH_M, box_distance, point_in_polygon
from floorsight.formatting import floor_error_text

STATIC_SPEED_MPS = 0.15  # slower than this counts as "not moving"
HEADING_MIN_SPEED_MPS = 0.2  # below this keep the forklift's last heading
NEAR_MISS_MERGE_S = 3.0  # hits for the same pair within this gap are one incident
# People step aside in an emergency; anything parked does not.
BLOCKING_CLASSES = (TrackClass.object, TrackClass.unknown, TrackClass.forklift)
EPISODE_GAP_S = 2.0  # other incidents end after this long without a hit
UNSEEN_MERGE_S = 30.0  # tracking flicker must not reopen a "someone is in the blind zone" incident
SPOT_RADIUS_M = 1.0  # remembered false-alarm spots apply within this radius
SAME_OBJECT_RADIUS_M = 0.6  # a new static track this close to an open one is the same object
SPOT_PENALTY_S = 10.0  # added to a blocked threshold per rejected false alarm


def _what(cls: TrackClass) -> str:
    return {"object": "a static object", "person": "a person standing", "forklift": "a parked forklift"}.get(cls.value, "something")


@dataclass
class Candidate:
    """One incident (new or continuing) produced by the rules this tick."""

    key: tuple
    type: EventType
    rule: Rule | None
    t: float
    t_start: float
    location: XY
    zone_id: str | None
    track_ids: list[str]
    camera_ids: list[str]
    severity: str
    summary: str
    data: dict = field(default_factory=dict)


@dataclass
class SpotMemory:
    """A place where a camera produced a false alarm; raises that spot's threshold."""

    zone_id: str
    xy: XY
    camera_ids: list[str]
    extra_s: float
    t: float
    event_id: str
    reason: str


def default_rules(site: SitePlan) -> list[Rule]:
    """Blocked exits, person/forklift near-miss, unseen people in restricted zones.

    On a site with the real-camera floor stage (``site.floor_stage``) the near-miss rule uses the stage's measured
    forklift footprints, danger zones and 3 s warnings (``WorldState.hazards``) instead of a fixed forklift box.
    """
    rules = []
    for z in site.zones:
        if z.kind == ZoneKind.exit:
            rules.append(
                Rule(
                    rule_id=f"R-{z.zone_id}-blocked",
                    text=f"Alert when an object or a parked forklift stays in {z.name} for more than 20 s",
                    type=RuleType.blocked,
                    zone_id=z.zone_id,
                    classes=list(BLOCKING_CLASSES),
                    seconds=20.0,
                )
            )
    if getattr(site, "floor_stage", False):
        rules.append(
            Rule(
                rule_id="R-floor-warning",
                text="Warn when a moving forklift will come within 0.8 m of a person in the next 3 s if it keeps its course, or a person is in its stopping zone (3 frames in a row)",
                type=RuleType.floor_zone,
                classes=[TrackClass.person, TrackClass.forklift],
                distance_m=0.8,
                seconds=3.0,
            )
        )
    else:
        rules.append(
            Rule(
                rule_id="R-near-miss",
                text="Alert when a person comes within 0.8 m of a forklift while they are closing faster than 0.2 m/s",
                type=RuleType.proximity,
                classes=[TrackClass.person, TrackClass.forklift],
                distance_m=0.8,
                closing_mps=0.2,
            )
        )
    for z in site.zones:
        if z.kind == ZoneKind.restricted:
            rules.append(
                Rule(
                    rule_id=f"R-{z.zone_id}-unseen",
                    text=f"Alert when someone is inside {z.name} and no camera can see them for more than 10 s",
                    type=RuleType.unseen,
                    zone_id=z.zone_id,
                    classes=[TrackClass.person, TrackClass.unknown],
                    seconds=10.0,
                )
            )
    return rules


def speed(tr: Track) -> float:
    return math.hypot(tr.vx, tr.vy)


def closing_speed(person: Track, other: Track, heading: float, dt: float = 0.1) -> float:
    """How fast the person/forklift-body distance is shrinking (m/s, positive = closing)."""
    d0 = box_distance(person.x, person.y, other.x, other.y, heading, FORKLIFT_LENGTH_M, FORKLIFT_WIDTH_M)
    d1 = box_distance(
        person.x + person.vx * dt,
        person.y + person.vy * dt,
        other.x + other.vx * dt,
        other.y + other.vy * dt,
        heading,
        FORKLIFT_LENGTH_M,
        FORKLIFT_WIDTH_M,
    )
    return (d0 - d1) / dt


class RulesEngine:
    def __init__(self, site: SitePlan, rules: list[Rule] | None = None):
        self.site = site
        self.zones = {z.zone_id: z for z in site.zones}
        self.rules: list[Rule] = list(rules if rules is not None else default_rules(site))
        self.spots: list[SpotMemory] = []
        self._since: dict[tuple, float] = {}
        self._heading: dict[str, float] = {}
        self._heading_observations: dict[str, deque] = {}
        self._episodes: dict[tuple, dict] = {}  # incident key -> {"start", "last", "location", "zone_id"}
        self._alias: dict[tuple, tuple] = {}

    # ---- rules ----------------------------------------------------------- #

    def add_rule(self, rule: Rule) -> None:
        self.rules = [r for r in self.rules if r.rule_id != rule.rule_id] + [rule]

    def get_rule(self, rule_id: str | None) -> Rule | None:
        return next((r for r in self.rules if r.rule_id == rule_id), None)

    # ---- false-alarm memory ---------------------------------------------- #

    def remember_false_alarm(self, zone_id: str, xy: XY, camera_ids: list[str], t: float, event_id: str, reason: str) -> SpotMemory:
        spot = SpotMemory(zone_id, (round(xy[0], 2), round(xy[1], 2)), sorted(camera_ids), SPOT_PENALTY_S, t, event_id, reason)
        self.spots.append(spot)
        return spot

    def spot_penalty(self, zone_id: str, xy: XY, camera_ids: list[str]) -> float:
        """Extra seconds for a detection at a remembered spot seen only by the cameras that fooled us."""
        cams = set(camera_ids)
        return sum(
            s.extra_s
            for s in self.spots
            if s.zone_id == zone_id and math.dist(s.xy, xy) <= SPOT_RADIUS_M and cams and cams <= set(s.camera_ids)
        )

    # ---- per-tick evaluation --------------------------------------------- #

    def heading(self, tr: Track) -> float:
        return self._heading.get(tr.track_id, 0.0)

    def evaluate(self, world: WorldState) -> list[Candidate]:
        """Raw rule hits for this tick, one per incident key (first rule wins)."""
        t = world.t
        for tid in list(self._heading_observations):
            samples = self._heading_observations[tid]
            while samples and t - samples[0][0] > 1.0:
                samples.popleft()
            if not samples:
                del self._heading_observations[tid]
        for tr in world.tracks:
            if tr.cls == TrackClass.forklift and tr.state == TrackState.seen and tr.sigma_m <= .35:
                samples = self._heading_observations.setdefault(tr.track_id, deque(maxlen=32))
                if not samples or samples[-1][0] < t:
                    samples.append((t, tr.x, tr.y))
            if tr.cls == TrackClass.forklift and speed(tr) >= HEADING_MIN_SPEED_MPS:
                self._heading[tr.track_id] = math.atan2(tr.vy, tr.vx)
        hits: dict[tuple, Candidate] = {}
        active: set[tuple] = set()
        for rule in self.rules:
            if not rule.enabled:
                continue
            if rule.type == RuleType.blocked:
                self._eval_blocked(rule, world, hits, active)
            elif rule.type == RuleType.proximity:
                self._eval_proximity(rule, world, hits)
            elif rule.type == RuleType.dwell:
                self._eval_dwell(rule, world, hits, active)
            elif rule.type == RuleType.unseen:
                self._eval_unseen(rule, world, hits, active)
            elif rule.type == RuleType.floor_zone:
                self._eval_floor_zone(rule, world, hits)
        for h in world.cameras:
            if h.status == CameraStatus.moved:
                cam = next((c for c in self.site.cameras if c.camera_id == h.camera_id), None)
                loc = tuple(cam.nominal_position[:2]) if cam and cam.nominal_position else (0.0, 0.0)
                key = ("camera_moved", h.camera_id)
                hits.setdefault(
                    key,
                    Candidate(
                        key=key,
                        type=EventType.camera_moved,
                        rule=None,
                        t=t,
                        t_start=t,
                        location=loc,
                        zone_id=None,
                        track_ids=[],
                        camera_ids=[h.camera_id],
                        severity="warning",
                        summary=f"{h.camera_id} reports it has moved (floor alignment error {floor_error_text(h.calib_error_m)}, {h.markers_visible} markers visible)",
                        data={"calib_error_m": h.calib_error_m, "markers_visible": h.markers_visible},
                    ),
                )
        for k in list(self._since):
            if k not in active:
                del self._since[k]
        return list(hits.values())

    def _timer(self, key: tuple, t: float, active: set) -> float:
        active.add(key)
        return t - self._since.setdefault(key, t)

    def _eval_blocked(self, rule: Rule, world: WorldState, hits: dict, active: set) -> None:
        zone = self.zones.get(rule.zone_id or "")
        if zone is None:
            return
        classes = set(rule.classes or [TrackClass.object, TrackClass.unknown])
        for tr in world.tracks:
            if tr.state != TrackState.seen or tr.cls not in classes or speed(tr) >= STATIC_SPEED_MPS:
                continue
            if not point_in_polygon(tr.x, tr.y, zone.polygon):
                continue
            dur = self._timer(("static", rule.rule_id, tr.track_id), world.t, active)
            threshold = (rule.seconds or 0.0) + self.spot_penalty(zone.zone_id, (tr.x, tr.y), tr.camera_ids)
            if dur < threshold:
                continue
            key = ("blocked", zone.zone_id, tr.track_id)
            hits.setdefault(
                key,
                Candidate(
                    key=key,
                    type=EventType.blocked_exit,
                    rule=rule,
                    t=world.t,
                    t_start=round(world.t - dur, 3),
                    location=(round(tr.x, 2), round(tr.y, 2)),
                    zone_id=zone.zone_id,
                    track_ids=[tr.track_id],
                    camera_ids=list(tr.camera_ids),
                    severity="critical",
                    summary=f"{zone.name} blocked: {_what(tr.cls)} at ({tr.x:.1f}, {tr.y:.1f}) for {dur:.0f} s, seen by {', '.join(tr.camera_ids) or 'no camera'}",
                    data={"static_s": round(dur, 1), "threshold_s": threshold},
                ),
            )

    def _eval_proximity(self, rule: Rule, world: WorldState, hits: dict) -> None:
        classes = rule.classes or [TrackClass.person, TrackClass.forklift]
        a_cls, b_cls = classes[0], classes[1] if len(classes) > 1 else TrackClass.forklift
        zone = self.zones.get(rule.zone_id or "")
        limit = rule.distance_m if rule.distance_m is not None else 0.8
        seen = [tr for tr in world.tracks if tr.state == TrackState.seen]
        for a in seen:
            if a.cls != a_cls:
                continue
            if zone is not None and not point_in_polygon(a.x, a.y, zone.polygon):
                continue
            for b in seen:
                if b.cls != b_cls or b.track_id == a.track_id:
                    continue
                if b.cls == TrackClass.forklift:
                    h = self.heading(b)
                    d = box_distance(a.x, a.y, b.x, b.y, h, FORKLIFT_LENGTH_M, FORKLIFT_WIDTH_M)
                    closing = closing_speed(a, b, h)
                else:
                    d = math.dist((a.x, a.y), (b.x, b.y))
                    d1 = math.dist((a.x + a.vx * 0.1, a.y + a.vy * 0.1), (b.x + b.vx * 0.1, b.y + b.vy * 0.1))
                    closing = (d - d1) / 0.1
                if d >= limit:
                    continue
                if rule.closing_mps is not None and closing <= rule.closing_mps:
                    continue
                # Presence in another camera cannot establish oriented clearance.
                # Require a recent observed displacement baseline for the vehicle.
                geometry_supported = b.cls != TrackClass.forklift
                historical_gap = None
                baseline_m = None
                if b.cls == TrackClass.forklift:
                    prior = [s for s in self._heading_observations.get(b.track_id, ())
                             if .5 <= world.t - s[0] <= 1.0]
                    if prior:
                        _, x0, y0 = prior[0]
                        dx, dy = b.x-x0, b.y-y0
                        baseline_m = math.hypot(dx, dy)
                        if baseline_m >= .2:
                            historical_gap = box_distance(a.x, a.y, b.x, b.y, math.atan2(dy,dx), FORKLIFT_LENGTH_M, FORKLIFT_WIDTH_M)
                            geometry_supported = historical_gap < limit
                key = ("near_miss", a.track_id, b.track_id)
                where = f" in {zone.name}" if zone else ""
                hits.setdefault(
                    key,
                    Candidate(
                        key=key,
                        type=EventType.near_miss,
                        rule=rule,
                        t=world.t,
                        t_start=world.t,
                        location=(round(a.x, 2), round(a.y, 2)),
                        zone_id=zone.zone_id if zone else None,
                        track_ids=[a.track_id, b.track_id],
                        camera_ids=sorted(set(a.camera_ids) | set(b.camera_ids)),
                        severity="warning" if zone else "critical",
                        summary=f"Near-miss{where}: {a.track_id} {d:.2f} m from {b.track_id}, closing at {closing:.1f} m/s",
                        data={"distance_m": round(d, 3), "closing_mps": round(closing, 3),
                              "geometry_supported": geometry_supported,
                              "historical_gap_m": historical_gap, "heading_baseline_m": baseline_m},
                    ),
                )

    def _eval_dwell(self, rule: Rule, world: WorldState, hits: dict, active: set) -> None:
        zone = self.zones.get(rule.zone_id or "")
        if zone is None:
            return
        classes = set(rule.classes or [TrackClass.person])
        for tr in world.tracks:
            if tr.state != TrackState.seen or tr.cls not in classes or not point_in_polygon(tr.x, tr.y, zone.polygon):
                continue
            dur = self._timer(("dwell", rule.rule_id, tr.track_id), world.t, active)
            if dur < (rule.seconds or 0.0):
                continue
            key = ("dwell", zone.zone_id, tr.track_id)
            hits.setdefault(
                key,
                Candidate(
                    key=key,
                    type=EventType.dwell,
                    rule=rule,
                    t=world.t,
                    t_start=round(world.t - dur, 3),
                    location=(round(tr.x, 2), round(tr.y, 2)),
                    zone_id=zone.zone_id,
                    track_ids=[tr.track_id],
                    camera_ids=list(tr.camera_ids),
                    severity="warning",
                    summary=f"{tr.track_id} has been in {zone.name} for {dur:.0f} s",
                    data={"dwell_s": round(dur, 1)},
                ),
            )

    def _eval_unseen(self, rule: Rule, world: WorldState, hits: dict, active: set) -> None:
        zone = self.zones.get(rule.zone_id or "")
        if zone is None:
            return
        classes = set(rule.classes or [TrackClass.person, TrackClass.unknown])
        for tr in world.tracks:
            if tr.state != TrackState.unseen or tr.cls not in classes or not point_in_polygon(tr.x, tr.y, zone.polygon):
                continue
            dur = self._timer(("unseen", rule.rule_id, tr.track_id), world.t, active)
            if dur < (rule.seconds or 0.0):
                continue
            key = ("unseen", zone.zone_id, tr.track_id)
            hits.setdefault(
                key,
                Candidate(
                    key=key,
                    type=EventType.unseen_in_zone,
                    rule=rule,
                    t=world.t,
                    t_start=round(world.t - dur, 3),
                    location=(round(tr.x, 2), round(tr.y, 2)),
                    zone_id=zone.zone_id,
                    track_ids=[tr.track_id],
                    camera_ids=[],
                    severity="warning",
                    summary=f"{tr.track_id} is predicted inside {zone.name} but no camera has seen them for {dur:.0f} s",
                    data={"unseen_s": round(dur, 1)},
                ),
            )

    def _eval_floor_zone(self, rule: Rule, world: WorldState, hits: dict) -> None:
        """Confirmed floor warnings/dangers from the real-camera floor stage become near-miss-like incidents."""
        hz = world.hazards
        if hz is None:
            return
        tracks = {tr.track_id: tr for tr in world.tracks}
        for a in hz.alerts:
            if not a.confirmed or not a.vehicle_moving:
                continue
            person = tracks.get(a.person_id)
            loc = (round(person.x, 2), round(person.y, 2)) if person is not None else (0.0, 0.0)
            key = ("floor", a.person_id, a.vehicle_id or "-")
            clear = (a.clearance_m or {}).get("median")
            if a.level == "danger":
                what = " + ".join(a.reasons) or "in the forklift's danger zone"
                summary = f"{a.person_id}: {what} of {a.vehicle_id}" + (f" (floor clearance {clear:.1f} m)" if clear is not None else "")
            else:
                ttc = (a.ttc_s or {}).get("median")
                when = "" if ttc is None else " now" if ttc < 0.05 else f" in {ttc:.1f} s"
                summary = f"{a.vehicle_id} will pass within 0.8 m of {a.person_id}{when} if it keeps its course" + (f" (gap {clear:.1f} m)" if clear is not None else "")
            hits.setdefault(
                key,
                Candidate(
                    key=key,
                    type=EventType.floor_warning,
                    rule=rule,
                    t=world.t,
                    t_start=world.t,
                    location=loc,
                    zone_id=None,
                    track_ids=[a.person_id] + ([a.vehicle_id] if a.vehicle_id else []),
                    camera_ids=list(person.camera_ids) if person is not None else [],
                    severity="critical" if a.level == "danger" else "warning",
                    summary=summary,
                    data={"level": a.level, "clearance_m": clear, "alert": a.model_dump(mode="json"), "cut_by_frame": a.cut_by_frame},
                ),
            )

    # ---- dedup ------------------------------------------------------------ #

    def update(self, world: WorldState) -> tuple[list[Candidate], list[Candidate]]:
        """Evaluate and dedup. Returns (new incidents, continuing incidents) for this tick."""
        t = world.t
        new, cont = [], []
        for c in self.evaluate(world):
            key = self._alias.get(c.key, c.key)
            if key not in self._episodes and c.type == EventType.blocked_exit:
                twin = self._same_object(c, t)
                if twin is not None:
                    self._alias[c.key] = key = twin
            if key not in self._episodes and c.type == EventType.unseen_in_zone:
                twin = self._same_unseen_zone(c, t)
                if twin is not None:
                    self._alias[c.key] = key = twin
            if c.type in (EventType.near_miss, EventType.floor_warning):
                gap = NEAR_MISS_MERGE_S
            elif c.type == EventType.unseen_in_zone:
                gap = UNSEEN_MERGE_S
            else:
                gap = EPISODE_GAP_S
            ep = self._episodes.get(key)
            c.key = key
            if ep is not None and t - ep["last"] <= gap:
                ep["last"] = t
                cont.append(c)
            else:
                self._episodes[key] = {"start": t, "last": t, "location": c.location, "zone_id": c.zone_id}
                new.append(c)
        return new, cont

    def _same_object(self, c: Candidate, t: float) -> tuple | None:
        for key, ep in self._episodes.items():
            if key[0] == "blocked" and ep["zone_id"] == c.zone_id and t - ep["last"] <= EPISODE_GAP_S:
                if math.dist(ep["location"], c.location) <= SAME_OBJECT_RADIUS_M:
                    return key
        return None

    def _same_unseen_zone(self, c: Candidate, t: float) -> tuple | None:
        """One open "unseen in zone" incident per zone, whatever track id the tracker gives the person."""
        for key, ep in self._episodes.items():
            if key[0] == "unseen" and ep["zone_id"] == c.zone_id and t - ep["last"] <= UNSEEN_MERGE_S:
                return key
        return None

    def last_hit(self, key: tuple) -> float | None:
        ep = self._episodes.get(key)
        return ep["last"] if ep else None
