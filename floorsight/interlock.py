"""Forklift interlock: slow or stop a forklift by itself the moment a warning is confirmed (24 Oct).

Why: a warning comes 1-3 s before the close pass, but the agent's second look takes seconds and a person's approval
5-30 s. So the machine acts first and people review afterwards:

    warning: it will come within 0.8 m of a person in the next 3 s if it keeps its course   -> SLOW to walking pace
    danger:  the person is (or within STOP_TTC_S will be) in its path, closer than PATH_M   -> STOP
             (a person standing beside the path, e.g. at a rack face in a narrow aisle, is passed slowly, not blocked)
    the person walks into a forklift whose own motion is not closing the gap (stopping would not help)
                                                                                              -> HORN (warn the person)
    no warning or danger for RESUME_CLEAR_S                                                   -> RESUME (or back to SLOW)
    ...but a truck stopped for a person resumes only if the LAST time it saw that person, they were out of its path
    and not coming back into it: "no longer visible" is not "gone" (interlock-sim-04: a picker standing still in the
    lane faded from the background model and the truck drove on into the picker it could no longer see), while a person
    last seen standing clear of the path need not be seen again (interlock-sim-05: the truck waited 35 s for someone who
    had crossed and stood still out of view). "Clear" is judged from where they were, not where a drifting track says
    they are heading, and a truck we stopped is taken to stand where it stopped (interlock-sim-06: the standing picker's
    and the stopped truck's tracks both drifted by 1-2 m and read as walking apart).

Slow comes before stop because a hard stop with a raised load can tip a truck or drop the load. Commands go to the
vehicle's receiver: on a real truck a unit wired to its speed-limit input (hardware this project does not include);
in the simulator the simulated truck really brakes, so the loop can be measured. Every command is logged with the
reason; the agent and a person still review every warning afterwards (was it real? -> the official alert and record),
and needless stops are counted, because they cost work.

Two inputs, one decision:
  - sites with the real-camera floor stage (``WorldState.hazards``): its confirmed alerts (3 frames in a row), which
    already model each forklift's course, stopping zone and uncertainty;
  - other sites (the simulated warehouse): a 3 s constant-velocity prediction on the fused floor tracks, held for
    CONFIRM_S.
For recorded video (the real clips) the truck cannot be stopped; the command is logged with what it would have done:
the truck's stopping distance at its measured speed against the floor gap to the person.
"""

from __future__ import annotations

import math
from collections import deque
from types import SimpleNamespace

from floorsight.contracts import TrackClass, TrackState, WorldState
from floorsight.site import FORKLIFT_LENGTH_M, FORKLIFT_WIDTH_M, box_distance

CLEAR_M = 0.8            # the near-miss distance used everywhere in Floorsight
PERSON_STILL_MPS = 0.2   # slower than this a person is standing
PATH_M = 0.25            # closer than this the person is in the truck's path: STOP (between this and CLEAR_M: pass slowly)
WARN_TTC_S = 3.0         # look-ahead for SLOW
STOP_TTC_S = 1.5         # contact predicted sooner than this -> STOP
STEP_S = 0.1             # prediction step
MOVING_MPS = 0.3         # slower than this a forklift is treated as standing
CONFIRM_S = 0.2          # a track-based condition must hold this long (and 2 ticks) before the machine acts
RESUME_CLEAR_S = 2.0     # no condition for this long -> the next lower state
HORN_REPEAT_S = 3.0      # one horn per person/forklift pair in this time
SLOW_MPS = 0.7           # "walking pace" speed limit sent with SLOW
REIDENT_M = 2.0          # a forklift track that appears this close to where a forklift was just lost is the same truck
LOST_HOLD_S = 5.0        # a slowed/stopped truck the cameras lose keeps its state this long (absent is not clear)
BLOCKER_NEAR_M = 1.5     # the person a truck stopped for counts as seen again if a person track is this close to them
LATENCY_S = 0.3          # camera + processing + radio + brake actuation, from the frame to the brakes acting
BRAKE_MPS2 = (1.5, 3.0)  # forklift braking range (the floor stage's prior); the worst case is used for "would stop short"

LEVEL = {"normal": 0, "slow": 1, "stop": 2}
NAME = {v: k for k, v in LEVEL.items()}


def stopping_distance(speed: float, decel: float = BRAKE_MPS2[0], latency: float = LATENCY_S) -> float:
    """Metres a truck at `speed` covers from the frame that triggers a stop until it stands still."""
    return speed * latency + speed * speed / (2.0 * decel)


def heading_of(vx: float, vy: float) -> float:
    return math.atan2(vy, vx)


def predict(person, fork, heading: float, horizon: float = WARN_TTC_S, fork_moves: bool = True):
    """Constant-velocity look-ahead of the person against the forklift body (FORKLIFT_LENGTH_M x FORKLIFT_WIDTH_M).
    Returns (ttc_s: first time within CLEAR_M or None, ttc_path_s: first time within PATH_M or None, gap_now_m).
    With fork_moves=False the forklift stands where it is."""
    fvx, fvy = (fork.vx, fork.vy) if fork_moves else (0.0, 0.0)
    gap_now = box_distance(person.x, person.y, fork.x, fork.y, heading, FORKLIFT_LENGTH_M, FORKLIFT_WIDTH_M)
    ttc = ttc_path = None
    for k in range(int(round(horizon / STEP_S)) + 1):
        dt = k * STEP_S
        d = box_distance(person.x + person.vx * dt, person.y + person.vy * dt, fork.x + fvx * dt, fork.y + fvy * dt,
                         heading, FORKLIFT_LENGTH_M, FORKLIFT_WIDTH_M)
        if ttc is None and d < CLEAR_M:
            ttc = dt
        if ttc_path is None and d < PATH_M:
            ttc_path = dt
            break
    return ttc, ttc_path, gap_now


class ForkliftInterlock:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.vehicles: dict[str, dict] = {}     # vehicle id -> {state, since, clear_since, reason, person, ...}
        self.streak: dict[tuple, list] = {}     # (vehicle, person) -> [first_t, ticks, level]
        self.last_horn: dict[tuple, float] = {}
        self.log: list[dict] = []
        self.recent: deque = deque(maxlen=8)
        self.counts = {"slow": 0, "stop": 0, "resume": 0, "horn": 0}
        # The vision can lose a standing truck (it fades into the background model) and find it again under a new track
        # id; states belong to the physical truck, so tracks are mapped to vehicle keys V1, V2, ... by position.
        self.track_key: dict[str, str] = {}
        self.vmeta: dict[str, dict] = {}    # vehicle key -> {"xy", "t" last seen, "track"}
        self._nkeys = 0
        self._present: set[str] = set()
        self._floor = False   # this tick's input is the real-camera floor stage (its alerts stop once the truck stands)

    def _vehicle_keys(self, forks, t: float) -> dict[str, str]:
        out: dict[str, str] = {}
        for f in forks:  # tracks we already know
            k = self.track_key.get(f.track_id)
            if k is not None and k not in out.values():
                out[f.track_id] = k
        for f in forks:  # new tracks: the nearest recently lost truck, else a new key
            if f.track_id in out:
                continue
            near = [(math.hypot(m["xy"][0] - f.x, m["xy"][1] - f.y), k) for k, m in self.vmeta.items()
                    if k not in out.values() and t - m["t"] <= LOST_HOLD_S]
            near = [n for n in near if n[0] < REIDENT_M]
            if near:
                k = min(near)[1]
            else:
                self._nkeys += 1
                k = f"V{self._nkeys}"
            self.track_key[f.track_id] = k
            out[f.track_id] = k
        for f in forks:
            self.vmeta[out[f.track_id]] = {"xy": (f.x, f.y), "t": t, "track": f.track_id}
        return out

    # ------------------------------------------------------------------ conditions
    def _from_hazards(self, world: WorldState) -> list[dict]:
        tracks = {tr.track_id: tr for tr in world.tracks}
        out = []
        for a in world.hazards.alerts:
            if not (a.confirmed and a.vehicle_moving and a.vehicle_id):
                continue
            v = tracks.get(a.vehicle_id)
            p = tracks.get(a.person_id)  # the person it is for: a stop waits until they are seen clear (audit C1)
            spd = math.hypot(v.vx, v.vy) if v is not None else None
            gap = (a.clearance_m or {}).get("median")
            ttc = (a.ttc_s or {}).get("median")
            level = "stop" if a.level == "danger" or (ttc is not None and ttc < STOP_TTC_S) else "slow"
            why = (" + ".join(a.reasons) if a.level == "danger" and a.reasons else
                   f"will pass within {CLEAR_M} m of {a.person_id}" + (f" in {ttc:.1f} s" if ttc is not None else ""))
            out.append({"vehicle": a.vehicle_id, "track": a.vehicle_id, "person": a.person_id, "level": level, "action": level,
                        "ttc_s": ttc, "gap_m": gap, "speed_mps": spd, "reason": why, "confirmed": True,
                        "heading": heading_of(v.vx, v.vy) if v is not None else None,
                        "xy": [round(v.x, 2), round(v.y, 2)] if v is not None else None,
                        "person_xy": [round(p.x, 2), round(p.y, 2)] if p is not None else None})
        return out

    def _from_tracks(self, world: WorldState) -> list[dict]:
        live = (TrackState.seen, TrackState.estimated)
        forks = [tr for tr in world.tracks if tr.cls == TrackClass.forklift and tr.state in live]
        people = [tr for tr in world.tracks if tr.cls == TrackClass.person and tr.state in live]
        keys = self._vehicle_keys(forks, world.t)
        self._present = set(keys.values())
        out = []
        for f in forks:
            key = keys[f.track_id]
            spd = math.hypot(f.vx, f.vy)
            st = self.vehicles.get(key)
            intent = st.get("intent") if st and st["state"] != "normal" else None
            if st and st["state"] == "stop" and st.get("stop_xy") is not None:
                # we stopped it: it stands where it stopped, whatever a fading track says
                f = SimpleNamespace(track_id=f.track_id, x=st["stop_xy"][0], y=st["stop_xy"][1], vx=f.vx, vy=f.vy)
            if intent is not None and spd < intent[1]:
                # slowed or stopped by us: judge the path as if it drove on at its own speed, so it resumes only when
                # the way is clear (otherwise a stopped truck "sees" nothing, resumes, stops again and creeps closer)
                f = SimpleNamespace(track_id=f.track_id, x=f.x, y=f.y, vx=intent[1] * math.cos(intent[0]), vy=intent[1] * math.sin(intent[0]))
                spd = intent[1]
            if spd < MOVING_MPS:
                continue
            h = heading_of(f.vx, f.vy)
            for p in people:
                ttc, ttc_path, gap_now = predict(p, f, h)
                if ttc is None:
                    continue
                # Is the forklift's own motion closing the gap? If the person would reach the truck even with the truck
                # standing still (walking into its side or back while it drives away), stopping it does not help:
                # warn the person instead.
                if math.hypot(p.vx, p.vy) < PERSON_STILL_MPS:
                    truck_caused = True  # a person standing still is not walking into anything
                elif gap_now >= CLEAR_M:
                    ttc_still, _, _ = predict(p, f, h, fork_moves=False)
                    truck_caused = ttc_still is None or ttc + STEP_S < ttc_still
                else:  # already within 0.8 m: whose motion closes the gap faster?
                    g_truck = box_distance(p.x, p.y, f.x + f.vx * STEP_S, f.y + f.vy * STEP_S, h, FORKLIFT_LENGTH_M, FORKLIFT_WIDTH_M)
                    g_person = box_distance(p.x + p.vx * STEP_S, p.y + p.vy * STEP_S, f.x, f.y, h, FORKLIFT_LENGTH_M, FORKLIFT_WIDTH_M)
                    truck_caused = gap_now - g_truck >= gap_now - g_person
                level = "stop" if ttc_path is not None and ttc_path < STOP_TTC_S else "slow"
                action = level if truck_caused else "horn"
                why = (f"{p.track_id} {gap_now:.1f} m from {f.track_id}, contact predicted in {ttc:.1f} s" if truck_caused
                       else f"{p.track_id} is walking into {f.track_id} ({gap_now:.1f} m)")
                out.append({"vehicle": key, "track": f.track_id, "person": p.track_id, "level": level, "action": action,
                            "ttc_s": round(ttc, 2), "gap_m": round(gap_now, 2), "speed_mps": round(spd, 2), "heading": h,
                            "reason": why, "confirmed": False,
                            "xy": [round(f.x, 2), round(f.y, 2)], "person_xy": [round(p.x, 2), round(p.y, 2)]})
        return out

    def _confirm(self, conds: list[dict], t: float) -> list[dict]:
        keep, seen = [], set()
        for c in conds:
            key = (c["vehicle"], c["person"])
            seen.add(key)
            s = self.streak.get(key)
            if s is None or t - s[1] > 0.5:  # broken streak (or first tick)
                s = self.streak[key] = [t, t, 0]
            s[1], s[2] = t, s[2] + 1
            if c["confirmed"] or (s[2] >= 2 and t - s[0] >= CONFIRM_S - 1e-6):
                keep.append(c)
        for key in [k for k in self.streak if k not in seen]:
            del self.streak[key]
        return keep

    # ------------------------------------------------------------------ public
    def update(self, world: WorldState) -> list[dict]:
        """One tick. Returns the commands issued now (slow / stop / resume / horn), each with its reason."""
        if not self.enabled:
            return []
        t = world.t
        self._floor = world.hazards is not None
        if world.hazards is not None:
            conds = self._from_hazards(world)
            self._present = {tr.track_id for tr in world.tracks if tr.cls == TrackClass.forklift}
        else:
            conds = self._from_tracks(world)
        conds = self._confirm(conds, t)
        cmds = []
        by_vehicle: dict[str, dict] = {}
        for c in conds:
            if c["action"] == "horn":
                key = (c["vehicle"], c["person"])
                if t - self.last_horn.get(key, -1e9) >= HORN_REPEAT_S:
                    self.last_horn[key] = t
                    cmds.append(self._cmd(t, "horn", c))
                continue
            if c["vehicle"] not in by_vehicle or LEVEL[c["level"]] > LEVEL[by_vehicle[c["vehicle"]]["level"]]:
                by_vehicle[c["vehicle"]] = c
        present = self._present
        for vid in set(by_vehicle) | set(self.vehicles):
            st = self.vehicles.setdefault(vid, {"state": "normal", "since": t, "clear_since": None})
            c = by_vehicle.get(vid)
            target = LEVEL[c["level"]] if c else 0
            cur = LEVEL[st["state"]]
            if c is not None and c.get("person_xy") is not None and target >= 1:
                pv = next(([tr.vx, tr.vy] for tr in world.tracks if tr.track_id == c["person"]), [0.0, 0.0])
                st["blocker"] = {"track": c["person"], "xy": c["person_xy"], "v": pv, "t": t}  # latest person it slows/stops for
            if target > cur:
                if cur == 0:  # remember how it was driving, to judge when the way is clear again
                    sp = c.get("speed_mps") or 0.0
                    st["intent"] = (c["heading"], sp) if c.get("heading") is not None and sp >= MOVING_MPS else None
                if NAME[target] == "stop" and c.get("xy") is not None and st.get("intent") is not None:
                    h, sp = st["intent"]
                    d = stopping_distance(min(sp, c.get("speed_mps") or sp), decel=sum(BRAKE_MPS2) / 2)
                    st["stop_xy"] = [c["xy"][0] + d * math.cos(h), c["xy"][1] + d * math.sin(h)]
                st.update(state=NAME[target], since=t, clear_since=None, reason=c["reason"], person=c["person"])
                cmds.append(self._cmd(t, NAME[target], c))
            elif target < cur:
                if st["state"] == "stop" and not self._blocker_seen(st.get("blocker"), world, vid, st):
                    st["clear_since"], st["waiting"] = None, "waiting to see the person clear of its path"
                    continue
                st["waiting"] = None
                if vid not in present:
                    lost_t = self.vmeta.get(vid, {}).get("t")
                    if st["state"] == "normal" or (lost_t is not None and t - lost_t <= LOST_HOLD_S):
                        continue  # out of view is not "clear": a lost slowed/stopped truck keeps its state a while
                if st["clear_since"] is None:
                    st["clear_since"] = t
                elif t - st["clear_since"] >= RESUME_CLEAR_S - 1e-6:
                    st.update(state=NAME[target], since=t, clear_since=None, stop_xy=None)
                    if target == 0:
                        st["intent"] = None
                    cmds.append(self._cmd(t, "resume" if target == 0 else NAME[target],
                                          c or {"vehicle": vid, "person": st.get("person"), "reason": f"clear for {RESUME_CLEAR_S:.0f} s"}))
            else:
                st["clear_since"] = None
        return cmds

    def _blocker_seen(self, blocker: dict | None, world: WorldState, vid: str, st: dict) -> bool:
        """May the truck stopped for this person resume? Yes if the person is in view now (the normal conditions then
        decide), or if the last time they were seen they were out of the truck's path and not heading back into it.
        With the floor stage as input the normal conditions cannot decide: its alerts need a moving truck, so they stop
        the moment the truck stands. There the person must be seen clear of the path the truck would drive on, and a
        stop for a person it never had on the floor map stays latched (audit C1, 27 Oct)."""
        if not blocker or blocker.get("xy") is None:
            return not self._floor
        live = (TrackState.seen, TrackState.estimated)
        for tr in world.tracks:
            if tr.cls != TrackClass.person or tr.state not in live:
                continue
            if tr.track_id == blocker["track"] or math.hypot(tr.x - blocker["xy"][0], tr.y - blocker["xy"][1]) < BLOCKER_NEAR_M:
                blocker.update(xy=[round(tr.x, 2), round(tr.y, 2)], v=[tr.vx, tr.vy], track=tr.track_id)  # follow them
                if not self._floor:
                    return True
                return self._clear_of_path(SimpleNamespace(x=tr.x, y=tr.y, vx=tr.vx, vy=tr.vy), world, vid, st)
        # Not in view: judge by where they were last seen (standing there), against the truck driving on from where it
        # stopped at its own speed. Only "last seen clear of the path" lets it go.
        return self._clear_of_path(SimpleNamespace(x=blocker["xy"][0], y=blocker["xy"][1], vx=0.0, vy=0.0), world, vid, st)

    def _clear_of_path(self, person, world: WorldState, vid: str, st: dict) -> bool:
        """Is this person out of the path the stopped truck would take if it drove on from where it stopped?"""
        fork_xy = st.get("stop_xy") or (self.vmeta.get(vid) or {}).get("xy")
        if fork_xy is None and self._floor:  # floor stage: the vehicle key is the truck's own track
            v = next((tr for tr in world.tracks if tr.track_id == vid), None)
            fork_xy = [v.x, v.y] if v is not None else None
        intent = st.get("intent")
        if fork_xy is None or intent is None:
            return False
        fork = SimpleNamespace(x=fork_xy[0], y=fork_xy[1], vx=intent[1] * math.cos(intent[0]), vy=intent[1] * math.sin(intent[0]))
        ttc, _, _ = predict(person, fork, intent[0])
        return ttc is None

    def _cmd(self, t: float, command: str, c: dict) -> dict:
        cmd = {"t": round(t, 3), "vehicle": c["vehicle"], "track": c.get("track"), "command": command, "person": c.get("person"),
               "reason": c.get("reason"), "ttc_s": c.get("ttc_s"), "gap_m": c.get("gap_m"), "speed_mps": c.get("speed_mps"),
               "xy": c.get("xy"), "person_xy": c.get("person_xy"), "applied": None}
        if command == "slow":
            cmd["limit_mps"] = SLOW_MPS
        spd, gap = c.get("speed_mps"), c.get("gap_m")
        if command in ("slow", "stop") and spd is not None and gap is not None:
            # what a stop issued now does, worst case: the truck's stopping distance at its measured speed
            d = stopping_distance(spd)
            cmd["stoppingDistance_m"] = round(d, 2)
            cmd["stopsShortBy_m"] = round(gap - d, 2)
        self.counts[command] += 1
        self.log.append(cmd)
        self.recent.append(cmd)
        return cmd

    def mark_applied(self, cmd: dict, applied: bool | None) -> None:
        cmd["applied"] = applied

    def snapshot(self) -> dict:
        """For the dashboard and the trace: each vehicle's state and the latest commands."""
        return {"enabled": self.enabled,
                "vehicles": {vid: {k: st.get(k) for k in ("state", "since", "reason", "person", "waiting")} for vid, st in self.vehicles.items()},
                "recent": list(self.recent), "counts": dict(self.counts)}
