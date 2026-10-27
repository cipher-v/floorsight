"""Closed-loop simulator for the forklift interlock (24 Oct).

A recorded video cannot be changed, and near misses cannot be staged with real people. Here the simulated forklift
obeys Floorsight's commands (slow / stop / resume), so the whole loop is measured: rendered camera frames -> the same
vision pipeline as the app -> the interlock -> the forklift's brakes -> the next frame.

People are scripted relative to the forklift (a person steps out when the truck is a few metres away), so the same
four situations happen with the interlock off and on:

    S2  crossing:  a person crosses the main aisle at the Aisle 4 crossing in front of the forklift          -> SLOW/STOP
    S4  tail:      a person walks into the back of the forklift as it drives away (down the east side)    -> HORN, no stop
    S1  in path:   a picker steps from the picking face into the lower lane ahead of the forklift, 6 s     -> STOP
    S3  alongside: a person walks the picking face, 1.15 m from the forklift's side as it passes            -> nothing

Ground truth (true positions, not what the cameras saw) gives each person's closest gap to the forklift body.

    python -m floorsight.sim.live --out <dir> [--duration 50]      (writes nothing heavy by itself; see scripts/interlock_sim.py)
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from floorsight.contracts import FrameMeta, SitePlan
from floorsight.interfaces import FramePacket
from floorsight.interlock import LEVEL, SLOW_MPS
from floorsight.sim.scenario import demo_cameras, demo_site
from floorsight.sim.world import Renderer, make_forklift, make_person
from floorsight.site import FORKLIFT_LENGTH_M, FORKLIFT_WIDTH_M, box_distance

FPS = 10.0
FORK_SPEED = 2.0        # m/s, normal driving speed in the scenario
FORK_ACCEL = 1.0        # m/s2
FORK_DECEL = 2.5        # m/s2, inside the 1.5-3.0 braking range the floor stage assumes
ACTUATION_S = 0.2       # radio + brake actuation after the command; the vision frame adds 0.1 s
ROUTE = [(1.5, 5.0), (13.0, 5.0), (13.0, 1.2), (1.5, 1.2)]   # main aisle east, down, lower lane west, up (loop)
CONTACT_M = 0.05        # true gap below this = the forklift touched the person
START_S = 0.0           # start at the west end of the main aisle, heading east


class RouteFollower:
    """Closed polyline; position and heading at a distance along it."""

    def __init__(self, points):
        self.points = [np.asarray(p, float) for p in points]
        self.seg = []
        s = 0.0
        for i in range(len(self.points)):
            a, b = self.points[i], self.points[(i + 1) % len(self.points)]
            L = float(np.linalg.norm(b - a))
            self.seg.append((s, L, a, b))
            s += L
        self.length = s

    def at(self, s: float):
        s %= self.length
        for s0, L, a, b in self.seg:
            if s <= s0 + L:
                k = (s - s0) / L
                p = a + k * (b - a)
                return float(p[0]), float(p[1]), math.atan2(b[1] - a[1], b[0] - a[0])
        a, b = self.seg[-1][2], self.seg[-1][3]
        return float(b[0]), float(b[1]), math.atan2(b[1] - a[1], b[0] - a[0])


@dataclass
class ControlledForklift:
    """A forklift that follows its route at FORK_SPEED unless commanded: slow -> SLOW_MPS, stop -> 0, resume -> normal.
    Commands take effect ACTUATION_S after they are issued; speed changes at FORK_ACCEL / FORK_DECEL."""

    route: RouteFollower = field(default_factory=lambda: RouteFollower(ROUTE))
    speed_normal: float = FORK_SPEED
    s: float = START_S
    v: float = FORK_SPEED
    mode: str = "normal"
    pending: list = field(default_factory=list)   # (t_effective, vehicle track id, mode)
    requests: dict = field(default_factory=dict)  # vehicle track id -> slow/stop; the truck obeys the strictest one
    log: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def command(self, cmd: dict, t: float) -> bool:
        mode = {"slow": "slow", "stop": "stop", "resume": "normal"}.get(cmd.get("command"))
        if mode is None:
            return False
        with self.lock:  # the app's runtime thread commands, its frame thread steps
            self.pending.append((t + ACTUATION_S, str(cmd.get("vehicle")), mode))
        return True

    def step(self, t: float, dt: float) -> None:
        with self.lock:
            due = sorted(p for p in self.pending if p[0] <= t + 1e-9)
            self.pending = [p for p in self.pending if p[0] > t + 1e-9]
        # The vision can lose a standing truck and find it again under a new track id; a "resume" for the old id must
        # not release a stop requested under the new one, so each id's request is kept and the strictest one wins.
        for _, vid, mode in due:
            if mode == "normal":
                self.requests.pop(vid, None)
            else:
                self.requests[vid] = mode
        new = max(self.requests.values(), key=lambda m: LEVEL[m], default="normal")
        if new != self.mode:
            self.log.append({"t": round(t, 2), "mode": new, "v": round(self.v, 2)})
        self.mode = new
        target = {"normal": self.speed_normal, "slow": SLOW_MPS, "stop": 0.0}[self.mode]
        if self.v > target:
            self.v = max(target, self.v - FORK_DECEL * dt)
        else:
            self.v = min(target, self.v + FORK_ACCEL * dt)
        self.s += self.v * dt

    def pose(self):
        x, y, h = self.route.at(self.s)
        return x, y, h, self.v * math.cos(h), self.v * math.sin(h)


class Walker:
    """A scripted person: waits at `start`; once `trigger(fork_xy, t)` is true, walks the legs [(x, y, wait_s), ...]."""

    def __init__(self, pid: str, scene: str, shirt, start, legs, trigger, speed: float = 1.3):
        self.pid, self.scene, self.shirt = pid, scene, shirt
        self.pos = np.asarray(start, float)
        self.legs = [(np.asarray(p[:2], float), p[2]) for p in legs]
        self.trigger, self.speed = trigger, speed
        self.started_t = None
        self.leg = 0
        self.wait_until = None
        self.v = np.zeros(2)

    def step(self, t: float, dt: float, fork) -> None:
        """fork: (x, y, distance driven since the start)."""
        self.v = np.zeros(2)
        if self.started_t is None:
            if not self.trigger(fork, t):
                return
            self.started_t = t
        if self.leg >= len(self.legs):
            return
        target, wait = self.legs[self.leg]
        if self.wait_until is not None:
            if t >= self.wait_until:
                self.wait_until = None
                self.leg += 1
            return
        d = target - self.pos
        n = float(np.linalg.norm(d))
        stepd = self.speed * dt
        if n <= stepd:
            self.pos = target.copy()
            if wait > 0:
                self.wait_until = t + wait
            else:
                self.leg += 1
        else:
            self.v = d / n * self.speed
            self.pos = self.pos + self.v * dt


def interlock_walkers() -> list[Walker]:
    """Every situation happens where the cameras can see it (the first closed-loop run, interlock-sim-01, put the picker
    in the middle of the rack aisle: only cam4 looks down that aisle and the approaching forklift blocks its view, so
    Floorsight never saw the picker; that blind spot is reported, not hidden)."""
    lap1 = lambda f, lo, hi: lo <= f[2] < hi  # noqa: E731  (distance driven since the start)
    return [
        # S2: crosses the main aisle at the Aisle 4 crossing as the forklift approaches it (first lap)
        Walker("w2", "S2 crossing", (200, 60, 170), (11.8, 7.6), [(11.8, 2.6, 0.0)],
               lambda f, t: lap1(f, 0.0, 11.5) and 11.8 - f[0] < 4.6, speed=1.4),
        # S4: walks into the back half of the forklift as it drives south down the east side (stopping would not help)
        Walker("w4", "S4 tail", (180, 80, 40), (14.7, 2.6), [(12.0, 2.6, 0.0)],
               lambda f, t: lap1(f, 11.5, 15.3) and f[1] < 3.5, speed=1.4),
        # S1: a picker steps from the picking face into the lower lane ahead of the forklift, stays 6 s, steps back
        Walker("w1", "S1 in path", (60, 150, 60), (6.0, 2.45), [(6.0, 1.2, 6.0), (6.0, 2.45, 0.0)],
               lambda f, t: lap1(f, 15.3, 26.8) and f[0] < 12.2, speed=1.0),
        # S3: walks the picking face, 1.15 m from the forklift's side, as it passes on the lower lane (no stop expected)
        Walker("w3", "S3 alongside", (40, 140, 220), (12.0, 2.9), [(3.5, 2.9, 0.0)],
               lambda f, t: lap1(f, 15.3, 26.8) and f[0] < 12.9, speed=1.1),
    ]


@dataclass
class TruthLog:
    gaps: dict = field(default_factory=dict)   # pid -> list of (t, gap_m, fork_v)

    def add(self, t, pid, gap, v):
        self.gaps.setdefault(pid, []).append((t, gap, v))

    def summary(self, walkers) -> dict:
        out = {}
        for w in walkers:
            g = self.gaps.get(w.pid, [])
            started = [x for x in g if w.started_t is not None and x[0] >= w.started_t]
            if not started:
                out[w.scene] = {"person": w.pid, "started": False}
                continue
            t, gap, v = min(started, key=lambda x: x[1])
            out[w.scene] = {"person": w.pid, "started": True, "startedT": round(w.started_t, 2), "closestGap_m": round(gap, 3),
                            "closestT": round(t, 2), "forkliftSpeedAtClosest_mps": round(v, 2), "contact": gap < CONTACT_M}
        return out


class ClosedLoopSim:
    """One run of the interlock scenario. step() renders the four cameras at time t after moving everyone."""

    def __init__(self, render: bool = True, seed: int = 11):
        self.cams = demo_cameras()
        self.site_dict = demo_site(self.cams)
        self.site = SitePlan.model_validate(self.site_dict)
        self.renderer = Renderer(self.site_dict, self.cams, seed=seed) if render else None
        self.fork = ControlledForklift()
        self.walkers = interlock_walkers()
        self.truth = TruthLog()
        self.frame = 0

    def plates(self) -> dict:
        r = Renderer(self.site_dict, self.cams, seed=99)
        return {c.camera_id: r.render(c, 0.0, []) for c in self.cams}

    def advance(self, t: float, dt: float) -> dict:
        if t > 0:
            self.fork.step(t, dt)
        fx, fy, fh, fvx, fvy = self.fork.pose()
        driven = self.fork.s - START_S
        for w in self.walkers:
            w.step(t, dt if t > 0 else 0.0, (fx, fy, driven))
            gap = box_distance(float(w.pos[0]), float(w.pos[1]), fx, fy, fh, FORKLIFT_LENGTH_M, FORKLIFT_WIDTH_M)
            self.truth.add(t, w.pid, gap, self.fork.v)
        return {"t": t, "forklift": {"x": fx, "y": fy, "heading": fh, "v": self.fork.v, "mode": self.fork.mode},
                "people": {w.pid: {"x": float(w.pos[0]), "y": float(w.pos[1]), "scene": w.scene} for w in self.walkers}}

    def render(self, state: dict) -> dict:
        f = state["forklift"]
        bodies = [make_forklift(f["x"], f["y"], f["heading"])] + [make_person(float(w.pos[0]), float(w.pos[1]), w.shirt) for w in self.walkers]
        return {c.camera_id: self.renderer.render(c, state["t"], bodies) for c in self.cams}

    def packets(self, images: dict, t: float) -> dict[str, FramePacket]:
        out = {}
        for c in self.cams:
            img = images[c.camera_id]
            out[c.camera_id] = FramePacket(FrameMeta(camera_id=c.camera_id, frame_index=self.frame, t=round(t, 3), width=img.shape[1],
                                                     height=img.shape[0], faces_blurred=0, blur_applied=False), img)
        self.frame += 1
        return out


class SimLiveSource:
    """Frame source for the app (``--source simlive``): renders the interlock scenario frame by frame and lets the
    runtime's interlock drive the forklift (``command``). Not real time: every frame is processed (closed loop)."""

    realtime = False
    feedback_driven = True  # advance the next frame only after applying this frame's commands

    def __init__(self, duration_s: float = 50.0, site: SitePlan | None = None, loop: bool = False):
        if loop and duration_s * FPS < 2:
            raise ValueError("A looping scenario needs at least two frames")
        self.sim = ClosedLoopSim(render=True)
        self.site = self.sim.site
        self.fps = FPS
        self.duration_s = duration_s
        self.loop = loop
        self._closed = False

    def setup_plates(self) -> dict:
        return self.sim.plates()

    def __iter__(self) -> Iterator[dict[str, FramePacket]]:
        dt = 1.0 / FPS
        while not self._closed:
            for i in range(int(self.duration_s * FPS)):
                if self._closed:
                    return
                t = i * dt
                state = self.sim.advance(t, dt)
                yield self.sim.packets(self.sim.render(state), t)
            if not self.loop or self._closed:
                return
            # Frame index returns to zero, so Runtime starts a fresh vision,
            # agent and interlock session. No brake requests or walkers carry
            # over from the previous scenario.
            self.sim = ClosedLoopSim(render=True)

    def command(self, cmd: dict) -> bool:
        return self.sim.fork.command(cmd, cmd.get("t", 0.0))

    def close(self) -> None:
        self._closed = True


__all__ = ["ClosedLoopSim", "ControlledForklift", "RouteFollower", "SimLiveSource", "interlock_walkers", "LEVEL"]
