"""Cross-camera fusion and Kalman tracking on the floor plane."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from floorsight.contracts import Track, TrackClass, TrackState

FORKLIFT_FOOTPRINT_M = 0.9
SIGMA_CAP_M = 1.5


@dataclass
class FloorDet:
    """One camera's blob projected to the floor."""

    camera_id: str
    xy: np.ndarray
    sigma: float
    footprint: float
    kind: str
    stationary_s: float
    score: float
    bbox: tuple[int, int, int, int]
    foot_px: tuple[float, float]
    height_m: float = 0.0


@dataclass
class Cluster:
    xy: np.ndarray
    sigma: float
    footprint: float
    stationary_s: float
    camera_ids: list[str]
    dets: list[FloorDet] = field(default_factory=list)
    height_m: float = 0.0

    @property
    def person_like(self) -> bool:
        """A narrow column about a person's height (a standing person, not a pallet or a glare patch)."""
        # Standing people come out ~2x taller than wide; a pallet ~1:1, a glare patch flat.
        return self.footprint <= 1.3 and 1.2 <= self.height_m <= 2.3 and self.height_m >= 1.6 * self.footprint


def cluster(dets: list[FloorDet], radius: float, vehicle_radius: float | None = None) -> list[Cluster]:
    """Greedy fusion: at most one detection per camera per cluster, inverse-variance weighted position."""
    groups: list[list[FloorDet]] = []
    for d in sorted(dets, key=lambda d: d.sigma):
        best, best_d = None, max(radius, vehicle_radius or 0.0)
        for g in groups:
            if any(o.camera_id == d.camera_id for o in g):
                continue
            w = np.array([1.0 / o.sigma**2 for o in g])
            c = (np.stack([o.xy for o in g]) * w[:, None]).sum(0) / w.sum()
            dist = float(np.linalg.norm(c - d.xy))
            r = radius
            # A nearby narrow person must not inherit a vehicle's broad fusion gate.
            # Only two vehicle-sized observations justify the larger radius.
            if vehicle_radius and d.footprint >= FORKLIFT_FOOTPRINT_M and all(o.footprint >= FORKLIFT_FOOTPRINT_M for o in g):
                r = vehicle_radius
            if dist < min(r, best_d if best is not None else r):
                best, best_d = g, dist
        if best is None:
            groups.append([d])
        else:
            best.append(d)
    out = []
    for g in groups:
        w = np.array([1.0 / o.sigma**2 for o in g])
        xy = (np.stack([o.xy for o in g]) * w[:, None]).sum(0) / w.sum()
        out.append(
            Cluster(
                xy=xy,
                sigma=float(math.sqrt(1.0 / w.sum() + np.average(np.sum((np.stack([o.xy for o in g]) - xy) ** 2, axis=1), weights=w))),
                footprint=float(np.median([o.footprint for o in g])),
                stationary_s=float(max(o.stationary_s for o in g)),
                camera_ids=sorted(o.camera_id for o in g),
                dets=g,
                height_m=float(np.median([o.height_m for o in g])),
            )
        )
    return out


class _Track:
    def __init__(self, tid: int, kind: str, xy: np.ndarray, sigma: float, t: float):
        self.tid = tid
        self.kind = kind  # "mover" | "object"
        kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_64F)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float64)
        kf.statePost = np.array([[xy[0]], [xy[1]], [0.0], [0.0]])
        kf.errorCovPost = np.diag([sigma**2, sigma**2, 1.0, 1.0]).astype(np.float64)
        self.kf = kf
        self.first_t = t
        self.last_t = t
        self.last_move_t = t
        self.hits = 1
        self.confirmed = False
        self.birth_xy = np.asarray(xy, float).copy()
        self.max_disp = 0.0
        self.votes: Counter = Counter()
        self.camera_ids: list[str] = []
        self.cam_seen: dict[str, float] = {}
        self.blind = False
        self.merged = False
        self.hold_since: float | None = None
        self.seen_xy = np.asarray(xy, float).copy()  # last measured position and velocity, for re-linking
        self.seen_v = np.zeros(2)
        self.measurement_sigma = sigma

    @property
    def xy(self) -> np.ndarray:
        return self.kf.statePost[:2, 0]

    @property
    def v(self) -> np.ndarray:
        return self.kf.statePost[2:, 0]

    @property
    def sigma(self) -> float:
        P = self.kf.errorCovPost
        return float(math.sqrt(max(P[0, 0], P[1, 1])))

    def predict(self, dt: float, accel: float) -> None:
        kf = self.kf
        kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
        q = accel**2
        g = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * q
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = g
        Q[np.ix_([1, 3], [1, 3])] = g
        kf.processNoiseCov = Q
        kf.predict()

    def update(self, xy: np.ndarray, sigma: float, t: float) -> None:
        self.measurement_sigma = sigma
        self.kf.measurementNoiseCov = np.eye(2) * sigma**2
        self.kf.correct(np.asarray(xy, np.float64).reshape(2, 1))
        self.last_t = t
        self.hits += 1
        self.max_disp = max(self.max_disp, float(np.linalg.norm(self.xy - self.birth_xy)))
        self.seen_xy = self.xy.copy()
        self.seen_v = self.v.copy()

    def clamp(self, w: float, d: float) -> None:
        """Keep the state on the floor; stop motion into a wall."""
        s = self.kf.statePost
        x, y = float(s[0, 0]), float(s[1, 0])
        cx, cy = min(max(x, 0.05), w - 0.05), min(max(y, 0.05), d - 0.05)
        if cx != x:
            s[2, 0] = 0.0
        if cy != y:
            s[3, 0] = 0.0
        s[0, 0], s[1, 0] = cx, cy
        self.kf.statePost = s

    def damp_velocity(self, factor: float) -> None:
        s = self.kf.statePost
        s[2:, 0] *= factor
        self.kf.statePost = s

    def inflate(self, var: float, cap: float = SIGMA_CAP_M) -> None:
        """Grow position uncertainty, then cap it by a symmetric rescale (keeps the covariance valid)."""
        P = self.kf.errorCovPost
        P[0, 0] += var
        P[1, 1] += var
        f = np.ones(4)
        for i in (0, 1):
            if P[i, i] > cap**2:
                f[i] = cap / math.sqrt(P[i, i])
        P = P * np.outer(f, f)
        self.kf.errorCovPost = P

    @property
    def cls(self) -> TrackClass:
        if self.kind == "object":
            return TrackClass.object
        if not self.votes:
            return TrackClass.unknown
        return self.votes.most_common(1)[0][0]


class Tracker:
    """Movers (people, forklifts) from moving clusters; objects from static clusters.

    A mover whose prediction falls where no camera sees (coverage 0) is kept as `unseen` with its
    velocity decaying to zero and its uncertainty growing, for up to `unseen_max_s`.
    """

    def __init__(
        self,
        covered: Callable[[float, float], int],
        floor_wh: tuple[float, float],
        presence: Callable[[np.ndarray], tuple[float, list[str]]] | None = None,
        nearest_blind: Callable[[np.ndarray, float], np.ndarray | None] | None = None,
        free_point: Callable[[np.ndarray], np.ndarray] | None = None,
        restricted: Callable[[np.ndarray], bool] | None = None,
    ):
        self.free_point = free_point
        self.restricted = restricted
        self.covered = covered
        self.presence = presence
        self.nearest_blind = nearest_blind
        self.floor_wh = floor_wh
        self.tracks: list[_Track] = []
        self.next_id = 1
        self.t: float | None = None
        self.gate_m = 1.0
        self.trusted_background = False  # set when an empty-floor plate is the background (no startup ghosts)
        self.unseen_max_s = 20.0
        self.unseen_restricted_max_s = 20.0  # bounded position hypothesis; expiry does not prove a safe/empty zone
        self.coast_s = 3.0
        self.confirm_hits = 4
        self.confirm_disp_m = 0.8
        self.object_drop_s = 1.5
        self.velocity_tau_s = 1.5
        self.new_confirmed: list[tuple[np.ndarray, float]] = []  # (birth xy, t) of movers confirmed this step
        self.graveyard: list[_Track] = []  # recently dropped movers whose identity a new track may continue

    def step(self, t: float, moving: list[Cluster], static: list[Cluster]) -> list[Track]:
        dt = 0.1 if self.t is None else max(1e-3, t - self.t)
        self.t = t
        self.new_confirmed = []
        W, D = self.floor_wh
        for tr in self.tracks:
            tr.predict(dt, 2.0 if tr.kind == "mover" else 0.05)
            if tr.kind == "object":
                tr.damp_velocity(0.0)
            tr.clamp(W, D)
            if self.free_point is not None and tr.kind == "mover":
                fx, fy = self.free_point(tr.xy)
                if (fx, fy) != tuple(tr.xy):
                    st = tr.kf.statePost
                    st[0, 0], st[1, 0], st[2, 0], st[3, 0] = fx, fy, 0.0, 0.0
                    tr.kf.statePost = st
            x, y = tr.xy
            tr.blind = tr.kind == "mover" and self.covered(float(x), float(y)) == 0
            if tr.kind == "mover" and t - tr.last_t > 0.3:
                # Vehicles keep rolling along their lane; people who vanish usually stop.
                tau = 4.0 if tr.cls == TrackClass.forklift else self.velocity_tau_s
                tr.damp_velocity(math.exp(-dt / (tau if tr.blind else 0.5 * tau)))
                if tr.blind:
                    tr.inflate(0.15 * dt)
            P = tr.kf.errorCovPost
            if max(P[0, 0], P[1, 1]) > SIGMA_CAP_M**2:
                tr.inflate(0.0)

        movers = [tr for tr in self.tracks if tr.kind == "mover"]
        objects = [tr for tr in self.tracks if tr.kind == "object"]

        # 1) moving clusters -> movers (confirmed first, then tentative)
        used: set[int] = set()
        for group in ([m for m in movers if m.confirmed], [m for m in movers if not m.confirmed]):
            free = [i for i in range(len(moving)) if i not in used]
            for ti, ci in self._assign(group, [moving[i] for i in free], self._mover_gate):
                c = moving[free[ci]]
                tr = group[ti]
                used.add(free[ci])
                tr.update(c.xy, c.sigma, t)
                # A narrow fragment of an established vehicle does not locate its
                # floor centre precisely. Keep filter noise separate from this
                # systematic contact ambiguity; expose the latter to consumers.
                if tr.confirmed and tr.cls == TrackClass.forklift and c.footprint < FORKLIFT_FOOTPRINT_M:
                    tr.measurement_sigma = max(tr.measurement_sigma, 0.6)
                tr.last_move_t = t
                tr.hold_since = None
                tr.camera_ids = c.camera_ids
                self._vote(tr, c, objects)

        # 2) objects <- static clusters
        sused: set[int] = set()
        for ti, ci in self._assign(objects, static, lambda tr: 1.0):
            c = static[ci]
            sused.add(ci)
            tr = objects[ti]
            tr.update(c.xy, max(c.sigma, 0.15), t)
            for cam in c.camera_ids:
                tr.cam_seen[cam] = t

        # 3) a person who has stopped is kept alive by the person-shaped static region they leave.
        #    Bulky static regions (a set-down pallet) are objects and never feed a person or vehicle track.
        for tr in movers:
            if not tr.confirmed or tr.last_t == t or tr.cls == TrackClass.forklift:
                continue
            if np.linalg.norm(tr.v) > 0.8 and t - tr.last_move_t < 0.5:
                continue
            d = [float(np.linalg.norm(c.xy - tr.xy)) if c.person_like else 1e9 for c in static]
            if not d:
                continue
            i = int(np.argmin(d))
            if d[i] < 0.8 and i not in sused:
                c = static[i]
                sused.add(i)
                tr.update(c.xy, max(c.sigma, 0.2), t)
                tr.damp_velocity(0.5)
                tr.camera_ids = c.camera_ids

        # 3b) a stopped person whose silhouette still differs from the long-term background stays where it was
        #     last measured. Not next to a parked object or a moving vehicle, whose own pixels would hold it.
        if self.presence is not None:
            forklifts = [m.xy for m in movers if m.confirmed and m.cls == TrackClass.forklift and m.last_t == t]
            parked = [o.xy for o in objects if o.confirmed]
            bulky = [c.xy for c in static if not c.person_like]
            for tr in movers:
                if not tr.confirmed or tr.last_t == t or tr.blind or t - tr.last_t > 1.0:
                    continue
                if tr.cls == TrackClass.forklift or tr.max_disp < 1.5 or tr.last_move_t - tr.first_t < 2.0:
                    continue  # only someone who has really walked here can be standing here
                at = tr.seen_xy
                if any(np.linalg.norm(at - o) < 1.0 for o in parked + bulky):
                    continue
                if any(np.linalg.norm(at - f) < 2.0 for f in forklifts if f is not tr.xy):
                    continue
                # Someone else standing or walking right there would supply the pixels: not evidence for this track.
                if any(o is not tr and o.confirmed and t - o.last_t < 0.5 and np.linalg.norm(at - o.xy) < 1.2 for o in movers):
                    continue
                # A real stop leaves a person-shaped static region within a few seconds; without one, let go.
                if tr.hold_since is not None and t - tr.hold_since > 3.0 and not any(
                    c.person_like and np.linalg.norm(c.xy - at) < 0.7 for c in static
                ):
                    continue
                frac, cams = self.presence(at)
                if frac >= 0.15:
                    if tr.hold_since is None:
                        tr.hold_since = t
                    tr.damp_velocity(0.0)
                    tr.update(at.copy(), 0.15, t)
                    tr.damp_velocity(0.0)
                    tr.camera_ids = cams

        # 3c) someone who vanished right beside a blind spot most likely walked into it
        if self.nearest_blind is not None:
            for tr in movers:
                if tr.confirmed and not tr.blind and 1.0 <= t - tr.last_t < 1.0 + dt + 1e-6:
                    spot = self.nearest_blind(tr.seen_xy, 1.2)
                    if spot is not None and np.linalg.norm(tr.seen_v) >= 0.25 and float((spot - tr.seen_xy) @ tr.seen_v) > 0:
                        s_ = tr.kf.statePost
                        s_[0, 0], s_[1, 0], s_[2, 0], s_[3, 0] = spot[0], spot[1], 0.0, 0.0
                        tr.kf.statePost = s_
                        tr.blind = True

        # 4) births
        live_movers = [m for m in movers if m.confirmed and t - m.last_t < 2.0]
        for i, c in enumerate(moving):
            if i in used:
                continue
            if any(np.linalg.norm(m.xy - c.xy) < 0.5 for m in live_movers):
                continue
            near_fork = [float(np.linalg.norm(m.xy - c.xy)) for m in live_movers if m.cls == TrackClass.forklift and t - m.last_t < 0.5]
            limit = 2.5 if c.footprint >= FORKLIFT_FOOTPRINT_M else (2.0 if len(c.camera_ids) == 1 else 1.0)
            if any(d < limit for d in near_fork):
                continue  # a loose piece of a vehicle
            tr = _Track(self.next_id, "mover", c.xy, max(c.sigma, 0.2), t)
            tr.camera_ids = c.camera_ids
            self._vote(tr, c, objects)
            self.next_id += 1
            self.tracks.append(tr)
        for i, c in enumerate(static):
            if i in sused or c.stationary_s < (2.0 if c.person_like else 3.0):
                continue
            if not c.person_like and c.footprint < 0.4:
                continue  # a sliver (e.g. the edge of someone in a blind spot), not an object
            # A person standing beside a new bulky object does not stop it being an object; someone walking
            # through it (whose pixels may merge with it) does, until they have gone.
            blockers = [m for m in movers if m.confirmed and t - m.last_t < 2.0]
            if not c.person_like:
                blockers = [m for m in blockers if t - m.last_move_t < 1.0]
            if any(np.linalg.norm(m.xy - c.xy) < 1.0 for m in blockers):
                continue
            if c.person_like:
                # Someone standing still: a person track straight away, if every camera that should see them agrees.
                need = 1 if self.trusted_background and c.stationary_s >= 3.0 else min(2, self.covered(float(c.xy[0]), float(c.xy[1])))
                if len(c.camera_ids) < need:
                    continue
                tr = _Track(self.next_id, "mover", c.xy, max(c.sigma, 0.2), t)
                tr.confirmed = True
                tr.votes[TrackClass.person] += 3
                tr.camera_ids = c.camera_ids
                self.next_id += 1
                self.tracks.append(tr)
                movers.append(tr)
                continue
            if any(np.linalg.norm(o.xy - c.xy) < 1.0 for o in objects):
                continue
            tr = _Track(self.next_id, "object", c.xy, max(c.sigma, 0.15), t)
            for cam in c.camera_ids:
                tr.cam_seen[cam] = t
            self.next_id += 1
            self.tracks.append(tr)

        self._manage(t)
        return [self._to_contract(tr, t) for tr in self.tracks if tr.confirmed]

    @staticmethod
    def _vote(tr: _Track, c: Cluster, objects: list[_Track]) -> None:
        """Class vote from a cluster's width; someone brushing past a parked object looks wide, so no vote there."""
        wide = c.footprint >= FORKLIFT_FOOTPRINT_M
        if wide and any(o.confirmed and np.linalg.norm(o.xy - c.xy) < 1.5 for o in objects):
            return
        tr.votes[TrackClass.forklift if wide else TrackClass.person] += 1

    def _relink_candidate(self, new: _Track) -> tuple[_Track | None, float]:
        """The confirmed track that went out of sight where (and when) this one appeared, if any."""
        best, best_d = None, 1e9
        for old in self.tracks + self.graveyard:
            if old is new or old.kind != "mover" or not old.confirmed or old.merged:
                continue
            gone = new.first_t - old.last_t
            if not (0.3 < gone < 30.0):
                continue
            if old.cls != TrackClass.unknown and new.cls != TrackClass.unknown and old.cls != new.cls and sum(old.votes.values()) >= 10:
                continue
            speed = float(np.linalg.norm(old.seen_v))
            reach = 1.5 + min(speed * gone, 8.0)
            off = new.birth_xy - old.seen_xy
            dist = float(np.linalg.norm(off))
            if dist > reach:
                continue
            if speed > 0.4 and dist > 1.5 and float(off @ old.seen_v) <= 0:
                continue  # it would have had to turn back
            if dist < best_d:
                best, best_d = old, dist
        return best, best_d

    def _relink(self, new: _Track, t: float) -> bool:
        """A track confirmed where an older one went out of sight continues the older identity."""
        best, _ = self._relink_candidate(new)
        if best is None:
            return False
        new.tid, new.first_t = best.tid, best.first_t
        new.votes.update(best.votes)
        new.max_disp = max(new.max_disp, best.max_disp)
        new.last_move_t = max(new.last_move_t, new.first_t + 2.0) if best.last_move_t - best.first_t >= 2.0 else new.last_move_t
        best.merged = True
        self.graveyard = [g for g in self.graveyard if g is not best]
        return True

    def _mover_gate(self, tr: _Track) -> float:
        base = 1.5 if tr.cls == TrackClass.forklift else self.gate_m
        if tr.blind or self.t - tr.last_t > 0.5:
            return min(2.0, base + tr.sigma)
        return base

    def _assign(self, tracks: list[_Track], clusters: list[Cluster], gate_fn) -> list[tuple[int, int]]:
        if not tracks or not clusters:
            return []
        cost = np.full((len(tracks), len(clusters)), 1e6)
        for i, tr in enumerate(tracks):
            g = gate_fn(tr)
            settled = tr.kind == "mover" and sum(tr.votes.values()) >= 10
            for j, c in enumerate(clusters):
                if settled and (c.footprint >= FORKLIFT_FOOTPRINT_M) != (tr.cls == TrackClass.forklift):
                    if tr.cls == TrackClass.person or len(c.camera_ids) > 1:
                        continue  # even one broad view must not drag a person identity onto a vehicle
                d = float(np.linalg.norm(tr.xy - c.xy))
                if d < g:
                    cost[i, j] = d / g
        rows, cols = linear_sum_assignment(cost)
        return [(r, c) for r, c in zip(rows, cols) if cost[r, c] < 1e5]

    def _manage(self, t: float) -> None:
        keep = []
        for tr in self.tracks:
            age = t - tr.first_t
            miss = t - tr.last_t
            if tr.kind == "object":
                if not tr.confirmed and tr.hits >= 3:
                    tr.confirmed = True
                if miss > (self.object_drop_s if tr.confirmed else 0.5):
                    continue
            else:
                if not tr.confirmed:
                    # Someone we just lost reappearing close by needs no fresh proof of being real.
                    quick = tr.hits >= 3 and tr.max_disp >= 0.3 and self._relink_candidate(tr)[1] <= 1.5
                    if quick or (tr.hits >= self.confirm_hits and tr.max_disp >= self.confirm_disp_m):
                        tr.confirmed = True
                        if not self._relink(tr, t):  # a brand-new identity may have stood here since before we watched
                            self.new_confirmed.append((tr.birth_xy.copy(), tr.first_t))
                    elif miss > 0.5 or age > 8.0:
                        continue
                elif tr.blind and miss > 0:
                    limit = self.unseen_restricted_max_s if self.restricted is not None and self.restricted(tr.xy) else self.unseen_max_s
                    if miss > limit:
                        continue
                elif miss > self.coast_s:
                    continue
                x, y = tr.xy
                W, D = self.floor_wh
                if not (-1.0 <= x <= W + 1.0 and -1.0 <= y <= D + 1.0):
                    continue
            keep.append(tr)
        kept = set(id(tr) for tr in keep)
        self.graveyard = [g for g in self.graveyard if t - g.last_t < 30.0] + [
            tr for tr in self.tracks if id(tr) not in kept and tr.kind == "mover" and tr.confirmed and not tr.merged
        ]
        keep = [tr for tr in keep if not tr.merged]
        # Duplicate movers that sit on top of each other: keep the older one.
        keep.sort(key=lambda tr: tr.first_t)
        out: list[_Track] = []
        for tr in keep:
            if tr.kind == "mover" and tr.confirmed and any(
                o.kind == "mover" and o.confirmed and np.linalg.norm(o.xy - tr.xy) < 0.35 and t - o.last_t < 0.5 for o in out
            ):
                continue
            out.append(tr)
        self.tracks = out

    def _to_contract(self, tr: _Track, t: float) -> Track:
        miss = t - tr.last_t
        if tr.kind == "object":
            state = TrackState.seen
            cams = sorted(c for c, ts in tr.cam_seen.items() if t - ts < 2.0)
            vx = vy = 0.0
        else:
            if miss < 0.8:
                state = TrackState.estimated if tr.measurement_sigma > 0.35 or miss > 0.2 else TrackState.seen
            elif tr.blind:
                state = TrackState.unseen
            else:
                state = TrackState.lost
            cams = tr.camera_ids if state in (TrackState.seen, TrackState.estimated) else []
            vx, vy = (float(v) for v in tr.v)
        x, y = (float(v) for v in tr.xy)
        return Track(
            track_id=f"T{tr.tid}",
            cls=tr.cls,
            x=round(x, 3),
            y=round(y, 3),
            vx=round(vx, 3),
            vy=round(vy, 3),
            sigma_m=round(min(max(tr.sigma, tr.measurement_sigma), SIGMA_CAP_M), 3),
            camera_ids=cams,
            first_seen_t=round(tr.first_t, 3),
            last_seen_t=round(tr.last_t, 3),
            state=state,
        )
