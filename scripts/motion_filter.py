"""Motion smoothing for floor tracks: a Kalman filter that ignores single-frame jumps, and on/off decisions that do not
flicker. Numpy only; imported by danger_zone.py and real_clip_perception.py.

Why: the detector redraws every outline from scratch each frame, so a forklift's floor position jumped 0.3-1.0 m between
frames (9 in 10 frames; worst 5.9 m) while the truck itself moves about 0.1 m per frame, and "turning" flipped on and off
up to 21 times per clip (audit_motion_noise.py on the 23 Oct run). The zones, distances and warnings inherited that noise.

FloorKF - constant-velocity Kalman filter on floor positions (x, y, vx, vy). Floorsight carries every Monte Carlo draw
          (each draw has its own camera calibration, so its own floor scale) as its own state; all draws share one
          covariance and gain because they see the same measurement times and noise model. Causal (forward filter only),
          as a live system would run it. Measurements whose innovation is beyond GATE (squared Mahalanobis distance on
          the nominal draw, chi-square 2 dof, 99.9 %) are treated as outliers and skipped; RESET_AFTER outliers in a
          row mean the object really moved there, and the filter restarts from the measurement. With
          keep_velocity the restart re-anchors the position only (v10): the outline switched to another reading of
          where the truck meets the floor, the truck did not jump, so its speed is kept and `jump` reports the shift.
TurnKF  - heading and turn rate (v10): a constant-turn-rate-and-velocity extended Kalman filter on the outline's floor
          centre, so "turning" comes from a filtered turn rate with an honest error instead of a 1 s line fit (which
          flickered on straight runs).
Hysteresis - a yes/no decision that needs `frames` consecutive frames above `on` to switch on and below `off` to switch
          off (used for "turning" and "seated").
"""
from __future__ import annotations

import numpy as np

GATE = 13.8          # chi-square, 2 dof, 99.9 %
RESET_AFTER = 4      # outliers in a row -> restart from the measurement


def wrap(a: float) -> float:
    """Angle to (-pi, pi]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def wrap_axis(a: float) -> float:
    """Angle between two body axes, to (-pi/2, pi/2]: a truck pointing the other way has the same axis."""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


class FloorKF:
    def __init__(self, accel_sigma: float, meas_sigma: float, init_speed_sigma: float = 3.0,
                 gate: float = GATE, reset_after: int = RESET_AFTER, keep_velocity: bool = False):
        self.q, self.r = float(accel_sigma) ** 2, float(meas_sigma) ** 2
        self.v0 = float(init_speed_sigma) ** 2
        self.gate, self.reset_after, self.keep_velocity = gate, reset_after, keep_velocity
        self.jump = None         # [D, 2] position shift of the last reset (measurement minus prediction)
        self.pending = []        # (t, z) outliers since the last accepted measurement
        self.x = None            # [D, 4] states
        self.P = None            # [4, 4] shared covariance
        self.t = None            # time of the state
        self.t_update = None     # time of the last accepted measurement
        self.updates = 0
        self.outliers = 0        # consecutive
        self.total_outliers = 0

    # ------------------------------------------------------------------ model
    @staticmethod
    def _F(dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        q = self.q
        a, b, c = dt ** 4 / 4, dt ** 3 / 2, dt ** 2
        Q = np.zeros((4, 4))
        for i in (0, 1):
            Q[i, i], Q[i, i + 2], Q[i + 2, i], Q[i + 2, i + 2] = a * q, b * q, b * q, c * q
        return Q

    def _init(self, t: float, z: np.ndarray) -> None:
        self.x = np.zeros((len(z), 4))
        self.x[:, :2] = z
        self.P = np.diag([self.r, self.r, self.v0, self.v0])
        self.t = self.t_update = t
        self.updates, self.outliers = 1, 0
        self.pending = []

    # ------------------------------------------------------------------ public
    def predict(self, t: float) -> None:
        if self.x is None or t <= self.t:
            return
        dt = t - self.t
        F = self._F(dt)
        self.x = self.x @ F.T
        self.P = F @ self.P @ F.T + self._Q(dt)
        self.t = t

    def update(self, t: float, z) -> str:
        """z: [D, 2] measured floor positions (one per draw). Returns 'init', 'update', 'outlier' or 'reset'."""
        z = np.asarray(z, float).reshape(-1, 2)
        if not np.isfinite(z).all():
            self.predict(t)
            return "skipped"
        if self.x is None or len(z) != len(self.x):
            self._init(t, z)
            return "init"
        self.predict(t)
        S = self.P[:2, :2] + self.r * np.eye(2)
        y = z - self.x[:, :2]
        d2 = float(y[0] @ np.linalg.solve(S, y[0]))
        if d2 > self.gate:
            self.outliers += 1
            self.total_outliers += 1
            self.pending.append((t, z))
            if self.outliers >= self.reset_after:
                vel, Pv = self.x[:, 2:].copy(), self.P[2:, 2:].copy()
                if self.keep_velocity:
                    # re-anchor to the mean of the run of new readings, each moved on to now at the current velocity
                    # (v10: the last reading alone overshot by ~1 m on screen-032 when it was itself a spike)
                    z = np.mean([zi + vel * (t - ti) for ti, zi in self.pending], axis=0)
                self.jump = z - self.x[:, :2]
                self._init(t, z)
                if self.keep_velocity:  # re-anchor: same speed, a little less sure of it
                    self.x[:, 2:] = vel
                    self.P[2:, 2:] = Pv + np.eye(2)
                return "reset"
            return "outlier"
        K = self.P[:, :2] @ np.linalg.inv(S)          # [4, 2]
        self.x = self.x + y @ K.T
        self.P = (np.eye(4) - K @ np.hstack([np.eye(2), np.zeros((2, 2))])) @ self.P
        self.t_update = t
        self.updates += 1
        self.outliers = 0
        self.pending = []
        return "update"

    @property
    def pos(self) -> np.ndarray:
        return self.x[:, :2]

    @property
    def vel(self) -> np.ndarray:
        return self.x[:, 2:]

    def speed_se(self, direction) -> float:
        """Standard error of the speed along a unit direction (from the velocity covariance)."""
        u = np.asarray(direction, float)
        return float(np.sqrt(max(u @ self.P[2:, 2:] @ u, 0.0)))


class TurnKF:
    """Constant turn rate and velocity (CTRV) extended Kalman filter on one unit's floor position (v10). State: x, y,
    v (signed speed along the body axis), theta (body axis), omega (turn rate); speed and turn rate change as random
    walks (accel_sigma m/s2, alpha_sigma rad/s2). Measured: the outline's floor centre, white noise meas_sigma. A
    forklift that reverses passes through v = 0 with the same theta, so reversing never reads as a turn, and the turn
    rate's error is honest because the measurements are the raw, independent per-frame positions (the 1 s line fit
    before it treated overlapping samples as independent and called straight runs turns). Outliers beyond the 2-dof
    gate are skipped; `reset_after` in a row re-anchor the position and keep speed, heading and turn rate."""

    def __init__(self, accel_sigma: float = 3.0, alpha_sigma: float = 1.0, meas_sigma: float = 0.35,
                 gate: float = GATE, reset_after: int = RESET_AFTER):
        self.qa, self.qw, self.r = float(accel_sigma) ** 2, float(alpha_sigma) ** 2, float(meas_sigma) ** 2
        self.gate, self.reset_after = gate, reset_after
        self.x = None            # [x, y, v, theta, omega]
        self.P = None
        self.t = self.t_update = None
        self.updates = self.outliers = 0
        self.pending = []        # (t, z) outliers since the last accepted measurement

    def _init(self, t: float, z: np.ndarray, theta0) -> None:
        self.x = np.array([z[0], z[1], 0.0, 0.0 if theta0 is None else float(theta0), 0.0])
        self.P = np.diag([self.r, self.r, 9.0, 0.25 if theta0 is not None else 3.0, 0.25])
        self.t = self.t_update = t
        self.updates, self.outliers = 1, 0

    @staticmethod
    def _f(s: np.ndarray, dt: float) -> np.ndarray:
        x, y, v, th, w = s
        if abs(w) > 1e-4:
            x += v / w * (np.sin(th + w * dt) - np.sin(th))
            y += v / w * (np.cos(th) - np.cos(th + w * dt))
        else:
            x += v * np.cos(th) * dt
            y += v * np.sin(th) * dt
        return np.array([x, y, v, th + w * dt, w])

    def predict(self, t: float) -> None:
        if self.x is None or t <= self.t:
            return
        dt = t - self.t
        f0 = self._f(self.x, dt)
        F = np.empty((5, 5))
        for i in range(5):  # Jacobian by central differences (5 states: cheap, and no algebra to get wrong)
            e = np.zeros(5)
            e[i] = 1e-5
            F[:, i] = (self._f(self.x + e, dt) - self._f(self.x - e, dt)) / 2e-5
        c, s_ = np.cos(self.x[3]), np.sin(self.x[3])
        G = np.array([[dt * dt / 2 * c, 0.0], [dt * dt / 2 * s_, 0.0], [dt, 0.0], [0.0, dt * dt / 2], [0.0, dt]])
        self.x = f0
        self.P = F @ self.P @ F.T + G @ np.diag([self.qa, self.qw]) @ G.T
        self.t = t

    def update(self, t: float, z, theta0=None) -> str:
        """z: measured floor centre (2,). theta0: the body axis to start from (e.g. the parked outline's long axis)."""
        z = np.asarray(z, float).reshape(2)
        if not np.isfinite(z).all():
            self.predict(t)
            return "skipped"
        if self.x is None:
            self._init(t, z, theta0)
            return "init"
        self.predict(t)
        S = self.P[:2, :2] + self.r * np.eye(2)
        y = z - self.x[:2]
        if float(y @ np.linalg.solve(S, y)) > self.gate:
            self.outliers += 1
            self.pending.append((t, z))
            if self.outliers >= self.reset_after:
                vv = self.x[2] * np.array([np.cos(self.x[3]), np.sin(self.x[3])])
                self.x[:2] = np.mean([zi + vv * (t - ti) for ti, zi in self.pending], axis=0)
                self.pending = []
                self.P[:2, :] = 0.0
                self.P[:, :2] = 0.0
                self.P[0, 0] = self.P[1, 1] = self.r
                self.outliers = 0
                self.t_update = t
                return "reset"
            return "outlier"
        K = self.P[:, :2] @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.P[:2, :]
        self.P = (self.P + self.P.T) / 2
        self.t_update = t
        self.updates += 1
        self.outliers = 0
        self.pending = []
        return "update"

    @property
    def speed(self) -> float:
        return float(self.x[2])

    @property
    def theta(self) -> float:
        return float(self.x[3])

    @property
    def omega(self) -> float:
        return float(self.x[4])

    @property
    def omega_se(self) -> float:
        return float(np.sqrt(max(self.P[4, 4], 0.0)))


class Hysteresis:
    def __init__(self, on: float, off: float, frames: int = 3, state: bool = False):
        self.on, self.off, self.frames, self.state = on, off, frames, state
        self.run = 0

    def update(self, value: float) -> bool:
        crossing = value > self.on if not self.state else value < self.off
        self.run = self.run + 1 if crossing else 0
        if self.run >= self.frames:
            self.state, self.run = not self.state, 0
        return self.state
