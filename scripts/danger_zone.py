"""Forklift danger zone on the calibrated floor, with uncertainty. GCP ONLY (imported by the perception run).

For every moving forklift unit (truck plus carried load) and every Monte Carlo draw (camera calibration
sample + priors below) this builds, on the floor in metres:
  stop  - the area the unit will sweep before it can stop: it keeps its measured speed and turn rate for
          the driver's reaction time, then brakes. Stopping distance = v*t_react + v^2 / (2*a).
  tail  - tail swing: area swept by the rear half of the unit outside the area swept by its front half
          (counterbalance trucks steer with the rear wheels, so the rear swings out in turns).
  blind - the driver's blind zone in front of a load taller than the driver's eye: the floor region in
          the load's shadow seen from the eye, out to BLIND_REACH_M. The load height is measured from
          the pixels (top of the mask above its ground line). OSHA 29 CFR 1910.178(n)(4): when the load
          obstructs the forward view, travel with the load trailing. Travelling load-first while the
          load is taller than the eye is flagged.
Each zone is accumulated over draws into a probability map; the overlay shows the "possible" (>=5% of
draws) and "likely" (>=50%) outlines. A person's chance of being inside a zone is the share of draws in
which their floor point (same draw) falls inside that draw's zone. Speeds and turn rates come only from
the past second of floor positions (causal, as a live system would see them).

Motion smoothing (v9, 24 Oct): every forklift's and person's floor position first goes through a causal Kalman filter
(motion_filter.FloorKF) that rejects single-frame jumps of the detector's outline; the forklift's footprint keeps a
steady shape (length and side offsets from the last second, robust statistics) at the filtered position; its heading is
held while it is slow or stopped instead of being re-guessed from the outline; "turning" and "seated" switch only after
3 or 5 frames in a row; and a forklift the detector misses for a moment keeps its zone for up to COAST_MAX_S on the
filter's prediction (marked "coasting").
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("cvsafe", __import__("pathlib").Path(__file__).with_name("cvsafe.py"))
cvsafe = _ilu.module_from_spec(_spec); _spec.loader.exec_module(cvsafe)
_spec = _ilu.spec_from_file_location("motion_filter", __import__("pathlib").Path(__file__).with_name("motion_filter.py"))
mf = _ilu.module_from_spec(_spec); _spec.loader.exec_module(mf)

PRIORS = {
    "reaction_s": [0.75, 1.5],        # assumption: alerted-driver perception-reaction range
    "decel_mps2": [1.5, 3.0],         # 3.0 ~ manufacturer test (3 t truck stops in 3 m from 15 km/h); 1.5 assumed gentle braking with a tall load
    "eye_height_m": [1.85, 2.15],     # assumption: seated counterbalance operator
    "eye_to_load_face_m": [1.2, 1.8], # assumption: seat to load face (forks, mast)
    "vehicle_width_m": [1.0, 1.3],
    "vehicle_length_m": [2.2, 3.8],   # truck with or without a carried stillage
    "contact_height_m": [0.0, 0.25],  # height of the lowest visible vehicle edge
    "seated_foot_height_m": [0.0, 0.15],
    "load_face_to_drive_axle_m": [1.3, 1.8],  # assumption: stillage depth + fork heel to front (drive) axle
}
BLIND_REACH_M = 6.0      # drawn reach of the blind zone (a vendor system's first warning stage is 6 m)
TALL_LOAD_COLUMN_M = 2.3 # a mask column this tall above its ground point belongs to the load, not the truck
MOVING_MPS = 0.3
OMEGA_MIN = 0.15        # rad/s; a real forklift turn is ~0.3 rad/s or more (applied per draw, v6d)
WINDOW_S = 1.0
GRID_M = 0.10  # 10 cm floor raster (display and zone membership)
EDGE_PX = 8    # mask rows this close to the bottom image edge are a cut, not floor contact (v6f; masks stop ~6 px short)
COLOURS = {"stop": (40, 40, 235), "tail": (200, 60, 200), "blind": (230, 140, 30), "warn": (0, 175, 255), "blindTrailing": (200, 180, 150)}
NAMES = {"stop": "stopping zone", "tail": "tail swing", "blind": "driver blind zone", "warn": "3 s warning zone",
         "blindTrailing": "blind zone, inactive (load trailing)"}
SPEED_MIN_SPAN_S = 0.6     # v7: a track shorter than this has no speed (v6: 0.4 s, gave a 4.6 m/s spike)
ACCEL_LIMITS = (-3.0, 2.0) # m/s2, physical limits for the extrapolated along-track acceleration (v7)
V_MAX = 5.0                # m/s, cap for the 3 s prediction (v7)
WARN_HORIZON_S = 3.0       # v7 warning: will the unit come within WARN_CLEAR_M in the next 3 s if it keeps its course?
WARN_STEP_S = 0.1
WARN_CLEAR_M = 0.8
# v9 motion smoothing (tuned on the 23 Oct run: outline jitter ~0.3-0.5 m per frame; forklifts brake at up to 3 m/s2)
SMOOTHING = True
VEH_ACCEL_SIGMA = 3.0      # m/s2, Kalman process noise for forklifts
VEH_MEAS_SIGMA = 0.35      # m, floor position noise of one frame's outline
PER_ACCEL_SIGMA = 2.0      # m/s2, people
PER_MEAS_SIGMA = 0.25      # m, a person's floor point from one box
PER_RESTART_S = 0.3        # v10: a person unseen this long starts a fresh filter (and seated/standing call) when seen again:
                           # clip-dev-002's seated worker, lost 0.8 s, reappeared standing 2.8 m away; the filter read that as
                           # 3.5 m/s walking and carried a ghost of them to 0.65 m from the forklift
KF_RESTART_S = 1.0         # no accepted measurement for this long: start a fresh filter
COAST_MAX_S = 0.5          # a forklift the detector misses keeps its zone this long on the filter's prediction
TURN_ON, TURN_OFF, TURN_FRAMES = 0.20, 0.10, 3   # rad/s; "turning" needs 3 frames above 0.20 and ends after 3 below 0.10
SEAT_FRAMES = 5            # a person's seated/standing call changes only after 5 frames in a row
SHAPE_WINDOW_STAT = "median"  # each frame's outline points are clipped to the median extent of the last second (per side)
SHAPE_SMOOTH_S = 0.4       # the footprint's reach in each of 16 directions is the median over the last 0.4 s
SHAPE_NORMALS = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0.0, 2 * np.pi, 16, endpoint=False)])
HEADING_MIN_MPS = 0.6      # below this the travel direction from a 1 s fit is noise: the footprint keeps its last heading
HEADING_RATE_MAX = 1.0     # rad/s: the footprint's heading turns no faster than a forklift can (clip-dev-004 read 2.1 rad/s)
# v10 (24 Oct, second pass over the smoothed video and records)
VEH_RESET_AFTER = 6        # the outline can switch for a few frames between two readings of where the truck meets the floor
                           # (screen-032: ~1.2 m apart); re-anchor to the new reading only after 6 frames in a row (0.24 s)
TURN_ALPHA_SIGMA = 1.0     # rad/s2: how fast a forklift's turn rate changes (turn-rate filter, motion_filter.TurnKF)
RESTART_SLOW_S = 0.4       # a truck slower than HEADING_MIN_MPS for this long has stopped: it takes its next direction at once
REVERSE_FRAMES = 5         # the travel direction flips (a reversing truck) only after it reads reversed 5 frames in a row:
                           # a 1 s fit read screen-002's forklift reversed for 2 frames at 1.1 m/s as it left the picture
AXIS_SWAP_FRAMES = 5       # a parked outline's long axis changes by more than 30 deg only after 5 frames in a row (a near-
                           # square outline swapped its long and short sides for a frame, an 84 deg swing)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fc = _load("floor_calibration")
fr = _load("floor_retest")


def clip_halfplane(poly, normal, offset):
    """Part of a convex polygon with p . normal <= offset (Sutherland-Hodgman, one plane)."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        da, db = a @ normal - offset, b @ normal - offset
        if da <= 0:
            out.append(a)
        if da * db < 0:
            out.append(a + (b - a) * (da / (da - db)))
    return np.array(out)


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])



def fill(mask, pts, value) -> None:
    """Fill a shape that is convex by construction with cv2.fillConvexPoly (see the COOL note in DangerZone.g)."""
    pts = np.asarray(pts, np.int32).reshape(-1, 2)
    if len(pts) < 3:
        return
    if not cv2.isContourConvex(pts.reshape(-1, 1, 2)):
        # step() only builds convex shapes; integer rounding can leave duplicate/collinear points that fail the test.
        # Their hull is the intended shape. (These are the inputs that crashed COOL's fillPoly.)
        import os
        if os.environ.get("FS_DEBUG_FILLPOLY"):  # keep up to 300 such inputs for an isolated COOL reproduction
            fill.saved = getattr(fill, "saved", 0) + 1
            if fill.saved <= 300:
                np.savez(f"{os.environ['FS_DEBUG_FILLPOLY']}.{fill.saved:03d}.npz", pts=pts, shape=np.asarray(mask.shape))
        pts = cv2.convexHull(pts.reshape(-1, 1, 2)).reshape(-1, 2)
    cvsafe.fill_convex(mask, pts, value)


def safe_pts(a, lim: float = 1e5) -> np.ndarray:
    """Polygon -> int32 points for cv2 drawing: non-finite points dropped, the rest clipped to +-lim pixels."""
    a = np.atleast_2d(np.asarray(a, np.float64))
    a = a[np.isfinite(a).all(1)]
    return np.round(np.clip(a, -lim, lim)).astype(np.int32).reshape(-1, 2)


class DangerZone:
    def __init__(self, calibration: Path, size, fps: float, draws: int = 64, seed: int = 11):
        cal = json.loads(Path(calibration).read_text())
        self.size, self.fps = tuple(size), float(fps)
        self.nominal = fc.Camera(cal["K"], cal["R_worldAxesInCamera"], cal["cameraHeight_m"])
        samples = cal["monteCarloSamples"]
        rng = np.random.default_rng(seed)
        self.draws = [dict(cam=self.nominal, **{k: float(np.mean(v)) for k, v in PRIORS.items()})]
        for _ in range(draws - 1):
            s = samples[rng.integers(len(samples))]
            self.draws.append(dict(cam=fc.Camera(s["K"], s["R"], s["h"]), **{k: float(rng.uniform(*v)) for k, v in PRIORS.items()}))
        for k, d in enumerate(self.draws):  # turn-rate error: nominal + SE * z (v6d; drawn last so other samples are unchanged)
            d["omega_z"] = 0.0 if k == 0 else float(rng.standard_normal())
        for k, d in enumerate(self.draws):  # speed-fit error (v7b; drawn after the turn-rate numbers, which stay unchanged)
            d["speed_z"] = 0.0 if k == 0 else float(rng.standard_normal())
        # floor raster covering the visible floor (nominal camera), for probability maps
        rows = np.linspace(size[1] * 0.3, size[1] - 1, 12)
        cols = np.linspace(0, size[0] - 1, 12)
        pts = self.nominal.to_plane(np.array([[c, r] for r in rows for c in cols]))
        pts = pts[np.isfinite(pts).all(1)]
        self.lo = np.clip(np.floor(pts.min(0)) - 3.0, -35, 35)
        self.hi = np.clip(np.ceil(pts.max(0)) + 3.0, -35, 35)
        self.shape = (int((self.hi[1] - self.lo[1]) / GRID_M) + 1, int((self.hi[0] - self.lo[0]) / GRID_M) + 1)
        # Each draw has its own floor frame (its own scale). Display maps live in the nominal floor frame, so each
        # draw's raster is warped draw-floor -> image -> nominal floor (a planar homography, cv2.warpPerspective).
        G = np.array([[1 / GRID_M, 0, -self.lo[0] / GRID_M], [0, 1 / GRID_M, -self.lo[1] / GRID_M], [0, 0, 1.0]])
        Hn = self.nominal.H_img_to_floor()
        self.warp = [None] + [G @ Hn @ np.linalg.inv(d["cam"].H_img_to_floor()) @ np.linalg.inv(G) for d in self.draws[1:]]
        self.hist = {}  # vehicle id -> list of (t, per-draw centroids [D,2]); filtered positions when SMOOTHING
        self.kf, self.pkf = {}, {}          # vehicle / person id -> FloorKF
        self.vshape = {}                     # vehicle id -> list of (t, per-draw shape tuples) over the last second
        self.vlast = {}                      # vehicle id -> last footprint relative to its filtered centre, per draw
        self.vturn, self.vhead, self.vstatic = {}, {}, {}  # turning hysteresis; last travel heading; parked orientation
        self.tkf = {}                        # vehicle id -> TurnKF (heading and turn rate, nominal draw)
        self.vaxis = {}                      # (vehicle id, draw) -> body-axis direction the stored shapes are measured in
        self.vrev, self.vswap = {}, {}       # frames in a row the travel direction read reversed; the parked axis swapped
        self.pseat = {}                      # person id -> Hysteresis for "seated"
        self.pseen = {}                      # person id -> last time seen
        self.vmoving = {}                    # vehicle id -> (travel direction measured last frame, slow since) (v10)
        self.motion_stats = {"init": 0, "update": 0, "outlier": 0, "reset": 0, "skipped": 0, "coasted": 0,
                             "personOutliers": 0, "seatedFlipsSuppressed": 0}
        self.load = {}  # vehicle id -> list of (t, measured load height m, load-end floor point)
        self.camera_xy = np.zeros(2)
        self.priors = PRIORS

    # ------------------------------------------------------------------ helpers
    def g(self, xy):
        """floor metres -> raster pixel (int32). Non-finite points (a floor projection near the horizon) are replaced by
        the polygon's finite centre and every point is clipped to a few raster widths: cv2.fillPoly must never see
        INT_MIN (casting NaN), which the COOL 5.1 build rejects (and in one case crashed on) while stock 5.0 clips it."""
        xy = np.atleast_2d(np.asarray(xy, np.float64))
        bad = ~np.isfinite(xy).all(1)
        if bad.any():
            self.nonfinite_points = getattr(self, "nonfinite_points", 0) + int(bad.sum())
            fin = xy[~bad]
            xy = xy.copy()
            xy[bad] = fin.mean(0) if len(fin) else np.array([self.lo[0] - 1.0, self.lo[1] - 1.0])
        p = np.column_stack([(xy[:, 0] - self.lo[0]) / GRID_M, (xy[:, 1] - self.lo[1]) / GRID_M])
        lim = 4.0 * max(self.shape)
        return np.round(np.clip(p, -lim, lim)).astype(np.int32)

    def g_inv(self, pix):
        pix = np.atleast_2d(pix).astype(float)
        return np.column_stack([pix[:, 0] * GRID_M + self.lo[0], pix[:, 1] * GRID_M + self.lo[1]])

    def _load_columns(self, polygons):
        """Measured height above its own ground point of each mask column (nominal camera)."""
        mask = np.zeros((self.size[1], self.size[0]), np.uint8)
        cvsafe.fill_poly(mask, [safe_pts(p) for p in polygons if len(safe_pts(p)) >= 3], 255)
        cols = np.nonzero(mask.any(axis=0))[0][::6]
        out = []
        for x in cols:
            ys = np.nonzero(mask[:, x])[0]
            if ys.max() >= self.size[1] - 3:
                continue
            h = self.nominal.height_of((x + 0.5, ys.max() + 1.0), (x + 0.5, ys.min()))
            if np.isfinite(h):
                out.append((x + 0.5, ys.max() + 1.0, h))
        return np.array(out) if out else np.zeros((0, 3))

    def _footprint(self, vid, k, gk, heading, d, t, cur):
        """Footprint from visible near-side ground points gk (already moved to the filtered position): the v8 construction
        (floor_retest.rect_footprint: model rectangle through the near side, hull with the points seen), but each frame's
        points are first clipped to the outline's usual extent - in the truck's own frame, the median of each side's
        extreme over the last second. One frame's flare is cut back; what the outline shows consistently (a wide load, a
        body rotating at low speed) stays inside the footprint (v9)."""
        # v10: the stored shapes are measured along the body axis, pointed the same way as last frame's: a truck that
        # reverses (or starts moving "backwards" from its parked axis) has the same footprint, so its history must not
        # flip front for back (clip-dev-002 jumped 1.27 m as the truck pulled away). rect_footprint is symmetric in this.
        ax = self.vaxis.get((vid, k))
        if ax is not None and np.cos(heading - ax) < 0:
            heading = heading + np.pi
        self.vaxis[(vid, k)] = heading
        u = np.array([np.cos(heading), np.sin(heading)])
        w = np.array([-u[1], u[0]])
        pu, pw = gk @ u, gk @ w
        mu, mw = float(pu.mean()), float(pw.mean())
        ext = (float(pu.min() - mu), float(pu.max() - mu), float(pw.min() - mw), float(pw.max() - mw))
        past = [h[k] for tt, h in self.vshape.get(vid, []) if t - tt <= WINDOW_S + 1e-6 and k in h]
        lo_u, hi_u, lo_w, hi_w = (float(np.median([q[0][i] for q in past] + [ext[i]])) for i in range(4))
        ru = np.clip(pu - mu, min(lo_u, 0.0), max(hi_u, 0.0))
        rw = np.clip(pw - mw, min(lo_w, 0.0), max(hi_w, 0.0))
        pts = np.outer(mu + ru, u) + np.outer(mw + rw, w)
        hull = fr.rect_footprint(pts, heading, self.camera_xy, d["vehicle_length_m"], d["vehicle_width_m"])
        # shape smoothing: reach of the footprint in 16 directions (truck frame, about its filtered centre), median over
        # the last SHAPE_SMOOTH_S, rebuilt as the intersection of the 16 half-planes (a convex shape, like the input)
        c = np.array([mu, mw])
        rel = np.column_stack([hull @ u, hull @ w]) - c
        reach = (rel @ SHAPE_NORMALS.T).max(0)
        cur[k] = (ext, reach)
        recent = [h[k][1] for tt, h in self.vshape.get(vid, []) if t - tt <= SHAPE_SMOOTH_S + 1e-6 and k in h] + [reach]
        reach_s = np.median(np.array(recent), axis=0)
        poly = np.array([[-60.0, -60.0], [60.0, -60.0], [60.0, 60.0], [-60.0, 60.0]])
        for n_, h_ in zip(SHAPE_NORMALS, reach_s):
            poly = clip_halfplane(poly, n_, float(h_))
            if len(poly) < 3:
                return hull  # degenerate (should not happen: every reach is measured from one convex shape)
        return np.outer(c[0] + poly[:, 0], u) + np.outer(c[1] + poly[:, 1], w)

    def _prune(self, t):
        for store in (self.kf, self.pkf, self.tkf):
            for key in [k_ for k_, f_ in store.items() if f_.t_update is not None and t - f_.t_update > 3.0]:
                store.pop(key, None)

    # ------------------------------------------------------------------ per frame
    def step(self, t: float, vehicles: list[dict], persons: list[dict]) -> dict:
        """vehicles: dicts with id, polygons (image px), moving, estimated. persons: id, box, foot, operator."""
        D = len(self.draws)
        counts = {k: np.zeros(self.shape, np.uint16) for k in COLOURS}
        vstates, results = [], {}
        person_pts = {}
        for p in persons:
            if p.get("operator"):
                continue
            x1, y1, x2, y2 = p["box"]
            seated = (y2 - y1) < 2.0 * (x2 - x1)
            if SMOOTHING:  # v9: a box that is briefly wider than tall must not flip the foot height for one frame
                if t - self.pseen.get(p["id"], -1e9) > PER_RESTART_S:
                    self.pseat.pop(p["id"], None)  # v10: seen again after a gap: take the box as it is now
                self.pseen[p["id"]] = t
                hy = self.pseat.setdefault(p["id"], mf.Hysteresis(0.5, 0.5, SEAT_FRAMES, state=seated))
                raw_seated, seated = seated, hy.update(1.0 if seated else 0.0)
                self.motion_stats["seatedFlipsSuppressed"] += int(raw_seated != seated)
            pts = np.array([d["cam"].to_plane(np.array(p["foot"], float), d["seated_foot_height_m"] if seated else 0.0)[0] for d in self.draws])
            if SMOOTHING and np.isfinite(pts).all():
                kf = self.pkf.get(p["id"])
                if kf is None or t - kf.t_update > PER_RESTART_S:
                    kf = self.pkf[p["id"]] = mf.FloorKF(PER_ACCEL_SIGMA, PER_MEAS_SIGMA, init_speed_sigma=1.5)
                self.motion_stats["personOutliers"] += int(kf.update(t, pts) in ("outlier", "reset"))
                pts = kf.pos.copy()
            person_pts[p["id"]] = (pts, seated)
            ok_ = np.isfinite(pts).all(1)
            xy0 = pts[0] if np.isfinite(pts[0]).all() else (pts[ok_].mean(0) if ok_.any() else np.array([np.nan, np.nan]))
            spread = float(np.median(np.linalg.norm(pts[ok_] - xy0, axis=1))) if ok_.any() else float("nan")
            results[p["id"]] = {"byVehicle": {}, "seated": seated,
                                "xy_m": [round(float(xy0[0]), 3), round(float(xy0[1]), 3)] if np.isfinite(xy0).all() else None,
                                "xySigma_m": round(spread, 3) if np.isfinite(spread) else None}
        for v in vehicles:
            vid = v["id"]
            ground, loads, cut = None, [], False
            if not v.get("estimated") and v.get("polygons"):
                loads = [q for q in (v.get("loadPolygons") or []) if len(q) >= 3]
                # floor contact from the truck with its carried loads cut out (v8); truck plus load stays one unit for the edge checks
                px, umask = fr.near_side_pixels_excluding(list(v["polygons"]), loads, self.size, edge_px=EDGE_PX)
                cut = bool(umask[-EDGE_PX:, :].any())  # unit runs off the bottom of the picture: floor contact out of view
                if umask[:, :EDGE_PX].any() or umask[:, -EDGE_PX:].any():
                    # unit runs off the left/right edge: its length, heading and speed come from a partial outline (a forklift
                    # leaving screen-002 read as 5.5 m/s); no new measurement this frame rather than a wrong one
                    self.skipped_side_cut = getattr(self, "skipped_side_cut", 0) + 1
                elif len(px) >= 6:
                    gr = [d["cam"].to_plane(px, d["contact_height_m"]) for d in self.draws]
                    if any(not np.isfinite(gd).all() for gd in gr):
                        gr = [gd[np.isfinite(gd).all(1)] for gd in gr]
                    if any(len(gd) < 6 for gd in gr):  # floor contact projects behind the camera in some draw
                        self.skipped_no_contact = getattr(self, "skipped_no_contact", 0) + 1
                    else:
                        ground = gr
            kf = self.kf.get(vid)
            coasting = False
            if ground is None:
                # v9: no usable outline this frame (detector miss, side cut, no floor contact). Keep the zone for a short gap
                # on the Kalman prediction, as a live system must; otherwise (or without smoothing) no zone, as before.
                if not SMOOTHING or kf is None or vid not in self.vlast or t - kf.t_update > COAST_MAX_S:
                    continue
                kf.predict(t)
                coasting, loads, cut = True, [], self.vlast[vid]["cut"]
                self.motion_stats["coasted"] += 1
                cent = kf.pos.copy()
            else:
                cent = np.array([gd.mean(0) for gd in ground])
                if SMOOTHING:
                    if kf is None or t - kf.t_update > KF_RESTART_S:
                        kf = self.kf[vid] = mf.FloorKF(VEH_ACCEL_SIGMA, VEH_MEAS_SIGMA, reset_after=VEH_RESET_AFTER, keep_velocity=True)
                        self.vshape.pop(vid, None)
                    res_ = kf.update(t, cent)
                    self.motion_stats[res_] += 1
                    if res_ == "reset" and vid in self.hist:
                        # v10: a re-anchor is a new reading of the same truck, not motion: move its past positions with it,
                        # so the speed fit does not read the shift as speed (screen-032 read 5.7 m/s for a 2.5 m/s truck)
                        self.hist[vid] = [(th, ch + kf.jump) for th, ch in self.hist[vid]]
                    tk = self.tkf.get(vid)
                    if tk is None or t - tk.t_update > KF_RESTART_S:
                        tk = self.tkf[vid] = mf.TurnKF(VEH_ACCEL_SIGMA, TURN_ALPHA_SIGMA, VEH_MEAS_SIGMA, reset_after=VEH_RESET_AFTER)
                        # start from the measured travel direction if there is one; a parked outline's axis is not used
                        tk.update(t, cent[0], theta0=self.vhead[vid][1] if vid in self.vhead else None)
                    else:
                        tk.update(t, cent[0])  # raw nominal outline centre: independent per-frame noise, as the filter assumes
            cent_f = kf.pos.copy() if SMOOTHING else cent
            hist = self.hist.setdefault(vid, [])
            hist.append((t, cent_f))
            hist[:] = [h for h in hist if t - h[0] <= WINDOW_S + 1e-6]
            # Load height and which end carries it, from the detected LOAD masks attached to this truck (nominal
            # camera, smoothed over the window). No load mask -> no blind zone and the fork end stays unknown.
            # (v1 used the truck mask's tallest columns, which found the overhead guard instead of the load.)
            cols = self._load_columns(loads) if loads else np.zeros((0, 3))
            load_h = float(np.percentile(cols[:, 2], 90)) if len(cols) >= 3 else float("nan")
            load_pt = self.nominal.to_plane(cols[:, :2]).mean(0) if len(cols) >= 3 else None
            lh = self.load.setdefault(v["id"], [])
            lh.append((t, load_h, load_pt))
            lh[:] = [h for h in lh if t - h[0] <= WINDOW_S + 1e-6]
            lhv = [h[1] for h in lh if np.isfinite(h[1])]
            load_h_s = float(np.median(lhv)) if lhv else float("nan")
            load_pts = [h[2] for h in lh if h[2] is not None]
            state = {"id": v["id"], "loadHeight_m": round(load_h_s, 2), "moving": bool(v.get("moving")), "draws": [], "omegaSE": 0.0, "omegaFit": 0.0, "cutByFrame": cut, "speedSE": 0.0,
                     "coasting": coasting}
            # Direction and turn rate are scale-free: estimate them once, from the nominal camera, so the draws
            # vary only what is uncertain (scale -> speed, sizes, timing) instead of re-fitting noisy heading.
            tt = np.array([h[0] for h in hist])
            c_nom = np.array([h[1][0] for h in hist])
            heading0, omega0, omega_se, v_nom_abs = None, 0.0, 0.0, 0.0
            if len(tt) >= 4 and tt[-1] - tt[0] >= SPEED_MIN_SPAN_S:
                v_nom = np.array([np.polyfit(tt, c_nom[:, i], 1)[0] for i in (0, 1)])
                v_nom_abs = float(np.linalg.norm(v_nom))
                if v_nom_abs >= MOVING_MPS:
                    heading0 = float(np.arctan2(v_nom[1], v_nom[0]))
                    lag = max(2, int(round(0.4 * self.fps)))
                    if len(tt) > lag + 2:
                        d_ = c_nom[lag:] - c_nom[:-lag]
                        ang = np.unwrap(np.arctan2(d_[:, 1], d_[:, 0]))
                        try:
                            coef, cov = np.polyfit(tt[lag:], ang, 1, cov=True)
                            omega0, omega_se = float(coef[0]), float(np.sqrt(max(cov[0, 0], 0.0)))
                        except (ValueError, np.linalg.LinAlgError):
                            omega0 = float(np.polyfit(tt[lag:], ang, 1)[0])
                    if not np.isfinite(omega0):
                        omega0, omega_se = 0.0, 0.0
                    if not np.isfinite(omega_se):
                        omega_se = 0.0
            turning_on = None
            if SMOOTHING:
                # v10: turn rate and its error from the turn-rate filter (TurnKF) instead of the 1 s line fit, whose error
                # treated overlapping samples as independent: on synthetic straight runs it said "turning" 73 % of the time
                tk = self.tkf.get(vid)
                if coasting and tk is not None:
                    tk.predict(t)
                omega0, omega_se = 0.0, 0.0
                if heading0 is not None and tk is not None and tk.updates >= 3:
                    omega0, omega_se = tk.omega, tk.omega_se
                # a turn counts only when it is measured (its error under half the turn), and it must hold for 3 frames
                measured_turn = abs(omega0) if (heading0 is not None and tk is not None and abs(tk.speed) >= HEADING_MIN_MPS
                                                and omega_se <= 0.5 * abs(omega0)) else 0.0
                turning_on = self.vturn.setdefault(vid, mf.Hysteresis(TURN_ON, TURN_OFF, TURN_FRAMES)).update(measured_turn)
                if not turning_on:
                    omega0 = 0.0  # v9: below a sustained turn the course is straight
                # footprint heading: follow the travel direction only above HEADING_MIN_MPS; the body axis turns no faster
                # than HEADING_RATE_MAX, but the travel direction may reverse at once (v10: the limit applied to the
                # direction, so a reversing truck's zone swung sideways for 3 s). A truck that starts moving takes its
                # first travel direction at once: the parked outline's long axis is not reliable enough to start from
                # (seeding from it pointed screen-002's zone about 20 deg off for 0.8 s)
                prev_h = self.vhead.get(vid)
                moving_dir = heading0 is not None and v_nom_abs >= HEADING_MIN_MPS
                was_moving, slow_since = self.vmoving.get(vid, (False, -1e9))
                restart = moving_dir and not was_moving and t - slow_since >= RESTART_SLOW_S
                self.vmoving[vid] = (moving_dir, slow_since if not moving_dir and not was_moving else (t if not moving_dir else -1e9))
                target = heading0 if moving_dir else (prev_h[1] if prev_h else None)
                if target is not None:
                    if restart:
                        self.vrev[vid] = 0  # v10: a truck that was stopped or slow may leave in any direction: take it at once
                    elif prev_h is not None:  # (clip-dev-002 backed up, stopped and drove off the other way; the zone
                                              #  pointed backwards for 0.3 s while the reversal rule waited)
                        base = prev_h[1]
                        step_ = mf.wrap(target - base)
                        if abs(step_) > np.pi / 2:  # travel reversed: same body axis, the other way, once it holds
                            # and the turn-rate filter, which sees the raw positions, agrees the truck now goes the other
                            # way (v10: the smoothed track sliding back after a re-anchor read as reversing at 2.4 m/s)
                            tk_dir = (tk.theta + (0.0 if tk.speed >= 0 else np.pi)
                                      if tk is not None and tk.updates >= 10 and abs(tk.speed) >= HEADING_MIN_MPS else None)
                            agrees = tk_dir is None or np.cos(target - tk_dir) > 0
                            self.vrev[vid] = self.vrev.get(vid, 0) + 1 if agrees else 0
                            if self.vrev[vid] >= REVERSE_FRAMES:
                                base, step_ = base + np.pi, mf.wrap_axis(step_)
                                self.vrev[vid] = 0
                            else:
                                step_ = 0.0
                        else:
                            self.vrev[vid] = 0
                        lim_ = HEADING_RATE_MAX * max(t - prev_h[0], 0.0)
                        target = mf.wrap(base + float(np.clip(step_, -lim_, lim_)))
                    self.vhead[vid] = (t, float(target))
            state["omegaSE"] = omega_se
            state["turning"] = turning_on
            state["restart"] = bool(SMOOTHING and target is not None and restart)  # took a new direction after a stop
            state["omegaFit"] = float(np.clip(omega0, -1.0, 1.0)) if heading0 is not None else 0.0
            cur_shape, rel_now = {}, {}
            for k, d in enumerate(self.draws):
                cc = np.array([h[1][k] for h in hist])
                accel, se_v, acc_sig = 0.0, 0.0, False
                if len(tt) >= 4 and tt[-1] - tt[0] >= SPEED_MIN_SPAN_S:
                    fits = [np.polyfit(tt, cc[:, i], 1, cov=True) for i in (0, 1)]  # mean velocity at the window's mid-time
                    vel = np.array([f_[0][0] for f_ in fits])
                    v_mid = float(np.linalg.norm(vel))
                    uv = vel / v_mid if v_mid > 1e-6 else np.array([1.0, 0.0])
                    se_v = float(np.sqrt(sum((np.sqrt(max(f_[1][0, 0], 0.0)) * uv[i]) ** 2 for i, f_ in enumerate(fits))))
                    if SMOOTHING and kf.updates >= 3:
                        se_v = max(se_v, kf.speed_se(uv))
                    if len(tt) >= 6 and v_mid > 1e-6:
                        q2 = [np.polyfit(tt - tt[-1], cc[:, i], 2, cov=True) for i in (0, 1)]
                        a_long = float(sum(2.0 * q_[0][0] * uv[i] for i, q_ in enumerate(q2)))
                        se_a = float(np.sqrt(sum((2.0 * np.sqrt(max(q_[1][0, 0], 0.0)) * uv[i]) ** 2 for i, q_ in enumerate(q2))))
                        acc_sig = abs(a_long) >= 2.0 * se_a  # only a clear acceleration is used (v7b)
                        accel = float(np.clip(a_long, *ACCEL_LIMITS)) if acc_sig else 0.0
                    v_now = v_mid + accel * (tt[-1] - tt.mean())
                    speed = max(0.0, max(v_mid, v_now) + se_v * d["speed_z"])  # v7c: a slowdown estimate never shrinks a zone; fit error (v7b)
                    if k == 0:
                        state["speedSE"] = se_v
                else:
                    vel = np.zeros(2)
                    speed = 0.0
                if heading0 is not None and speed >= MOVING_MPS:
                    heading, omega = (self.vhead[vid][1] if SMOOTHING and vid in self.vhead else heading0), omega0 + omega_se * d["omega_z"]
                    omega = 0.0 if abs(omega) < OMEGA_MIN else float(np.clip(omega, -1.0, 1.0))  # below a real turn: straight
                    if SMOOTHING:  # v9: the sticky decision alone says straight or turning; the draws then carry the turn's error
                        omega = float(np.clip(omega0 + omega_se * d["omega_z"], -1.0, 1.0)) if turning_on else 0.0
                elif SMOOTHING and vid in self.vhead:
                    heading, omega = self.vhead[vid][1], 0.0  # v9: slow or stopped: keep the last travel direction
                    speed = speed if heading0 is not None else 0.0
                elif ground is not None:
                    (_, _), (w_, h_), ang_ = cv2.minAreaRect(ground[k].astype(np.float32))
                    heading, omega = float(np.radians(ang_ if w_ >= h_ else ang_ + 90)), 0.0
                    if SMOOTHING:  # never seen moving: keep the outline's long axis steady (closest of the two directions to last frame's)
                        prev = self.vstatic.get((vid, k))
                        if prev is not None and np.cos(heading - prev) < 0:
                            heading += np.pi
                        if prev is not None and abs(mf.wrap_axis(heading - prev)) > np.radians(30):
                            n_ = self.vswap[(vid, k)] = self.vswap.get((vid, k), 0) + 1
                            if n_ < AXIS_SWAP_FRAMES:
                                heading = prev  # v10: one frame's swapped long axis is not a turn of the parked truck
                            else:
                                self.vswap[(vid, k)] = 0
                        else:
                            self.vswap[(vid, k)] = 0
                        self.vstatic[(vid, k)] = heading
                    speed = speed if heading0 is not None else 0.0
                else:
                    heading, omega = self.vstatic.get((vid, k), 0.0), 0.0
                    speed = speed if heading0 is not None else 0.0
                if not SMOOTHING:
                    foot = fr.rect_footprint(ground[k], heading, self.camera_xy, d["vehicle_length_m"], d["vehicle_width_m"])
                elif ground is not None:
                    foot = self._footprint(vid, k, ground[k] + (cent_f[k] - cent[k]), heading, d, t, cur_shape)
                else:
                    foot = self.vlast[vid]["rel"][k] + cent_f[k]
                rel_now[k] = foot - cent_f[k]
                u = np.array([np.cos(heading), np.sin(heading)])
                c0 = foot.mean(0)
                # load end relative to travel direction
                load_first = None
                if load_pts:
                    load_first = bool((np.mean(load_pts, 0) - c0) @ u > 0) if speed >= MOVING_MPS else None
                # ---- stopping sweep: constant v and turn rate for t_react, then braking. A counterbalance truck
                #      steers with its rear wheels and pivots about its front (drive) axle near the load, so the
                #      rotation centre lies on the line through that axle, perpendicular to the body axis.
                t_r, a_ = d["reaction_s"], d["decel_mps2"]
                t_stop = t_r + speed / a_
                d_stop = speed * t_r + speed ** 2 / (2 * a_)
                ts = np.linspace(0, t_stop, 9) if speed >= MOVING_MPS else np.array([0.0])
                s_of_t = np.where(ts <= t_r, speed * ts, speed * t_r + speed * (ts - t_r) - 0.5 * a_ * (ts - t_r) ** 2)
                if load_pts:
                    b = np.mean(load_pts, 0) - c0
                    b = b / max(np.linalg.norm(b), 1e-6)
                else:  # load end unknown: either end may carry the forks (half the draws each)
                    b = u if k % 2 == 0 else -u
                along_b = (foot - c0) @ b
                axle_off = along_b.max() - d["load_face_to_drive_axle_m"]
                axle = c0 + b * axle_off
                left = np.array([-u[1], u[0]])
                turning = speed >= MOVING_MPS and abs(omega) > 0.05
                icr = axle + left * (speed / omega) if turning else None
                def pose(poly, s_):
                    if turning:
                        return icr + (poly - icr) @ rot(omega * s_ / speed).T
                    return poly + u * s_
                front_poly = clip_halfplane(foot, -b, -axle_off - c0 @ b)   # load side of the axle
                rear_poly = clip_halfplane(foot, b, axle_off + c0 @ b)      # counterweight side
                stop_mask = np.zeros(self.shape, np.uint8); front_mask = np.zeros(self.shape, np.uint8); rear_mask = np.zeros(self.shape, np.uint8)
                prev = None
                for s_ in s_of_t:
                    body = pose(foot, s_)
                    fill(stop_mask, self.g(body), 1)
                    if prev is not None:  # fill between consecutive poses
                        fill(stop_mask, self.g(cv2.convexHull(np.vstack([prev, body]).astype(np.float32)).reshape(-1, 2)), 1)
                    prev = body
                    if len(front_poly) >= 3:
                        fill(front_mask, self.g(pose(front_poly, s_)), 1)
                    if len(rear_poly) >= 3:
                        fill(rear_mask, self.g(pose(rear_poly, s_)), 1)
                cur = np.zeros(self.shape, np.uint8); fill(cur, self.g(foot), 1)
                tail_mask = np.zeros(self.shape, np.uint8)
                if turning:  # counterweight-end sweep beyond the truck's side line on the outside of the turn
                    n_out = -left if omega > 0 else left
                    edge = float(((foot - c0) @ n_out).max())
                    big = 60.0
                    outer = np.array([c0 + n_out * edge - b * big, c0 + n_out * edge + b * big,
                                      c0 + n_out * (edge + big) + b * big, c0 + n_out * (edge + big) - b * big])
                    half = np.zeros(self.shape, np.uint8); fill(half, self.g(outer), 1)
                    tail_mask = (rear_mask & half & (1 - cur)).astype(np.uint8)
                # ---- blind zone: load taller than the eye -> shadow wedge in front of the load
                blind_mask = np.zeros(self.shape, np.uint8)
                if np.isfinite(load_h_s) and load_h_s > d["eye_height_m"] and load_pts:
                    to_load = np.mean(load_pts, 0) - c0
                    fwd = to_load / max(np.linalg.norm(to_load), 1e-6)
                    side = np.array([-fwd[1], fwd[0]])
                    proj = (foot - c0) @ fwd
                    face = c0 + fwd * proj.max()
                    half_w = 0.5 * d["vehicle_width_m"]
                    eye = face - fwd * d["eye_to_load_face_m"]
                    corners = [face + side * half_w, face - side * half_w]
                    reach = BLIND_REACH_M + d["eye_to_load_face_m"]
                    far = [eye + (c - eye) / np.linalg.norm(c - eye) * reach / max(1e-6, ((c - eye) / np.linalg.norm(c - eye)) @ fwd) for c in corners]
                    fill(blind_mask, self.g(np.array([corners[0], far[0], far[1], corners[1]])), 1)
                # ---- v7 warning: keep course (speed, clipped acceleration, turn) for WARN_HORIZON_S; poses every 0.1 s
                warn_mask = np.zeros(self.shape, np.uint8)
                w_bodies = []
                if speed >= MOVING_MPS:
                    tw = np.arange(0.0, WARN_HORIZON_S + 1e-6, WARN_STEP_S)
                    v_t = np.clip(speed + max(accel, 0.0) * tw, 0.0, V_MAX)  # v7c: only a speed-up changes the course; capped at V_MAX
                    sw = np.concatenate([[0.0], np.cumsum(0.5 * (v_t[1:] + v_t[:-1]) * WARN_STEP_S)])
                    w_bodies = [pose(foot, s_) for s_ in sw]
                    prev = None
                    for i_, body in enumerate(w_bodies):
                        if i_ % 2 == 0 or i_ == len(w_bodies) - 1:  # raster every 0.2 s; distances below use every 0.1 s
                            fill(warn_mask, self.g(body), 1)
                            if prev is not None:
                                fill(warn_mask, self.g(cv2.convexHull(np.vstack([prev, body]).astype(np.float32)).reshape(-1, 2)), 1)
                            prev = body
                    r_ = int(round(WARN_CLEAR_M / GRID_M))
                    warn_mask = cv2.dilate(warn_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1, 2 * r_ + 1)))
                blind_active = blind_mask if load_first else np.zeros_like(blind_mask)
                blind_trailing = blind_mask if not load_first else np.zeros_like(blind_mask)
                for name, m in (("stop", stop_mask), ("tail", tail_mask), ("blind", blind_active), ("warn", warn_mask), ("blindTrailing", blind_trailing)):
                    counts[name] += m if self.warp[k] is None else cv2.warpPerspective(m, self.warp[k], (self.shape[1], self.shape[0]), flags=cv2.INTER_NEAREST)
                state["draws"].append({"speed": speed, "accel": accel, "accelSig": acc_sig, "omega": omega, "heading": heading, "dStop": d_stop if speed >= MOVING_MPS else 0.0, "loadFirst": load_first,
                                       "footprint": foot, "axleAssumedEnd": not bool(load_pts)})
                for pid, (pts, seated) in person_pts.items():
                    q = pts[k]
                    if not np.isfinite(q).all():
                        continue
                    bv = results[pid]["byVehicle"].setdefault(v["id"], {"inZone": {k_: 0 for k_ in COLOURS}, "clearance": [], "ttc": [], "pass": [],
                                                                        "moving": bool(v.get("moving"))})
                    gp = self.g(q)[0]
                    inside = 0 <= gp[1] < self.shape[0] and 0 <= gp[0] < self.shape[1]
                    for name, m in (("stop", stop_mask), ("tail", tail_mask), ("blind", blind_mask)):
                        if inside and m[gp[1], gp[0]]:
                            if name != "blind" or load_first:
                                bv["inZone"][name] += 1
                    cl_now = fr.point_rect_distance(q, foot)
                    bv["clearance"].append(cl_now)
                    ttc = np.inf  # v7: first time (s) the unit comes within WARN_CLEAR_M if it keeps its course
                    pass_d, pass_t = cl_now, 0.0  # v7b: closest it will come in the next 3 s, and when
                    if w_bodies and cl_now <= speed * WARN_HORIZON_S + 0.5 * max(accel, 0.0) * WARN_HORIZON_S ** 2 + WARN_CLEAR_M + 4.0:
                        for i_, body in enumerate(w_bodies):
                            dd = fr.point_rect_distance(q, body)
                            if dd < pass_d:
                                pass_d, pass_t = dd, i_ * WARN_STEP_S
                            if dd < WARN_CLEAR_M and not np.isfinite(ttc):
                                ttc = i_ * WARN_STEP_S
                    bv["ttc"].append(ttc)
                    bv["pass"].append((pass_d, pass_t))
            if SMOOTHING:
                if cur_shape:
                    sh = self.vshape.setdefault(vid, [])
                    sh.append((t, cur_shape))
                    sh[:] = [h for h in sh if t - h[0] <= WINDOW_S + 1e-6]
                self.vlast[vid] = {"rel": rel_now, "cut": cut}
            vstates.append(state)
        if SMOOTHING:
            self._prune(t)
        for pid, r in results.items():
            # governing vehicle: the nearest MOVING one (the warning and danger rules only apply to moving forklifts),
            # else the nearest; the others are listed with their median clearance
            bvs = r.pop("byVehicle")
            med = {vid: (float(np.median(b["clearance"])) if b["clearance"] else float("inf")) for vid, b in bvs.items()}
            pool = [vid for vid, b in bvs.items() if b["moving"]] or list(bvs)
            gov = min(pool, key=lambda vid: med[vid]) if pool else None
            b = bvs[gov] if gov is not None else {"inZone": {k: 0 for k in COLOURS}, "clearance": [], "ttc": [], "pass": []}
            if gov is not None:
                r["vehicle"] = gov
            r["inZone"], r["clearance"], r["ttc"], r["pass"] = b["inZone"], b["clearance"], b["ttc"], b["pass"]
            r["otherVehicles"] = {vid: round(m_, 2) for vid, m_ in med.items() if vid != gov and np.isfinite(m_)}
            r["pInZone"] = {k: n / D for k, n in r["inZone"].items() if k in ("stop", "tail", "blind")}
            tt_ = np.array(r["ttc"]) if r["ttc"] else np.array([np.inf])
            fin = tt_[np.isfinite(tt_)]
            r["pWarn"] = float(len(fin) / D)
            r["ttc_s"] = {"median": float(np.median(fin)), "p5": float(np.percentile(fin, 5)), "p95": float(np.percentile(fin, 95))} if len(fin) else None
            if r["pass"]:
                pd_ = np.array([x[0] for x in r["pass"]]); pt_ = np.array([x[1] for x in r["pass"]])
                r["pass_m"] = {"median": float(np.median(pd_)), "p5": float(np.percentile(pd_, 5)), "p95": float(np.percentile(pd_, 95))}
                r["passT_s"] = float(np.median(pt_))
            else:
                r["pass_m"], r["passT_s"] = None, None
            if r["clearance"]:
                c = np.array(r["clearance"])
                r["clearance_m"] = {"median": float(np.median(c)), "p5": float(np.percentile(c, 5)), "p95": float(np.percentile(c, 95))}
            else:  # no forklift in view: no clearance (avoids percentile over infinities)
                r["clearance_m"] = {"median": float("inf"), "p5": float("inf"), "p95": float("inf")}
        prob = {k: c_.astype(np.float32) / D for k, c_ in counts.items()}
        summary = []
        for s in vstates:
            sp = np.array([x["speed"] for x in s["draws"]]); ds = np.array([x["dStop"] for x in s["draws"]])
            lf = [x["loadFirst"] for x in s["draws"] if x["loadFirst"] is not None]
            ac = np.array([x["accel"] for x in s["draws"]])
            summary.append({"id": s["id"], "speed_mps": [float(np.percentile(sp, q)) for q in (5, 50, 95)],
                            "accel_mps2": [float(np.percentile(ac, q)) for q in (5, 50, 95)],
                            "speedSE_mps": round(s["speedSE"], 3), "accelUsed": float(np.mean([x["accelSig"] for x in s["draws"]])),
                            "heading_rad": round(float(s["draws"][0]["heading"]), 4), "speedNominal_mps": round(float(s["draws"][0]["speed"]), 3),
                            "stoppingDistance_m": [float(np.percentile(ds, q)) for q in (5, 50, 95)],
                            "turnRate_radps": float(np.median([x["omega"] for x in s["draws"]])), "turnRateSE_radps": round(s["omegaSE"], 3), "turnRateFit_radps": round(s["omegaFit"], 3), "cutByFrame": s["cutByFrame"], "coasting": s["coasting"], "turning": s.get("turning"), "directionRestart": s.get("restart", False),
                            "pTurning": float(np.mean([x["omega"] != 0.0 for x in s["draws"]])), "loadHeight_m": s["loadHeight_m"],
                            "pLoadFirst": (float(np.mean(lf)) if lf else None),
                            "pLoadBlocksView": float(np.mean([s["loadHeight_m"] > d["eye_height_m"] for d in self.draws])) if np.isfinite(s["loadHeight_m"]) else 0.0,
                            "footprint": s["draws"][0]["footprint"]})
        return {"prob": prob, "persons": results, "vehicles": summary}

    # ------------------------------------------------------------------ app export (v7d)
    def zone_polygons(self, out: dict, level: float = 0.5) -> dict:
        """'Likely' outline of every zone in floor metres, simplified to one raster cell, for the dashboard map."""
        res = {}
        for name in COLOURS:
            m = (out["prob"][name] >= level).astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys = []
            for c in cs:
                if len(c) < 3 or cv2.contourArea(c) <= 4:
                    continue
                c = cv2.approxPolyDP(c, 1.0, True)
                if len(c) >= 3:
                    polys.append(np.round(self.g_inv(c.reshape(-1, 2)), 2).tolist())
            if polys:
                res[name] = polys
        return res

    # ------------------------------------------------------------------ drawing
    def contours(self, prob: np.ndarray, level: float):
        m = (prob >= level).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [self.g_inv(c.reshape(-1, 2)) for c in cs if len(c) >= 3 and cv2.contourArea(c) > 4]

    def draw_on_camera(self, canvas: np.ndarray, out: dict, scale: float = 1.0):
        wl = canvas.copy()  # v7: 3 s warning zone, light fill + thin outline, underneath everything
        for poly in self.contours(out["prob"]["warn"], 0.5):
            uv = self.nominal.project(np.column_stack([poly, np.zeros(len(poly))])) * scale
            if np.isfinite(uv).all():
                cvsafe.fill_poly(wl, [safe_pts(uv)], COLOURS["warn"])
                cv2.polylines(canvas, [safe_pts(uv)], True, COLOURS["warn"], 2, cv2.LINE_AA)
        canvas[:] = cv2.addWeighted(wl, 0.14, canvas, 0.86, 0)
        for poly in self.contours(out["prob"]["blindTrailing"], 0.5):  # v7: inactive blind zone, outline only + label
            uv = self.nominal.project(np.column_stack([poly, np.zeros(len(poly))])) * scale
            if np.isfinite(uv).all():
                cv2.polylines(canvas, [safe_pts(uv)], True, COLOURS["blindTrailing"], 1, cv2.LINE_AA)
                c_ = uv.mean(0).astype(int)
                cv2.putText(canvas, "blind zone inactive: load trailing", (int(c_[0]) - 150, int(c_[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 215, 190), 2, cv2.LINE_AA)
        layer = canvas.copy()
        for name in ("blind", "tail", "stop"):
            for poly in self.contours(out["prob"][name], 0.5):
                uv = self.nominal.project(np.column_stack([poly, np.zeros(len(poly))])) * scale
                if np.isfinite(uv).all():
                    cvsafe.fill_poly(layer, [safe_pts(uv)], COLOURS[name])
        canvas[:] = cv2.addWeighted(layer, 0.28, canvas, 0.72, 0)
        for name in ("blind", "tail", "stop"):
            for level, thick in ((0.05, 1), (0.5, 3)):
                for poly in self.contours(out["prob"][name], level):
                    uv = self.nominal.project(np.column_stack([poly, np.zeros(len(poly))])) * scale
                    if np.isfinite(uv).all():
                        cv2.polylines(canvas, [safe_pts(uv)], True, COLOURS[name], thick, cv2.LINE_AA)

    def inset(self, out: dict, persons_alert: dict, w: int = 440, h: int = 330) -> np.ndarray:
        """Top-down floor map: zones (likely filled, possible outline), footprints, people, 1 m grid."""
        pts = [np.zeros((1, 2))] + [v["footprint"] for v in out["vehicles"]]
        lo, hi = np.vstack(pts).min(0) - 4.0, np.vstack(pts).max(0) + 4.0
        s = min((w - 20) / (hi[0] - lo[0]), (h - 40) / (hi[1] - lo[1]))
        def m2p(xy):
            xy = np.atleast_2d(xy)
            return safe_pts(np.column_stack([10 + (xy[:, 0] - lo[0]) * s, h - 10 - (xy[:, 1] - lo[1]) * s]))
        img = np.full((h, w, 3), 245, np.uint8)
        for gx in np.arange(np.floor(lo[0]), np.ceil(hi[0]) + 1):
            cv2.line(img, tuple(m2p([gx, lo[1]])[0]), tuple(m2p([gx, hi[1]])[0]), (215, 215, 215), 1)
        for gy in np.arange(np.floor(lo[1]), np.ceil(hi[1]) + 1):
            cv2.line(img, tuple(m2p([lo[0], gy])[0]), tuple(m2p([hi[0], gy])[0]), (215, 215, 215), 1)
        wl = img.copy()
        for poly in self.contours(out["prob"]["warn"], 0.5):
            cvsafe.fill_poly(wl, [m2p(poly)], COLOURS["warn"])
        img = cv2.addWeighted(wl, 0.25, img, 0.75, 0)
        for poly in self.contours(out["prob"]["blindTrailing"], 0.5):
            cv2.polylines(img, [m2p(poly)], True, COLOURS["blindTrailing"], 1, cv2.LINE_AA)
        layer = img.copy()
        for name in ("blind", "tail", "stop"):
            for poly in self.contours(out["prob"][name], 0.5):
                cvsafe.fill_poly(layer, [m2p(poly)], COLOURS[name])
        img = cv2.addWeighted(layer, 0.35, img, 0.65, 0)
        for name in ("warn", "blind", "tail", "stop"):
            for poly in self.contours(out["prob"][name], 0.05):
                cv2.polylines(img, [m2p(poly)], True, COLOURS[name], 1, cv2.LINE_AA)
        for v in out["vehicles"]:
            cv2.polylines(img, [m2p(v["footprint"])], True, (0, 120, 230), 2, cv2.LINE_AA)
        for pid, r in persons_alert.items():
            col = (40, 40, 235) if r.get("alert") else (0, 150, 255) if r.get("warn") else (255, 150, 40)
            cv2.circle(img, tuple(m2p(r["xy"])[0]), 6, col, -1, cv2.LINE_AA)
            cv2.putText(img, pid, tuple(m2p(r["xy"])[0] + [8, 4]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.circle(img, tuple(m2p([0, 0])[0]), 6, (20, 20, 20), -1)
        cv2.line(img, (w - 70, h - 14), (w - 70 + int(s), h - 14), (20, 20, 20), 3)
        cv2.putText(img, "1 m", (w - 70, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (w - 1, 26), (40, 40, 40), -1)
        cv2.putText(img, "floor (m): amber 3 s warning, red stop, magenta tail, blue blind", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (60, 60, 60), 2)
        return img
