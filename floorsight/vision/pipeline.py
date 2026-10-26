"""Real vision pipeline: markers -> calibration, per-camera foreground, floor fusion, tracks, tools.

    from floorsight.vision import RealVision, VideoFeed
    feed = VideoFeed("data/sim/demo_001")
    vision = RealVision(feed.site, evidence_store=LocalEvidenceStore("data/evidence"))
    for packets in feed:
        world = vision.process(packets)

Only pixels and the public site plan are used; no simulator truth.
"""

from __future__ import annotations

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from floorsight.cvsafe import fill_convex

from floorsight.contracts import (
    XY,
    CameraHealth,
    CameraInfo,
    CameraStatus,
    CoverageMap,
    Evidence,
    SitePlan,
    WorldState,
)
from floorsight.interfaces import EvidenceStore, FramePacket
from floorsight.site import FORKLIFT_LENGTH_M, FORKLIFT_WIDTH_M, point_in_polygon
from floorsight.vision.calib import Calibration, MarkerDetector, fit_calibration, marker_floor_corners, merge_observations
from floorsight.vision.coverage import placeable_grid, sees
from floorsight.vision.detect import Blob, CameraDetector
from floorsight.vision.tracker import FORKLIFT_FOOTPRINT_M, FloorDet, Tracker, cluster

CALIB_FRAMES = 20
CHECK_EVERY_S = 2.0
RECHECK_S = 0.5
MOVED_ERROR_M = 0.15
MOVED_CHECKS = 3
RECAL_ERROR_M = 0.08
RECAL_MIN_MARKERS = 2  # each marker gives 4 corners; no camera in the demo site sees 4 markers
CELL_M = 0.5
SPOT_HEAL = True  # heal static regions where a brand-new track started (background it left behind)
COVER_HEIGHT_M = 1.0


def _lowest(prof: np.ndarray) -> np.ndarray:
    """Keep the lowest point per 4-pixel column bin."""
    best: dict[int, np.ndarray] = {}
    for p in prof:
        k = int(p[0] // 4)
        if k not in best or p[1] > best[k][1]:
            best[k] = p
    return np.array([best[k] for k in sorted(best)])


def _box_outline_table(length: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    """For a box seen at each heading: (width across the line of sight, centre offset behind its near outline)."""
    widths, shifts = [], []
    for th in np.radians(np.arange(0.0, 90.5, 1.0)):
        c, s = math.cos(th), math.sin(th)
        corners = np.array([(sx * length / 2, sy * width / 2) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        lat = corners[:, 0] * c - corners[:, 1] * s
        rad = corners[:, 0] * s + corners[:, 1] * c
        ls = np.linspace(lat.min(), lat.max(), 41)
        near = []
        for l in ls:
            rs = []
            for i in range(4):
                j = (i + 1) % 4
                l0, l1 = lat[i], lat[j]
                if min(l0, l1) <= l <= max(l0, l1) and abs(l1 - l0) > 1e-9:
                    rs.append(rad[i] + (l - l0) / (l1 - l0) * (rad[j] - rad[i]))
            near.append(min(rs))
        widths.append(lat.max() - lat.min())
        shifts.append(-float(np.mean(near)))
    return np.array(widths), np.array(shifts)


_VEHICLE_W, _VEHICLE_SHIFT = _box_outline_table(FORKLIFT_LENGTH_M, FORKLIFT_WIDTH_M)
PERSON_SHIFT_M = 0.08  # measured: lowest outline of a ~0.45 m wide body sits ~8 cm in front of its centre


def radial_shift(footprint_m: float) -> float:
    """Visible lower outline -> object centre, pushed away from the camera."""
    if footprint_m < FORKLIFT_FOOTPRINT_M:
        return PERSON_SHIFT_M
    ok = np.abs(_VEHICLE_W - footprint_m) < 0.12
    if not ok.any():
        ok = np.abs(_VEHICLE_W - footprint_m) == np.abs(_VEHICLE_W - footprint_m).min()
    return float(np.mean(_VEHICLE_SHIFT[ok]))


class _Camera:
    def __init__(self, info: CameraInfo, scale: float):
        self.info = info
        self.det = CameraDetector(info.width, info.height, info.fps, scale)
        self.calib: Calibration | None = None
        self.status = CameraStatus.ok
        self.frames = 0
        self.calib_hist: list[dict[int, np.ndarray]] = []
        self.last_image: np.ndarray | None = None
        self.last_t = 0.0
        self.fps = info.fps
        self.markers_visible = 0
        self.last_error = float("nan")
        self.next_check_t = 0.0
        self.bad_streak = 0
        self.streak_start_t = 0.0
        self.moved_t: float | None = None
        self.suspect_until = -1.0
        self.global_change_t = -1e9
        self.check_obs: list[tuple[float, dict[int, np.ndarray]]] = []
        self.rack_mask: np.ndarray | None = None
        self.plate: np.ndarray | None = None  # empty-floor setup image, full resolution
        self.plate_R: np.ndarray | None = None  # camera rotation the plate was taken with
        self.visible: np.ndarray | None = None
        self.blobs: list[Blob] = []

    @property
    def usable(self) -> bool:
        return self.calib is not None and self.status == CameraStatus.ok


class RealVision:
    """Implements floorsight.interfaces.VisionPipeline."""

    def __init__(self, site: SitePlan, evidence_store: EvidenceStore | None = None, process_scale: float = 0.5, workers: int | None = None):
        self.site = site
        self.store = evidence_store
        self.floor_corners = marker_floor_corners(site)
        self.markers = MarkerDetector(site)
        self.cams = {c.camera_id: _Camera(c, process_scale) for c in site.cameras}
        self.cols = int(round(site.width_m / CELL_M))
        self.rows = int(round(site.depth_m / CELL_M))
        self._counts = np.zeros((self.rows, self.cols), int)
        self._cov_key: tuple | None = None
        self.coverage_version = 0
        self.tracker = Tracker(
            self._covered, (site.width_m, site.depth_m), self._presence, self._nearest_blind, self._free_point, self._restricted
        )
        self.pool = ThreadPoolExecutor(max_workers=workers or len(self.cams))
        self.lock = threading.RLock()
        self.t = 0.0
        self._heal: list[tuple[np.ndarray, float]] = []
        self.last_dets: list[FloorDet] = []
        self._plates_checked = False

    # ---- setup plates ------------------------------------------------------ #

    def set_background_plates(self, plates: dict[str, np.ndarray]) -> None:
        """Empty-floor images recorded at installation (same pose, nobody on the floor), one per camera.

        Call before the first process(). Without plates the first frame is the background, so anyone
        standing still from the start stays invisible until they move. If never called, plates are
        loaded from $FLOORSIGHT_SETUP_DIR/setup_empty_<camera_id>.png when that variable is set.
        """
        with self.lock:
            self._plates_checked = True
            for cid, img in plates.items():
                cam = self.cams.get(cid)
                if cam is None or img is None:
                    continue
                if img.shape[1] != cam.info.width or img.shape[0] != cam.info.height:
                    img = cv2.resize(img, (cam.info.width, cam.info.height), interpolation=cv2.INTER_AREA)
                cam.plate = img
                cam.plate_R = cam.calib.R.copy() if cam.calib is not None else None
                cam.det.set_plate(img)

    def _load_plates_from_env(self) -> None:
        self._plates_checked = True
        root = os.environ.get("FLOORSIGHT_SETUP_DIR")
        if not root:
            return
        plates = {}
        for cid in self.cams:
            path = os.path.join(root, f"setup_empty_{cid}.png")
            if os.path.exists(path):
                plates[cid] = cv2.imread(path, cv2.IMREAD_COLOR)
        if plates:
            self.set_background_plates(plates)

    @property
    def has_plates(self) -> bool:
        return any(c.plate is not None for c in self.cams.values())

    # ---- VisionPipeline ---------------------------------------------------- #

    def process(self, packets: dict[str, FramePacket]) -> WorldState:
        t0 = time.perf_counter()
        if not self._plates_checked:
            self._load_plates_from_env()
        with self.lock:
            t = max((p.meta.t for p in packets.values()), default=self.t)
            if t < self.t:
                raise ValueError("Vision timestamps must be monotonic; start a new session for replay")
            self.t = t
            packets = {cid: p for cid, p in packets.items() if cid in self.cams and (self.cams[cid].frames == 0 or p.meta.t > self.cams[cid].last_t) and t - p.meta.t <= 0.5}
            for cid, cam in self.cams.items():
                if cid not in packets and t - cam.last_t > 0.5:
                    cam.status = CameraStatus.offline
                elif cid in packets and cam.status == CameraStatus.offline:
                    cam.status = CameraStatus.ok if cam.calib is not None else CameraStatus.needs_human
                    cam.next_check_t = 0.0
            jobs = {cid: self.pool.submit(self._camera_step, self.cams[cid], p) for cid, p in packets.items() if cid in self.cams}
            for f in jobs.values():
                f.result()
            self._update_coverage()
            dets = []
            for cam in self.cams.values():
                if cam.usable and t >= cam.suspect_until and cam.bad_streak == 0 and cam.info.camera_id in packets:
                    dets += self._to_floor(cam, t)
            dets = self._drop_ghosts(dets, t) if SPOT_HEAL and not self.has_plates else dets
            self.last_dets = dets
            moving = cluster([d for d in dets if d.kind == "moving"], 0.9, 1.6)
            static = cluster([d for d in dets if d.kind == "static"], 1.2)
            self.tracker.trusted_background = self.has_plates
            tracks = self.tracker.step(t, moving, static)
            self._heal += self.tracker.new_confirmed
            cams = [self._health(c) for c in self.cams.values()]
            latency = (time.perf_counter() - t0) * 1000.0
            return WorldState(t=t, tracks=tracks, cameras=cams, coverage_version=self.coverage_version, pipeline_latency_ms=round(latency, 2))

    # ---- per camera -------------------------------------------------------- #

    def _camera_step(self, cam: _Camera, packet: FramePacket) -> None:
        image, t = packet.image, packet.meta.t
        if image.shape[1] != cam.info.width or image.shape[0] != cam.info.height:
            image = cv2.resize(image, (cam.info.width, cam.info.height))
        if cam.frames > 0 and t > cam.last_t:
            cam.fps = 0.8 * cam.fps + 0.2 * (1.0 / (t - cam.last_t))
        cam.last_t = t
        cam.last_image = image
        cam.frames += 1
        if cam.frames <= CALIB_FRAMES:
            self._calibration_frame(cam, image, t)
        cam.blobs = cam.det.process(image, t)
        if cam.det.global_change:
            cam.global_change_t = t
            cam.suspect_until = t + 3.0
            cam.next_check_t = min(cam.next_check_t, t)
            cam.det.reset(drop_plate=True)  # the view changed; the plate no longer lines up
        if cam.calib is not None and cam.frames > CALIB_FRAMES and t >= cam.next_check_t:
            self._drift_check(cam, image, t)

    def _calibration_frame(self, cam: _Camera, image: np.ndarray, t: float) -> None:
        n = cam.frames
        cam.calib_hist.append(self.markers.detect(image, fine=n in (1, CALIB_FRAMES // 2, CALIB_FRAMES)))
        if n in (1, 5, CALIB_FRAMES // 2, CALIB_FRAMES):
            obs = merge_observations(cam.calib_hist)
            calib = fit_calibration(obs, self.floor_corners, cam.info)
            if calib is not None and math.isfinite(calib.error_m):
                self._set_calibration(cam, calib)
                cam.markers_visible = len(obs)
                cam.status = CameraStatus.ok
            elif n == CALIB_FRAMES and cam.calib is None:
                cam.status = CameraStatus.needs_human
        if n == CALIB_FRAMES and cam.plate is not None and cam.calib is not None:
            cam.plate_R = cam.calib.R.copy()
        if n == CALIB_FRAMES:
            cam.next_check_t = t + CHECK_EVERY_S * (0.25 + 0.25 * list(self.cams).index(cam.info.camera_id))

    def _set_calibration(self, cam: _Camera, calib: Calibration) -> None:
        cam.calib = calib
        cam.last_error = calib.error_m
        cam.rack_mask = self._rack_mask(cam.info, calib)
        cam.visible = None  # recomputed by _update_coverage

    def _drift_check(self, cam: _Camera, image: np.ndarray, t: float) -> None:
        obs = self.markers.detect(image)
        cam.markers_visible = len(obs)
        cam.check_obs = [(ts, o) for ts, o in cam.check_obs if t - ts < 30.0] + [(t, obs)]
        if obs:
            err = cam.calib.floor_error(obs, self.floor_corners)
        elif t - cam.global_change_t < 10.0:
            err = float("inf")
        else:
            cam.next_check_t = t + CHECK_EVERY_S
            return
        cam.last_error = err
        if not math.isfinite(err) or err > MOVED_ERROR_M:
            if cam.bad_streak == 0:
                cam.streak_start_t = t
            cam.bad_streak += 1
            cam.next_check_t = t + RECHECK_S
            if cam.bad_streak >= MOVED_CHECKS and cam.status == CameraStatus.ok:
                cam.status = CameraStatus.moved
                cam.moved_t = t
        else:
            cam.bad_streak = 0
            cam.next_check_t = t + CHECK_EVERY_S

    def _health(self, cam: _Camera) -> CameraHealth:
        err = cam.last_error if math.isfinite(cam.last_error) else None
        return CameraHealth(
            camera_id=cam.info.camera_id,
            status=cam.status,
            fps=round(cam.fps, 2),
            last_frame_t=cam.last_t,
            markers_visible=cam.markers_visible,
            calib_error_m=round(err, 4) if err is not None else None,
        )

    # ---- geometry ---------------------------------------------------------- #

    def _rack_mask(self, info: CameraInfo, calib: Calibration) -> np.ndarray:
        mask = np.zeros((info.height, info.width), np.uint8)
        for ob in self.site.obstacles:
            base = [(x, y, 0.0) for x, y in ob.polygon]
            top = [(x, y, ob.height_m) for x, y in ob.polygon]
            faces = [base, top] + [[base[i], base[(i + 1) % len(base)], top[(i + 1) % len(base)], top[i]] for i in range(len(base))]
            for f in faces:
                uv, z = calib.project(np.array(f))
                if (z <= 0.1).any():
                    continue
                fill_convex(mask, uv, 1)  # clipped first: COOL's polygon fills crash on out-of-image vertices
        return mask

    def _to_floor(self, cam: _Camera, t: float) -> list[FloorDet]:
        calib, info = cam.calib, cam.info
        out = []
        cpos = calib.position[:2]
        for b in self._merge_stacked(cam.blobs):
            x, y, w, h = b.bbox
            fx, fy = b.foot_px
            if fy >= info.height - 4:
                continue  # feet below the image: the lowest visible row is not where it stands
            side_cut = x <= 2 or x + w >= info.width - 2  # ground contact visible, but part of the body is not
            row = int(min(info.height - 1, fy + 4))
            if cam.rack_mask is not None and cam.rack_mask[row, max(0, x) : min(info.width, x + w)].mean() > 0.4:
                continue  # feet hidden behind a rack
            prof = b.bottom_px if b.bottom_px is not None and len(b.bottom_px) else np.array([[fx, fy]])
            prof = prof[prof[:, 1] >= fy - 0.25 * h - 2]
            if cam.rack_mask is not None:
                rr = np.clip(prof[:, 1].astype(int) + 4, 0, info.height - 1)
                cc = np.clip(prof[:, 0].astype(int), 0, info.width - 1)
                ok = cam.rack_mask[rr, cc] == 0
                prof = prof[ok] if ok.sum() >= max(2, 0.5 * len(prof)) else prof
            pts = calib.to_floor(prof)
            foot = pts.mean(axis=0)
            ray = foot - cpos
            ray /= max(np.linalg.norm(ray), 1e-6)
            # Width across the line of sight: ~0.5 m for a person, 1.1-2.3 m for a forklift.
            footprint = float(np.ptp(pts @ np.array([-ray[1], ray[0]]))) if len(pts) > 1 else 0.0
            if b.kind == "static":  # a standing person's feet are narrower than the body
                edge = calib.to_floor(np.array([(x, fy), (x + w, fy)]))
                footprint = max(footprint, float(np.linalg.norm(edge[1] - edge[0])))
            hgt = self._height_m(calib, calib.to_floor(np.array([[fx, fy]]))[0], y)
            if b.kind == "moving":
                if not (0.8 <= hgt <= 3.5) or footprint < 0.15 or footprint > 4.0:
                    continue
            elif footprint < 0.2 or footprint > 4.0:
                continue
            person_shaped = b.kind == "static" and 1.2 <= hgt <= 2.3 and hgt >= 1.6 * footprint
            xy = foot + ray * (PERSON_SHIFT_M if person_shaped else radial_shift(footprint))
            if not (-0.2 <= xy[0] <= self.site.width_m + 0.2 and -0.2 <= xy[1] <= self.site.depth_m + 0.2):
                continue
            if any(point_in_polygon(float(xy[0]), float(xy[1]), ob.polygon) for ob in self.site.obstacles):
                continue
            sigma = 0.08 + 2.0 * calib.metres_per_pixel((fx, fy))
            if side_cut:
                sigma *= 3.0
            if footprint >= FORKLIFT_FOOTPRINT_M:
                # Partial body or hidden ground contact cannot support a precise vehicle centre.
                contact_visible = sees(self.site, calib, info, (float(xy[0]), float(xy[1]), 0.05))
                if side_cut or not contact_visible:
                    sigma = max(sigma, 0.6)
            out.append(FloorDet(info.camera_id, xy, sigma, footprint, b.kind, b.stationary_s, b.score, b.bbox, b.foot_px, hgt))
        return out

    @staticmethod
    def _height_m(calib: Calibration, foot: np.ndarray, top_v: float) -> float:
        """Height of a vertical object at `foot` whose top appears at image row top_v."""
        uv, _ = calib.project(np.array([[foot[0], foot[1], 0.0], [foot[0], foot[1], 2.0]]))
        v0, v2 = uv[0, 1], uv[1, 1]
        if abs(v0 - v2) < 1e-6:
            return 0.0
        return 2.0 * (v0 - top_v) / (v0 - v2)

    @staticmethod
    def _merge_stacked(blobs: list[Blob]) -> list[Blob]:
        """Join pieces of one body split horizontally (e.g. head above torso)."""
        out: list[Blob] = []
        for kind in ("moving", "static"):
            items = sorted((b for b in blobs if b.kind == kind), key=lambda b: b.bbox[1])
            merged = True
            while merged:
                merged = False
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        a, b = items[i], items[j]
                        ax, ay, aw, ah = a.bbox
                        bx, by, bw, bh = b.bbox
                        overlap = min(ax + aw, bx + bw) - max(ax, bx)
                        gap = max(ay, by) - min(ay + ah, by + bh)
                        if overlap > 0.5 * min(aw, bw) and gap < max(8, 0.15 * max(ah, bh)):
                            x0, y0 = min(ax, bx), min(ay, by)
                            x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
                            low = a if ay + ah >= by + bh else b
                            prof = [p for p in (a.bottom_px, b.bottom_px) if p is not None]
                            items[i] = Blob(
                                kind,
                                (x0, y0, x1 - x0, y1 - y0),
                                low.foot_px,
                                a.area_px + b.area_px,
                                max(a.stationary_s, b.stationary_s),
                                max(a.score, b.score),
                                _lowest(np.concatenate(prof)) if prof else None,
                            )
                            items.pop(j)
                            merged = True
                            break
                    if merged:
                        break
            out += items
        return out

    def _box_px(self, cam: _Camera, xy: XY, half_m: float, top_m: float) -> tuple[int, int, int, int] | None:
        x, y = xy
        pts = [(x + dx, y + dy, z) for dx in (-half_m, half_m) for dy in (-half_m, half_m) for z in (0.0, top_m)]
        uv, z = cam.calib.project(np.array(pts))
        if (z <= 0.1).any():
            return None
        x0, y0 = np.floor(uv.min(0)).astype(int)
        x1, y1 = np.ceil(uv.max(0)).astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(cam.info.width, x1), min(cam.info.height, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return int(x0), int(y0), int(x1 - x0), int(y1 - y0)

    # ---- ghosts ------------------------------------------------------------ #

    def _drop_ghosts(self, dets: list[FloorDet], t: float) -> list[FloorDet]:
        """A static region where a mover started from, with nobody there now, is the background it left: heal it."""
        self._heal = [(xy, tb) for xy, tb in self._heal if t - tb < 60.0]
        if not self._heal:
            return dets
        live = [tr.xy for tr in self.tracker.tracks if tr.kind == "mover" and tr.confirmed and t - tr.last_t < 1.0]
        spots = [(xy, tb) for xy, tb in self._heal if not any(np.linalg.norm(xy - m) < 1.0 for m in live)]
        out = []
        for d in dets:
            # A ghost only appears once its owner has left: its static age must start after the birth.
            onset = t - d.stationary_s
            if d.kind == "static" and any(np.linalg.norm(d.xy - xy) < 1.2 and onset >= tb - 1.0 for xy, tb in spots):
                det = self.cams[d.camera_id].det
                e_cur, e_bg = det.object_edges(d.bbox)
                if e_cur < 1.5 * e_bg:  # the frame does not hold a clearly new outline: background left behind
                    det.heal(d.bbox)
                    continue
            out.append(d)
        return out

    # ---- coverage ---------------------------------------------------------- #

    def _update_coverage(self) -> None:
        key = tuple((cid, c.status.value, id(c.calib)) for cid, c in self.cams.items())
        if key == self._cov_key:
            return
        counts = np.zeros((self.rows, self.cols), int)
        for cam in self.cams.values():
            if not cam.usable:
                continue
            if cam.visible is None:
                cam.visible = placeable_grid(self.site, cam.calib, cam.info, CELL_M)
            counts += cam.visible
        self._cov_key = key
        if self.coverage_version == 0 or not np.array_equal(counts, self._counts):
            self._counts = counts
            self.coverage_version += 1

    def _presence(self, xy: np.ndarray) -> tuple[float, list[str]]:
        """Best share of a person-sized column at xy that differs from the long-term background, and which cameras."""
        fracs, cams = [], []
        c, r = int(xy[0] // CELL_M), int(xy[1] // CELL_M)
        for cid, cam in self.cams.items():
            if not cam.usable or cam.visible is None or self.t < cam.suspect_until or cam.bad_streak:
                continue
            if not (0 <= r < self.rows and 0 <= c < self.cols) or not cam.visible[r, c]:
                continue
            box = self._box_px(cam, (float(xy[0]), float(xy[1])), 0.3, 1.7)
            if box is None:
                continue
            f = cam.det.fraction("moving", box)
            fracs.append(f)
            if f >= 0.15:
                cams.append(cid)
        if not fracs:
            return 0.0, []
        # Another body in front of the spot can fill one camera's box; every camera (up to two) must agree.
        need = min(2, len(fracs))
        return (float(sorted(fracs)[-need]), sorted(cams))

    def _free_point(self, xy: np.ndarray) -> tuple[float, float]:
        """The point itself, or (inside a rack) the nearest point just outside that rack."""
        x, y = float(xy[0]), float(xy[1])
        for ob in self.site.obstacles:
            if not point_in_polygon(x, y, ob.polygon):
                continue
            best, best_d = (x, y), 1e9
            n = len(ob.polygon)
            for i in range(n):
                a, b = np.asarray(ob.polygon[i], float), np.asarray(ob.polygon[(i + 1) % n], float)
                ab = b - a
                k = float(np.clip(np.dot(np.array([x, y]) - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0))
                q = a + k * ab
                d = math.hypot(q[0] - x, q[1] - y)
                if d < best_d:
                    out = q - np.array([x, y])
                    out = out / max(np.linalg.norm(out), 1e-9)
                    best, best_d = (float(q[0] + 0.15 * out[0]), float(q[1] + 0.15 * out[1])), d
            return best
        return x, y

    def _restricted(self, xy: np.ndarray) -> bool:
        return any(z.kind.value == "restricted" and point_in_polygon(float(xy[0]), float(xy[1]), z.polygon) for z in self.site.zones)

    def _nearest_blind(self, xy: np.ndarray, radius_m: float) -> np.ndarray | None:
        """Centre of the closest floor cell no camera sees, within radius_m, or None."""
        rr, cc = np.nonzero(self._counts == 0)
        if not len(rr):
            return None
        centres = np.column_stack([(cc + 0.5) * CELL_M, (rr + 0.5) * CELL_M])
        d = np.linalg.norm(centres - np.asarray(xy)[None, :], axis=1)
        i = int(np.argmin(d))
        return centres[i] if d[i] <= radius_m else None

    def _covered(self, x: float, y: float) -> int:
        c, r = int(x // CELL_M), int(y // CELL_M)
        if not (0 <= c < self.cols and 0 <= r < self.rows):
            return 0
        return int(self._counts[r, c])

    def coverage(self) -> CoverageMap:
        with self.lock:
            self._update_coverage()
            return CoverageMap(cell_m=CELL_M, cols=self.cols, rows=self.rows, counts=self._counts.ravel().tolist(), version=self.coverage_version)

    # ---- VisionTools ------------------------------------------------------- #

    def cameras_covering(self, xy: XY) -> list[str]:
        with self.lock:
            return [cid for cid, c in self.cams.items() if c.usable and sees(self.site, c.calib, c.info, (xy[0], xy[1], COVER_HEIGHT_M))]

    def second_view(self, xy: XY, exclude: list[str], kind: str) -> list[Evidence]:
        with self.lock:
            out = []
            z = 0.3 if kind == "static" else COVER_HEIGHT_M
            for cid, cam in self.cams.items():
                if cid in exclude or not cam.usable or cam.last_image is None:
                    continue
                if not sees(self.site, cam.calib, cam.info, (xy[0], xy[1], z)):
                    continue
                box = self._box_px(cam, xy, 0.6 if kind == "static" else 0.4, 1.0 if kind == "static" else 1.8)
                if box is None:
                    continue
                frac = cam.det.fraction(kind, box)
                expected = 0.25 if kind == "static" else 0.2
                score = round(min(1.0, frac / expected), 3)
                supports = frac >= 0.4 * expected
                what = "a static object" if kind == "static" else "someone"
                out.append(
                    Evidence(
                        kind="second_view",
                        camera_id=cid,
                        t=self.t,
                        supports=supports,
                        score=score,
                        image_ref=self._store(cam, box, f"second_view_{cid}"),
                        note=f"{cid} {'sees' if supports else 'does not see'} {what} at ({xy[0]:.1f}, {xy[1]:.1f}): "
                        f"{frac:.0%} of the region differs from its background",
                    )
                )
            return out

    def sharper_crop(self, camera_id: str, xy: XY, radius_m: float = 1.0) -> Evidence:
        with self.lock:
            cam = self.cams.get(camera_id)
            box = None
            if cam is not None and cam.usable and cam.last_image is not None and self.t - cam.last_t <= 0.5:
                box = self._box_px(cam, xy, radius_m, 1.8)
            if box is None:
                return Evidence(kind="crop", camera_id=camera_id, t=self.t, supports=None, score=None, note=f"{camera_id} cannot frame ({xy[0]:.1f}, {xy[1]:.1f})")
            ref = self._store(cam, box, f"crop_{camera_id}")
            return Evidence(
                kind="crop",
                camera_id=camera_id,
                t=self.t,
                supports=None,
                score=None,
                image_ref=ref,
                note=f"{camera_id} full-resolution crop {box[2]}x{box[3]} px around ({xy[0]:.1f}, {xy[1]:.1f})",
            )

    def recalibrate(self, camera_id: str) -> dict:
        with self.lock:
            cam = self.cams.get(camera_id)
            if cam is None or cam.last_image is None:
                return {"ok": False, "calib_error_m": None, "markers_used": 0}
            since = cam.streak_start_t if cam.status != CameraStatus.ok else self.t - 4.0
            history = [o for ts, o in cam.check_obs if ts >= since] + [self.markers.detect(cam.last_image, fine=True)]
            obs = merge_observations(history[-6:])
            calib = fit_calibration(obs, self.floor_corners, cam.info, min_markers=RECAL_MIN_MARKERS)
            if calib is None or not math.isfinite(calib.error_m) or calib.error_m >= RECAL_ERROR_M:
                cam.status = CameraStatus.needs_human
                err = None if calib is None or not math.isfinite(calib.error_m) else calib.error_m
                return {"ok": False, "calib_error_m": round(err, 4) if err is not None else None, "markers_used": len(obs)}
            self._set_calibration(cam, calib)
            cam.status = CameraStatus.ok
            cam.bad_streak = 0
            cam.markers_visible = len(obs)
            cam.next_check_t = self.t + CHECK_EVERY_S
            self._restore_plate(cam)
            self._update_coverage()
            return {"ok": True, "calib_error_m": round(calib.error_m, 4), "markers_used": calib.markers_used}

    def _restore_plate(self, cam: _Camera) -> None:
        """A knock rotates the camera about its own centre, so the setup plate maps exactly onto the new view."""
        if cam.plate is None or cam.plate_R is None or cam.last_image is None:
            cam.det.reset()
            return
        K = cam.calib.K
        H = K @ cam.calib.R @ cam.plate_R.T @ np.linalg.inv(K)
        size = (cam.info.width, cam.info.height)
        H = self._refine_alignment(cam.plate, cam.last_image, H)
        warped = cv2.warpPerspective(cam.plate, H, size, flags=cv2.INTER_LINEAR)
        valid = cv2.warpPerspective(np.full(cam.plate.shape[:2], 255, np.uint8), H, size, flags=cv2.INTER_NEAREST)
        # Today's view is the background except inside blob-sized differences (people, vehicles present now),
        # which the plate fills in; thin residual misalignment along edges is opened away.
        frame = cam.last_image
        diff = cv2.absdiff(cv2.GaussianBlur(frame, (5, 5), 1.0), cv2.GaussianBlur(warped, (5, 5), 1.0)).max(axis=2)
        blobs = cv2.morphologyEx((diff > 28).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        blobs = cv2.dilate(blobs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        use = (blobs > 0) & (valid > 0)
        composite = frame.copy()
        composite[use] = warped[use]
        cam.det.set_plate(composite)

    @staticmethod
    def _refine_alignment(plate: np.ndarray, frame: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Polish the pose-derived plate->frame warp with ECC on blurred half-resolution gray images."""
        s = 0.5
        S = np.diag([s, s, 1.0])
        a = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(plate, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), (5, 5), 1.5)
        b = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), (5, 5), 1.5)
        # ECC estimates W with b(x) ~ a(W x), i.e. W maps frame -> plate: start from the inverse warp.
        W = (S @ np.linalg.inv(H) @ np.linalg.inv(S)).astype(np.float32)
        W /= W[2, 2]
        try:
            _, W = cv2.findTransformECC(a, b, W, cv2.MOTION_HOMOGRAPHY, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 5)
        except cv2.error:
            return H
        Hr = np.linalg.inv(S) @ np.linalg.inv(W.astype(np.float64)) @ S
        # Only accept a polish, not a different answer.
        corners = np.array([[0, 0], [plate.shape[1], 0], [0, plate.shape[0]], [plate.shape[1], plate.shape[0]]], np.float64).reshape(-1, 1, 2)
        moved = np.abs(cv2.perspectiveTransform(corners, Hr) - cv2.perspectiveTransform(corners, H)).max()
        return Hr if moved < 15.0 else H

    def _store(self, cam: _Camera, box: tuple[int, int, int, int], label: str) -> str | None:
        if self.store is None:
            return None
        x, y, w, h = box
        pad = 12
        crop = cam.last_image[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad]
        return self.store.put(crop.copy(), label)

    def close(self) -> None:
        self.pool.shutdown(wait=False)
