"""Real factory clips inside the Floorsight app (lead 20 Oct 17:05Z).

The perception stage (``scripts/real_clip_perception.py`` on GCP) already turned each real clip into
privacy-filtered frames and per-frame floor measurements in metres (people, forklift footprints, danger zones,
3 s warnings). ``scripts/export_app_replay.py`` packs one or more clips into a replay folder:

    site.json                 SitePlan of the real camera (floor_stage=True), floor coordinates shifted to >= 0
    playlist.json             {"camera_id", "calibration": {K, R, h}, "offset_m": [x0, y0], "clips": [...]}
    <clip>-filtered.mp4       privacy-filtered frames, no overlays (the only pixels this app ever sees)
    <clip>.frames.jsonl       one compact record per frame (see export_app_replay.py)

``RealClipSource`` plays the filtered videos one after another as one session; ``ReplayVision`` turns each frame's
record into a ``WorldState`` (tracks + ``hazards``) and gives the agent three second-look tools:
``sharper_crop`` (full-resolution close-up), ``short_replay`` (was it sustained, or a one-frame dip / tracking jump?)
and ``redetect`` (detectors run again on the crop; optional, configured with FLOORSIGHT_REDETECT_CMD).
No raw video is read here: frames are filtered upstream and fail closed there (black, withheld frames).

Admission (opt-in, FLOORSIGHT_REPLAY_ADMISSION=enforce): every clip must carry a reviewed floorsight.real-clip.v1
manifest whose file hashes check out (``admit_replay``), or the folder is refused before anything is played. Off by
default because export_app_replay.py does not write manifests yet.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shlex
import subprocess
import shutil
import tempfile
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from floorsight.contracts import (
    CameraHealth,
    CameraStatus,
    CoverageMap,
    Evidence,
    FloorAlert,
    FloorHazards,
    FrameMeta,
    SitePlan,
    Track,
    TrackClass,
    TrackState,
    WorldState,
)
from floorsight.interfaces import FramePacket
from floorsight.real_clip_contract import assess
from floorsight.real_clip_files import _bound_file, _digest, verify_file_bindings
from floorsight.site import load_site

log = logging.getLogger("floorsight.realclip")

REPLAY_BUFFER_S = 2.0  # filtered frames kept for the short-replay strip (downscaled)
REPLAY_THUMB_W = 640
MIN_RUN_FRAMES = 3  # same 3-frame rule as the perception stage
HEADING_JUMP_DEG = 35.0
SPEED_JUMP_RATIO = 1.6
GAP_FRAMES = 3  # vehicle not measured for >= this many frames just before onset = tracking gap
ADMISSION_ENV = "FLOORSIGHT_REPLAY_ADMISSION"  # unset / 0 / false / off: no admission check; anything else enforces it


class ReplayNotAdmitted(RuntimeError):
    """The replay folder failed admission (metadata, privacy review or file hashes); nothing from it is played."""

    def __init__(self, replay_dir, reasons):
        self.reasons = tuple(reasons)
        super().__init__(f"replay folder {replay_dir} refused: {', '.join(self.reasons)}")


def _clip_manifest(root: Path, reference) -> dict:
    """The clip's admission manifest, bound to the playlist by path + sha256; ValueError(reason) otherwise."""
    try:
        path = _bound_file(root, reference)
        if _digest(path) != reference["sha256"]:
            raise ValueError("hash_mismatch")
    except FileNotFoundError:
        raise ValueError("missing_file") from None
    except OSError:
        raise ValueError("unreadable_file") from None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("not_json") from None
    if not isinstance(manifest, dict):
        raise ValueError("not_json")
    return manifest


def admit_replay(replay_dir: str | os.PathLike, playlist: dict) -> tuple[str, ...]:
    """Reasons to refuse this replay folder ("<clip>:<reason>"); empty means every clip is admitted.

    Each playlist clip needs "manifest": {"path", "sha256"} naming a floorsight.real-clip.v1 manifest in the folder that
    passes the source/privacy admission (real_clip_contract.assess), names this clip's video as its privacy-filtered file,
    binds the same source clip the floor records came from (original.sha256 == clipSha256), and whose filtered video and
    privacy-review evidence match their sha256 on disk (real_clip_files). The raw original never sits in a replay folder,
    so it is bound by hash only. Unvalidated metric geometry is logged, not refused.
    """
    root = Path(replay_dir).resolve(strict=True)
    reasons: list[str] = []
    for clip in playlist.get("clips", []):
        name = clip.get("name", "?")
        if clip.get("manifest") is None:
            reasons.append(f"{name}:metadata_missing")
            continue
        try:
            manifest = _clip_manifest(root, clip["manifest"])
        except ValueError as error:
            reasons.append(f"{name}:manifest:{error}")
            continue
        admission = assess(manifest)
        reasons += [f"{name}:{r}" for r in admission.reasons if r != "metric_geometry_not_validated"]
        if not admission.metric_events:
            log.warning("replay clip %s: floor geometry is not validated for metric events", name)
        privacy = manifest.get("privacy") if isinstance(manifest.get("privacy"), dict) else {}
        filtered = privacy.get("filtered") if isinstance(privacy.get("filtered"), dict) else {}
        if filtered.get("path") != clip.get("video"):
            reasons.append(f"{name}:video_is_not_the_reviewed_file")
        original = manifest.get("original") if isinstance(manifest.get("original"), dict) else {}
        if not clip.get("clipSha256") or original.get("sha256") != clip.get("clipSha256"):
            reasons.append(f"{name}:records_not_from_the_reviewed_source")
        reasons += [f"{name}:{r}" for r in verify_file_bindings(root, manifest, skip=("original",)).reasons]
    return tuple(reasons)


def load_playlist(replay_dir: str | os.PathLike, admission: bool | None = None) -> dict:
    """playlist.json. With admission on (None: read FLOORSIGHT_REPLAY_ADMISSION) a folder failing admit_replay is refused."""
    with open(Path(replay_dir) / "playlist.json", encoding="utf-8") as f:
        playlist = json.load(f)
    if admission is None:
        admission = os.environ.get(ADMISSION_ENV, "").strip().lower() not in ("", "0", "false", "off", "no")
    if admission:
        reasons = admit_replay(replay_dir, playlist)
        if reasons:
            raise ReplayNotAdmitted(replay_dir, reasons)
    return playlist


class FloorCamera:
    """Pinhole camera over the floor plane z = 0 (perception-stage calibration), in SITE coordinates."""

    def __init__(self, K, R, h: float, offset_m=(0.0, 0.0), size=(1920, 1080)):
        self.K = np.asarray(K, float)
        self.R = np.asarray(R, float)
        self.h = float(h)
        self.offset = np.asarray(offset_m, float)
        self.size = tuple(size)

    def project(self, xyz_site) -> np.ndarray:
        P = np.atleast_2d(np.asarray(xyz_site, float)).copy()
        if P.shape[1] == 2:
            P = np.column_stack([P, np.zeros(len(P))])
        P[:, :2] += self.offset  # site -> calibration floor frame
        xc = (P - np.array([0.0, 0.0, self.h])) @ self.R.T
        uv = (self.K @ xc.T).T
        out = uv[:, :2] / uv[:, 2:3]
        out[xc[:, 2] <= 1e-6] = np.nan
        return out

    def to_site(self, px) -> np.ndarray:
        px = np.atleast_2d(np.asarray(px, float))
        rays = (np.linalg.inv(self.K) @ np.column_stack([px, np.ones(len(px))]).T).T @ self.R  # camera -> world dirs
        s = -self.h / rays[:, 2]
        pts = np.column_stack([s * rays[:, 0], s * rays[:, 1]])
        pts[s <= 0] = np.nan
        return pts - self.offset

    def sees(self, xy_site) -> bool:
        uv = self.project([xy_site[0], xy_site[1], 0.0])[0]
        return bool(np.isfinite(uv).all() and 0 <= uv[0] < self.size[0] and 0 <= uv[1] < self.size[1])


class RealClipSource:
    """Plays every clip's privacy-filtered video in playlist order as one session (one camera)."""

    realtime = False
    has_video = True
    prefiltered = True  # filtered upstream by the perception stage; the runtime must not blank or re-blur them

    def __init__(self, replay_dir: str, site: SitePlan | None = None):
        self.dir = Path(replay_dir)
        self.playlist = load_playlist(self.dir)
        self.site = site or load_site(str(self.dir / "site.json"))
        self.camera_id = self.playlist["camera_id"]
        self.fps = float(self.playlist["clips"][0]["fps"]) if self.playlist["clips"] else 25.0
        self._cap = None

    def __iter__(self) -> Iterator[dict[str, FramePacket]]:
        for clip in self.playlist["clips"]:
            path = self.dir / clip["video"]
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise OSError(f"cannot decode {path}")
            self._cap = cap
            try:
                local = 0
                while True:
                    ok, img = cap.read()
                    if not ok or img is None:
                        break
                    meta = FrameMeta(
                        camera_id=self.camera_id,
                        frame_index=clip["offset_frames"] + local,
                        t=round(clip["offset_t"] + local / clip["fps"], 3),
                        width=img.shape[1],
                        height=img.shape[0],
                        faces_blurred=0,
                        blur_applied=True,
                    )
                    yield {self.camera_id: FramePacket(meta, img)}
                    local += 1
            finally:
                cap.release()
                self._cap = None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def _alert_from(a: dict) -> FloorAlert:
    return FloorAlert(
        person_id=a["person"], vehicle_id=a.get("vehicle"), level=a["level"], confirmed=bool(a.get("confirmed")), streak=int(a.get("streak", 0)),
        clearance_m=a.get("clearance"), pass_m=a.get("pass"), pass_t_s=a.get("passT"), ttc_s=a.get("ttc"), p_warn=float(a.get("pWarn") or 0.0),
        p_path=float(a.get("pPath") or 0.0), reasons=list(a.get("reasons") or []), vehicle_moving=bool(a.get("moving")), cut_by_frame=bool(a.get("cut")),
    )


class ReplayVision:
    """VisionPipeline over a replay folder: per-frame floor records -> WorldState, plus second-look tools."""

    def __init__(self, site: SitePlan, evidence_store, replay_dir: str):
        self.site = site
        self.evidence = evidence_store
        self.dir = Path(replay_dir)
        pl = load_playlist(self.dir)
        self.camera_id = pl["camera_id"]
        cal = pl["calibration"]
        self.cam = FloorCamera(cal["K"], cal["R"], cal["h"], pl.get("offset_m", (0.0, 0.0)), tuple(pl.get("frame_size", (1920, 1080))))
        self.records: dict[int, dict] = {}
        self.clip_of: dict[int, str] = {}
        for clip in pl["clips"]:
            with open(self.dir / clip["records"], encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        g = clip["offset_frames"] + int(r["f"])
                        self.records[g] = r
                        self.clip_of[g] = clip["name"]
        self.fps = float(pl["clips"][0]["fps"]) if pl["clips"] else 25.0
        self.t = 0.0
        self.last_image: np.ndarray | None = None
        self.last_record: dict | None = None
        self.history: deque = deque(maxlen=int(4.0 * self.fps))  # (t, record) for short_replay
        self.frames: deque = deque(maxlen=int(REPLAY_BUFFER_S * self.fps))  # (t, small filtered image)
        self.first_seen: dict[str, float] = {}
        self.prev_xy: dict[str, tuple[float, float, float]] = {}
        self._fps_walls: deque = deque(maxlen=32)
        self._cov: CoverageMap | None = None
        self.last_idx = -1
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="redetect")
        self._jobs: dict[str, Future] = {}

    # ------------------------------------------------------------------ world
    def process(self, packets: dict[str, FramePacket]) -> WorldState:
        wall0 = time.perf_counter()
        pk = packets.get(self.camera_id) or next(iter(packets.values()))
        idx, t = pk.meta.frame_index, pk.meta.t
        self.t = t
        self.last_idx = idx
        self.last_image = pk.image
        h, w = pk.image.shape[:2]
        self.frames.append((t, cv2.resize(pk.image, (REPLAY_THUMB_W, int(h * REPLAY_THUMB_W / w)), interpolation=cv2.INTER_AREA)))
        rec = self.records.get(idx) or {"persons": [], "vehicles": [], "zones": {}, "alerts": []}
        self.last_record = rec
        self.history.append((t, rec))
        tracks: list[Track] = []
        for p in rec.get("persons", []):
            if p.get("xy") is None:
                continue
            tracks.append(self._track(p["id"], TrackClass.person, p["xy"], t, max(0.05, float(p.get("sigma") or 0.1)), TrackState.seen))
        footprints, notes = {}, {}
        for v in rec.get("vehicles", []):
            fp = v.get("footprint")
            if not fp:
                continue
            c = np.mean(np.asarray(fp, float), axis=0)
            sp = (v.get("speed") or [0, 0, 0])[1]
            hd = v.get("heading") or 0.0
            state = TrackState.estimated if v.get("cut") else TrackState.seen
            tr = self._track(v["id"], TrackClass.forklift, (float(c[0]), float(c[1])), t, 0.25 if not v.get("cut") else 0.6, state)
            tr.vx, tr.vy = round(sp * math.cos(hd), 3), round(sp * math.sin(hd), 3)
            tracks.append(tr)
            footprints[v["id"]] = [tuple(map(float, q)) for q in fp]
            notes[v["id"]] = list(v.get("notes") or [])
        hz = FloorHazards(
            source=self.clip_of.get(idx, ""),
            zones={k: [[tuple(map(float, q)) for q in poly] for poly in polys] for k, polys in (rec.get("zones") or {}).items()},
            footprints=footprints,
            notes=notes,
            alerts=[_alert_from(a) for a in rec.get("alerts", [])],
        )
        now = time.perf_counter()
        self._fps_walls.append(now)
        fps = (len(self._fps_walls) - 1) / (self._fps_walls[-1] - self._fps_walls[0]) if len(self._fps_walls) > 1 and self._fps_walls[-1] > self._fps_walls[0] else 0.0
        cams = [CameraHealth(camera_id=self.camera_id, status=CameraStatus.ok, fps=round(fps, 2), last_frame_t=t, markers_visible=0, calib_error_m=None)]
        return WorldState(t=t, tracks=tracks, cameras=cams, coverage_version=1, pipeline_latency_ms=round((time.perf_counter() - wall0) * 1000.0, 2), hazards=hz)

    def _track(self, tid, cls, xy, t, sigma, state) -> Track:
        self.first_seen.setdefault(tid, t)
        vx = vy = 0.0
        prev = self.prev_xy.get(tid)
        if prev is not None and t > prev[2]:
            dt = t - prev[2]
            vx, vy = (xy[0] - prev[0]) / dt, (xy[1] - prev[1]) / dt
            if math.hypot(vx, vy) > 8.0:  # a jump, not a speed
                vx = vy = 0.0
        self.prev_xy[tid] = (float(xy[0]), float(xy[1]), t)
        return Track(track_id=tid, cls=cls, x=round(float(xy[0]), 3), y=round(float(xy[1]), 3), vx=round(vx, 3), vy=round(vy, 3), sigma_m=round(sigma, 3),
                     camera_ids=[self.camera_id], first_seen_t=self.first_seen[tid], last_seen_t=t, state=state)

    def camera_views(self) -> dict:
        """Floor-map wedge of the real camera from its calibration (yaw of the optical axis, horizontal FOV)."""
        axis = self.cam.R[2]  # optical axis in floor coordinates
        yaw = math.degrees(math.atan2(axis[1], axis[0]))
        hfov = math.degrees(2 * math.atan(self.cam.size[0] / (2 * self.cam.K[0, 0])))
        bottom = self.cam.to_site([[self.cam.size[0] / 2, self.cam.size[1] - 1]])[0]
        cam_xy = -self.cam.offset
        near = float(np.hypot(*(bottom - cam_xy))) if np.isfinite(bottom).all() else 2.0
        return {self.camera_id: {"x": round(float(cam_xy[0]), 2), "y": round(float(cam_xy[1]), 2), "height_m": round(self.cam.h, 2), "yaw_deg": round(yaw, 2),
                                 "hfov_deg": round(hfov, 2), "near_m": round(near, 2), "far_m": 30.0, "source": "calibration (vanishing points + two size references)"}}

    # ------------------------------------------------------------------ tools
    def cameras_covering(self, xy) -> list[str]:
        return [self.camera_id] if self.cam.sees(xy) else []

    def second_view(self, xy, exclude, kind) -> list[Evidence]:
        return []  # one real camera: there is no second view, and saying so is the honest answer

    def recalibrate(self, camera_id: str) -> dict:
        return {"ok": False, "calib_error_m": None, "markers_used": 0}

    def coverage(self) -> CoverageMap:
        cell = 0.5
        cols, rows = int(math.ceil(self.site.width_m / cell)), int(math.ceil(self.site.depth_m / cell))
        if self._cov is None:
            counts = []
            for r in range(rows):
                for c in range(cols):
                    counts.append(1 if self.cam.sees(((c + 0.5) * cell, (r + 0.5) * cell)) else 0)
            self._cov = CoverageMap(cell_m=cell, cols=cols, rows=rows, counts=counts, version=1)
        return self._cov

    def _boxes_for(self, track_ids: list[str]) -> tuple[dict | None, dict | None]:
        rec = self.last_record or {}
        person = next((p for p in rec.get("persons", []) if p["id"] in track_ids), None)
        vehicle = next((v for v in rec.get("vehicles", []) if v["id"] in track_ids), None)
        return person, vehicle

    def _crop_region(self, person: dict | None, vehicle: dict | None, xy, radius_m: float) -> tuple[int, int, int, int]:
        pts = []
        if person and person.get("box"):
            x1, y1, x2, y2 = person["box"]
            pts += [(x1, y1), (x2, y2)]
        if vehicle and vehicle.get("box"):
            x1, y1, x2, y2 = vehicle["box"]
            pts += [(x1, y1), (x2, y2)]
        if not pts and xy is not None:
            ring = [(xy[0] + radius_m * math.cos(a), xy[1] + radius_m * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, 9) for z in (0.0, 1.8)]
            uv = self.cam.project(ring)
            pts = [tuple(p) for p in uv if np.isfinite(p).all()]
        if not pts:
            return 0, 0, self.cam.size[0], self.cam.size[1]
        a = np.asarray(pts, float)
        x1, y1 = a.min(0) - 40
        x2, y2 = a.max(0) + 40
        W, H = self.cam.size
        return int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2))

    def sharper_crop(self, camera_id: str, xy, radius_m: float = 1.0) -> Evidence:
        if self.last_image is None:
            return Evidence(kind="crop", camera_id=camera_id, t=self.t, supports=None, note="no frame yet")
        rec = self.last_record or {}
        people = [p for p in rec.get("persons", []) if p.get("xy") is not None]
        person = min(people, key=lambda p: math.dist(p["xy"], xy)) if people and xy is not None else None
        alert = next((a for a in rec.get("alerts", []) if person and a["person"] == person["id"]), None)
        vehicle = next((v for v in rec.get("vehicles", []) if alert and v["id"] == alert.get("vehicle")), None)
        x1, y1, x2, y2 = self._crop_region(person, vehicle, xy, radius_m)
        crop = self.last_image[y1:y2, x1:x2]
        if crop.size == 0:
            return Evidence(kind="crop", camera_id=camera_id, t=self.t, supports=None, note="crop outside the frame")
        scale = max(1.0, min(2.0, 720.0 / max(1, crop.shape[0])))
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else crop.copy()
        ref = self.evidence.put(big, f"crop-{camera_id}-t{self.t:.2f}")
        return Evidence(kind="crop", camera_id=camera_id, t=self.t, supports=None, image_ref=ref,
                        note=f"full-resolution close-up ({crop.shape[1]}x{crop.shape[0]} px of the privacy-filtered frame, x{scale:.1f})")

    def short_replay(self, track_ids: list[str], seconds: float = 1.5) -> Evidence:
        """Look back over the last `seconds` of floor records for this person/forklift pair (causal: no future frames)."""
        pid = next((i for i in track_ids if ":P" in i or i.startswith("P")), track_ids[0] if track_ids else None)
        vid = next((i for i in track_ids if ":F" in i or i.startswith("F")), None)
        rows = [(t, r) for t, r in self.history if self.t - t <= seconds + 1e-6]
        series = []
        for t, r in rows:
            a = next((a for a in r.get("alerts", []) if a["person"] == pid and (vid is None or a.get("vehicle") == vid)), None)
            v = next((v for v in r.get("vehicles", []) if v["id"] == vid), None) if vid else None
            series.append({"t": t, "alert": a, "vehicle": v})
        if not series:
            return Evidence(kind="replay", t=self.t, supports=None, note="no recent frames")
        # current run of alert frames
        run = 0
        for s in reversed(series):
            if s["alert"] is not None:
                run += 1
            else:
                break
        onset = len(series) - run
        pre = series[max(0, onset - int(0.5 * self.fps)):onset]
        gap = 0
        for s in reversed(pre):
            if s["vehicle"] is None or s["vehicle"].get("est"):
                gap += 1
            else:
                break
        measured = [s["vehicle"] for s in series[max(0, onset - int(0.3 * self.fps)):] if s["vehicle"] is not None and not s["vehicle"].get("est")]
        jumps, restarts = [], []
        for a, b in zip(measured, measured[1:]):
            dh = abs((math.degrees((b.get("heading") or 0) - (a.get("heading") or 0)) + 180) % 360 - 180)
            sa, sb = (a.get("speed") or [0, 0, 0])[1], (b.get("speed") or [0, 0, 0])[1]
            if dh > HEADING_JUMP_DEG and b.get("restart"):
                # the floor stage marks a stopped forklift taking its new travel direction: real motion, not a glitch
                restarts.append(f"forklift pulled away in a new direction after a stop (heading {dh:.0f} deg from its stopped one)")
            elif dh > HEADING_JUMP_DEG:
                jumps.append(f"heading jumped {dh:.0f} deg")
            if min(sa, sb) > 0.3 and max(sa, sb) / max(min(sa, sb), 1e-6) > SPEED_JUMP_RATIO:
                jumps.append(f"speed jumped {sa:.1f} -> {sb:.1f} m/s")
        cur = series[-1]["alert"]
        clear = [s["alert"]["clearance"]["median"] for s in series[onset:] if s["alert"] and s["alert"].get("clearance")]
        passes = [s["alert"]["pass"]["median"] for s in series[onset:] if s["alert"] and s["alert"].get("pass")]
        parts = [f"condition held {run} frame(s) in a row ({run / self.fps:.2f} s)"]
        if clear:
            parts.append(f"floor gap {min(clear):.2f}-{max(clear):.2f} m")
        if passes:
            parts.append(f"predicted pass {min(passes):.2f}-{max(passes):.2f} m")
        if gap >= GAP_FRAMES:
            parts.append(f"forklift not measured for {gap} frames just before the onset")
        if jumps:
            parts.append("; ".join(sorted(set(jumps))))
        if restarts:
            parts.append(restarts[-1])
        if cur is not None and cur.get("cut"):
            parts.append("forklift cut by the frame edge (floor outline uncertain)")
        if gap >= GAP_FRAMES or jumps:
            supports = False  # onset right after a tracking gap / jump: the course estimate may be an artefact
        elif run >= MIN_RUN_FRAMES and not (cur or {}).get("cut"):
            supports = True
        else:
            supports = None
        ref = None
        if self.frames:
            pick = [self.frames[int(k)] for k in np.linspace(0, len(self.frames) - 1, min(4, len(self.frames)))]
            tiles = []
            for t, img in pick:
                im = img.copy()
                cv2.rectangle(im, (0, 0), (im.shape[1], 30), (20, 20, 20), -1)
                cv2.putText(im, f"t={t:.2f} s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                tiles.append(im)
            ref = self.evidence.put(np.hstack(tiles), f"replay-t{self.t:.2f}")
        return Evidence(kind="replay", t=self.t, supports=supports, score=float(run), image_ref=ref, note="; ".join(parts))

    def _redetect_request(self, track_ids: list[str]):
        """Snapshot of what the second pass needs, taken NOW (frame, boxes, tracker clearance): (req, tmpdir, track_clear, t) or Evidence."""
        cmd = os.environ.get("FLOORSIGHT_REDETECT_CMD")
        if not cmd or self.last_image is None:
            return Evidence(kind="redetect", t=self.t, supports=None, note="second detector pass not configured (FLOORSIGHT_REDETECT_CMD)")
        person, vehicle = self._boxes_for(track_ids)
        if person is None or vehicle is None:
            return Evidence(kind="redetect", t=self.t, supports=None, note="person or forklift not in the current frame")
        x1, y1, x2, y2 = self._crop_region(person, vehicle, None, 1.0)
        crop = self.last_image[y1:y2, x1:x2]
        alert = next((a for a in (self.last_record or {}).get("alerts", []) if a["person"] == person["id"]), None)
        track_clear = ((alert or {}).get("clearance") or {}).get("median")
        td = tempfile.mkdtemp(prefix="redetect-")
        img_path = os.path.join(td, "crop.png")
        cv2.imwrite(img_path, crop)
        rec = self.last_record or {}
        req = {"image": img_path, "origin": [x1, y1], "person_box": person["box"], "vehicle_box": vehicle["box"],
               "heading": vehicle.get("heading"), "offset_m": list(map(float, self.cam.offset)),
               "calibration": {"K": self.cam.K.tolist(), "R": self.cam.R.tolist(), "h": self.cam.h},
               "clip": self.clip_of.get(self.last_idx), "frame": rec.get("f"), "annotated": os.path.join(td, "annotated.jpg")}
        return req, td, track_clear, self.t

    def _redetect_run(self, req: dict, td: str, track_clear, t: float) -> Evidence:
        cmd = os.environ.get("FLOORSIGHT_REDETECT_CMD")
        try:
            req_path = os.path.join(td, "request.json")
            with open(req_path, "w", encoding="utf-8") as f:
                json.dump(req, f)
            try:
                res = subprocess.run(shlex.split(cmd) + [req_path], capture_output=True, text=True, timeout=180,
                                     cwd=os.environ.get("FLOORSIGHT_REDETECT_CWD") or None)
                out = json.loads(res.stdout.strip().splitlines()[-1]) if res.stdout.strip() else {"error": res.stderr[-400:]}
                if res.returncode != 0:
                    out = {"error": f"second detector exited with status {res.returncode}"}
            except Exception as exc:  # the tool reports, never decides
                out = {"error": f"{type(exc).__name__}: {exc}"}
            ann = cv2.imread(req["annotated"]) if os.path.exists(req["annotated"]) else None
        finally:
            shutil.rmtree(td, ignore_errors=True)
        where = "raw frame inside the private perception stage" if out.get("pixels") == "raw-at-edge" else "privacy-filtered frame"
        if out.get("error") or out.get("clearance_m") is None:
            return Evidence(kind="redetect", t=t, supports=None, image_ref=self.evidence.put(ann, f"redetect-t{t:.2f}") if ann is not None else None,
                            note=f"second detector pass ({where}) inconclusive: {out.get('error') or out.get('note', 'no measurement')}")
        try:
            c = float(out["clearance_m"])
        except (TypeError, ValueError):
            c = math.nan
        if not math.isfinite(c) or c < 0:
            return Evidence(kind="redetect", t=t, supports=None, note="second detector returned an invalid clearance")
        if track_clear is None:
            supports, why = None, "no tracker clearance to compare"
        else:
            tol = max(0.4, 0.3 * track_clear)
            supports = abs(c - track_clear) <= tol
            why = f"tracker said {track_clear:.2f} m; agreement within {tol:.2f} m: {'yes' if supports else 'no'}"
        ref = self.evidence.put(ann, f"redetect-t{t:.2f}") if ann is not None else None
        return Evidence(kind="redetect", t=t, supports=supports, score=c, image_ref=ref,
                        note=f"re-detected {out.get('persons', '?')} person(s), {out.get('vehicles', '?')} forklift(s) on a x{out.get('scale', '?')} crop of the {where}; re-measured floor gap {c:.2f} m; {why}")

    def redetect(self, track_ids: list[str]) -> Evidence:
        """Second detector pass, synchronous (tests, simple callers)."""
        r = self._redetect_request(track_ids)
        return r if isinstance(r, Evidence) else self._redetect_run(*r)

    def redetect_start(self, track_ids: list[str]) -> str:
        """Start the second detector pass on a worker thread; poll with redetect_poll(token)."""
        token = f"rd-{len(self._jobs) + 1}"
        r = self._redetect_request(track_ids)
        fut: Future = Future()
        if isinstance(r, Evidence):
            fut.set_result(r)
        else:
            fut = self._pool.submit(self._redetect_run, *r)
        self._jobs[token] = fut
        return token

    def redetect_poll(self, token: str) -> Evidence | None:
        fut = self._jobs.get(token)
        if fut is None or not fut.done():
            return None
        try:
            return fut.result()
        except Exception as exc:
            return Evidence(kind="redetect", t=self.t, supports=None, note=f"second detector pass failed: {type(exc).__name__}: {exc}")

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
