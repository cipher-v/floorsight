"""Server runtime: source -> edge blur -> vision.process -> agent.on_world -> broadcast.

Runs in one background thread, paced at real time (x --speed) for recorded
sources. Keeps the latest world, a small blurred JPEG per camera, coverage,
a 2 Hz history of world snapshots (memory ring + JSONL on disk, purged after
the retention window only when the preservation hold is off) and pipeline metrics. Hands messages to the asyncio
websocket side through ``Hub`` (thread-safe).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np
from fastapi.encoders import jsonable_encoder

from floorsight.contracts import LiveMessage, SitePlan, WorldState
from floorsight.interfaces import FramePacket
from floorsight.server import components
from floorsight.server.sources import FakeFeedSource, LiveCameraSource, VideoFileSource, parse_live_spec
from floorsight.site import load_site
from floorsight.cloud import MirrorEvidenceStore, server_adapters
from floorsight.storage import LocalEvidenceStore

log = logging.getLogger("floorsight.server.runtime")

_NUMPY = {np.generic: lambda v: v.item(), np.ndarray: lambda a: a.tolist()}


def to_jsonable(obj):
    return jsonable_encoder(obj, custom_encoder=_NUMPY)


def message_dict(msg) -> dict | None:
    """Accepts LiveMessage or {"kind", "data"} dicts from the agent."""
    if isinstance(msg, LiveMessage):
        return to_jsonable(msg)
    if isinstance(msg, dict) and "kind" in msg:
        return to_jsonable({"kind": msg["kind"], "data": msg.get("data", {})})
    log.warning("ignoring agent message of type %s", type(msg).__name__)
    return None


# --------------------------------------------------------------------------- #
# Thread -> asyncio fan-out
# --------------------------------------------------------------------------- #


class Hub:
    """Fans JSON text out to every websocket client queue. ``publish`` is thread-safe."""

    def __init__(self, queue_size: int = 256):
        self.queue_size = queue_size
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[asyncio.Queue] = set()

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def register(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size)
        self._clients.add(q)
        return q

    def unregister(self, q: asyncio.Queue) -> None:
        self._clients.discard(q)

    def publish(self, text: str, kind: str) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._fanout, text, kind)
        except RuntimeError:  # loop shutting down
            pass

    def _fanout(self, text: str, kind: str) -> None:
        for q in list(self._clients):
            if q.full():
                if kind == "world":
                    continue  # a newer world message will follow; never block on it
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(text)
            except asyncio.QueueFull:
                pass


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

HOLD_ENV = "FLOORSIGHT_PRESERVATION_HOLD"


def preservation_hold(flag: str | None, env=None) -> bool:
    """--preservation-hold on|off, else $FLOORSIGHT_PRESERVATION_HOLD, else on (the owner hold is the default)."""
    value = (flag if flag is not None else (os.environ if env is None else env).get(HOLD_ENV, "")).strip().lower()
    if value in ("", "on", "true", "1", "yes"):
        return True
    if value in ("off", "false", "0", "no"):
        return False
    raise ValueError(f"preservation hold must be on or off, got {value!r} (--preservation-hold / {HOLD_ENV})")


@dataclass
class RuntimeConfig:
    source: str = "fake"  # sim | fake | live | realclip
    sim_dir: str = "data/sim/demo_001"
    site_path: str | None = None  # defaults to <sim_dir>/site.json
    vision: str = "fake"  # fake | real | replay
    agent: str = "stub"  # stub | real | demo
    speed: float = 1.0
    loop: bool = False
    live: dict = field(default_factory=dict)  # {camera_id: index_or_url}
    live_fps: float = 10.0
    blur: str = "auto"  # auto | off
    runtime_dir: str = "data/runtime"
    retention_s: float = 24 * 3600
    preserve_evidence: bool = True  # owner preservation hold; no automatic deletion
    history_hz: float = 2.0
    history_max_s: float = 4 * 3600
    world_max_hz: float = 12.0
    thumb_width: int = 320
    thumb_hz: float = 2.0
    max_lag_s: float = 0.5  # drop frames when further behind real time than this
    setup_dir: str | None = None  # empty-floor setup images; defaults to sim_dir for recorded footage
    interlock: bool = True  # forklift interlock: slow/stop a forklift automatically on a confirmed warning
    sim_duration_s: float = 40.0  # --source simlive: length of the closed-loop scenario

    @classmethod
    def from_args(cls, args) -> RuntimeConfig:
        live = parse_live_spec(args.live) if getattr(args, "live", None) else {}
        return cls(
            source=args.source, sim_dir=args.sim_dir, site_path=args.site, vision=args.vision, agent=args.agent,
            speed=args.speed, loop=args.loop, live=live, blur=args.blur, runtime_dir=args.runtime_dir,
            retention_s=args.retention_h * 3600, preserve_evidence=preservation_hold(getattr(args, "preservation_hold", None)),
            setup_dir=getattr(args, "setup_dir", None),
            max_lag_s=float("inf") if getattr(args, "no_drop", False) or args.source == "simlive" else 0.5,
            interlock=getattr(args, "interlock", "on") != "off", sim_duration_s=getattr(args, "sim_duration", 40.0),
        )


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #


class ApprovalConflict(Exception):
    """The approval exists but is no longer pending (already decided, or expired when its incident closed)."""

    def __init__(self, approval: dict):
        super().__init__(f"approval {approval.get('approval_id')} is already {approval.get('status')}")
        self.approval = approval


class Runtime:
    def __init__(self, cfg: RuntimeConfig, hub: Hub | None = None):
        self.cfg = cfg
        self.hub = hub or Hub()
        self.warnings: list[str] = []
        self._lock = threading.RLock()  # protects state below
        self.agent_lock = threading.RLock()  # serialises every agent call
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._simlive = None
        if cfg.source == "simlive":
            # closed-loop simulator: frames are rendered as the app runs and the interlock drives its forklift
            from floorsight.sim.live import SimLiveSource

            self._simlive = SimLiveSource(duration_s=cfg.sim_duration_s)
            self.site = self._simlive.site
        else:
            site_path = cfg.site_path or os.path.join(cfg.sim_dir, "site.json")
            self.site = load_site(site_path)
        self.truth = components.load_truth(cfg.sim_dir) if cfg.source == "fake" or cfg.vision == "fake" or cfg.agent == "demo" else None
        os.makedirs(cfg.runtime_dir, exist_ok=True)
        self.evidence = LocalEvidenceStore(os.path.join(cfg.runtime_dir, "evidence"), retention_s=cfg.retention_s)
        # AWS (FLOORSIGHT_CLOUD=aws): evidence mirrored to S3, alerts to SNS, track snapshots to DynamoDB, metrics to CloudWatch
        self.cloud = server_adapters()
        self.cloud_errors: dict[str, str] = {}
        if "evidence" in self.cloud:
            self.evidence = MirrorEvidenceStore(self.evidence, self.cloud["evidence"])
        self.history_dir = os.path.join(cfg.runtime_dir, "history")
        os.makedirs(self.history_dir, exist_ok=True)

        self._fake_feed = None
        if cfg.source == "fake" or cfg.vision == "fake":
            try:
                from floorsight.fake import FakeFeed

                self._fake_feed = FakeFeed(cfg.sim_dir)
            except FileNotFoundError as exc:
                if cfg.source == "fake":
                    raise FileNotFoundError(
                        f"--source fake needs simulator ground truth in {cfg.sim_dir} (truth_events.json, truth_frames.jsonl)"
                    ) from exc
                self.warnings.append(f"No simulator ground truth in {cfg.sim_dir}; fake vision unavailable")

        self.source = self._make_source()
        self.blur_fn = None
        self._blur_pool: ThreadPoolExecutor | None = None
        self.blur_state = "off"
        self.trace_path = None
        self._trace_file = None
        if getattr(self.source, "prefiltered", False):
            # Real clips: heads/faces were pixelated by the perception stage, which fails closed (black, withheld
            # frames). Re-blurring adds nothing; blanking them (the missing-blur fallback) would hide the floor.
            self.blur_state = "prefiltered"
        elif cfg.blur != "off":
            self.blur_fn, err = components.load_blur()
            self.blur_state = "on" if self.blur_fn else "missing"
            if self.blur_fn is not None:
                # one edge device per camera in reality; threads approximate that here
                self._blur_pool = ThreadPoolExecutor(max_workers=max(1, len(self.site.cameras)), thread_name_prefix="blur")
            if err:
                self.warnings.append("Face blur module not loaded: camera previews are NOT face-blurred")
                log.warning(err)
        else:
            self.warnings.append("Face blur turned off (--blur off)")

        self.session = 0
        self.vision = None
        self.agent = None
        self.vision_label = self.agent_label = "none"
        self._new_session(initial=True)
        self.camera_views = components.camera_views(self.site, self.truth, self.vision)

        # live state
        self.mode = "starting"  # starting | live | ended | error
        self.world: WorldState | None = None
        self.world_dict: dict | None = None
        self.coverage: dict | None = None
        self._coverage_version: int | None = None
        self.thumbs: dict[str, bytes] = {}
        self._thumb_wall: dict[str, float] = {}
        self.cam_extra: dict[str, dict] = {}
        self.history_t: list[float] = []
        self.history: list[dict] = []
        self._hist_file = None
        self._last_hist_t = -1e9
        self._last_world_pub = 0.0
        self._last_purge = 0.0
        self.metrics = {
            "fps": 0.0, "target_fps": 0.0, "latency_ms": 0.0, "vision_latency_ms": 0.0, "t": 0.0,
            "frame_index": -1, "frames": 0, "dropped": 0, "errors": 0, "last_error": None, "lag_s": 0.0,
            "speed": cfg.speed, "session": 0,
        }
        self._tick_walls: deque[float] = deque(maxlen=64)
        self._refresh_coverage(force=True)

    # ---- construction ---------------------------------------------------- #

    def _make_source(self):
        cfg = self.cfg
        if cfg.source == "fake":
            return FakeFeedSource(self._fake_feed, loop=cfg.loop)
        if cfg.source == "sim":
            return VideoFileSource(cfg.sim_dir, loop=cfg.loop, site=self.site)
        if cfg.source == "live":
            spec = cfg.live or {c.camera_id: i for i, c in enumerate(self.site.cameras[:1])}
            return LiveCameraSource(spec, fps=cfg.live_fps, site=self.site)
        if cfg.source == "realclip":
            from floorsight.realclip import RealClipSource

            return RealClipSource(cfg.sim_dir, site=self.site)
        if cfg.source == "simlive":
            return self._simlive
        raise ValueError(f"unknown source {cfg.source!r}")

    def _new_session(self, initial: bool = False) -> None:
        """Fresh vision + agent (start, and every time looped footage wraps)."""
        vision_kind = self.cfg.vision
        if self.cfg.source == "live" and vision_kind == "fake":
            vision_kind = "null"  # simulator truth does not describe live cameras
        setup_dir = self.cfg.setup_dir or (self.cfg.sim_dir if self.cfg.source == "sim" else None)
        vision, vlabel, vwarn = components.build_vision(vision_kind, self.site, self.evidence, self._fake_feed, setup_dir,
                                                        replay_dir=self.cfg.sim_dir if vision_kind == "replay" else None)
        if self._simlive is not None and hasattr(vision, "set_background_plates"):
            vision.set_background_plates(self._simlive.setup_plates())  # the simulator renders its own empty floor
            vwarn = [w for w in vwarn if "setup images" not in w]
        agent, alabel, awarn = components.build_agent(self.cfg.agent, self.site, vision, self.truth, notifier=self.cloud.get("notifier"))
        from floorsight.interlock import ForkliftInterlock

        self.interlock = ForkliftInterlock(enabled=self.cfg.interlock)
        with self.agent_lock:
            old_vision = self.vision
            self.vision, self.agent = vision, agent
        self._close_vision(old_vision)
        self.vision_label, self.agent_label = vlabel, alabel
        if initial:
            self.warnings += vwarn + awarn
            if alabel == "demo":
                self.warnings.append("Agent is the scripted demo stand-in, not the real agent")
        else:
            self.session += 1
            with self._lock:
                self.history_t.clear()
                self.history.clear()
                self._last_hist_t = -1e9
                self.metrics["session"] = self.session
            self._close_history_file()
            self._refresh_coverage(force=True)

    # ---- lifecycle ------------------------------------------------------- #

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._purge()
        self._thread = threading.Thread(target=self._run, name="floorsight-runtime", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if getattr(self.source, "realtime", False):
            self._close_source()  # live readers only set flags; unblocks the iterator
        if self._thread:
            self._thread.join(timeout)
        if not self.running:
            self._close_source()  # decoders must not be released while the runtime thread reads them
            self._close_vision(self.vision)
        if self._blur_pool is not None:
            self._blur_pool.shutdown(wait=False)
        self._close_history_file()

    @staticmethod
    def _close_vision(vision) -> None:
        close = getattr(vision, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                log.exception("closing the vision pipeline failed")

    def _close_source(self) -> None:
        try:
            self.source.close()
        except Exception:
            log.exception("closing the source failed")

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ---- main loop ------------------------------------------------------- #

    def _run(self) -> None:
        cfg = self.cfg
        paced = not getattr(self.source, "realtime", False)
        fps = float(getattr(self.source, "fps", 10.0) or 10.0)
        self.metrics["target_fps"] = fps * (cfg.speed if paced else 1.0)
        t0 = wall0 = None
        last_index = None
        first = True
        try:
            for packets in self._edge_stage():
                if self._stop.is_set():
                    break
                if not packets:
                    continue
                meta = next(iter(packets.values())).meta
                if last_index is not None and meta.frame_index < last_index:
                    log.info("footage looped; starting a fresh session")
                    # no "reset" message kind exists: clients see world.t jump back and re-sync
                    self._new_session()
                    t0 = wall0 = None
                last_index = meta.frame_index
                if paced:
                    now = time.perf_counter()
                    if t0 is None:
                        t0, wall0 = meta.t, now
                    due = wall0 + (meta.t - t0) / max(cfg.speed, 1e-3)
                    if due > now:
                        if self._stop.wait(due - now):
                            break
                    elif now - due > cfg.max_lag_s and not first:
                        with self._lock:
                            self.metrics["dropped"] += 1
                            self.metrics["lag_s"] = round(now - due, 3)
                        continue
                first = False
                self._tick(packets)
                self._maybe_purge()
            self._drain_agent()
            with self._lock:
                self.mode = "ended" if not self._stop.is_set() else self.mode
            if "metrics" in self.cloud:
                try:
                    self.cloud["metrics"].flush()
                except Exception as exc:  # noqa: BLE001
                    self.cloud_errors["metrics"] = f"{type(exc).__name__}: {exc}"
            self.save_snapshot("ended")
        except Exception as exc:
            log.exception("runtime stopped")
            with self._lock:
                self.mode = "error"
                self.metrics["last_error"] = f"{type(exc).__name__}: {exc}"

    def _tick(self, packets: dict[str, FramePacket]) -> None:
        wall = time.perf_counter()
        try:
            world = self.vision.process(packets)
        except Exception as exc:
            self._error("vision.process", exc)
            return
        # The interlock acts before the agent and before any person: a confirmed warning slows or stops the forklift
        # now; the agent's second look and the human decision follow for the record and the floor lead's alert.
        try:
            for cmd in self.interlock.update(world):
                applier = getattr(self.source, "command", None)
                if cmd["command"] == "horn":
                    cmd["applied"] = None
                else:  # recorded video cannot be braked: the command is logged and shown as "not applied"
                    cmd["applied"] = bool(applier(cmd)) if callable(applier) else False
                self._trace_write({"kind": "interlock", **cmd})
            world.interlock = self.interlock.snapshot()
        except Exception as exc:
            self._error("interlock", exc)
        msgs = []
        try:
            with self.agent_lock:
                msgs = list(self.agent.on_world(world) or [])
        except Exception as exc:
            self._error("agent.on_world", exc)
        proc_ms = (time.perf_counter() - wall) * 1000.0
        world_dict = to_jsonable(world)
        now = time.perf_counter()
        with self._lock:
            self.mode = "live"
            self.world, self.world_dict = world, world_dict
            self._tick_walls.append(now)
            m = self.metrics
            m["frames"] += 1
            m["t"] = world.t
            m["frame_index"] = next(iter(packets.values())).meta.frame_index
            m["latency_ms"] = round(proc_ms if m["latency_ms"] == 0 else 0.8 * m["latency_ms"] + 0.2 * proc_ms, 2)
            m["vision_latency_ms"] = round(float(world.pipeline_latency_ms), 2)
            if len(self._tick_walls) >= 2:
                span = self._tick_walls[-1] - self._tick_walls[0]
                m["fps"] = round((len(self._tick_walls) - 1) / span, 2) if span > 0 else 0.0
        self._thumbnails(packets, now)
        self._refresh_coverage()
        self._record_history(world, world_dict)
        self._cloud_tick(world, msgs)
        if now - self._last_world_pub >= 1.0 / self.cfg.world_max_hz:
            self._last_world_pub = now
            self.hub.publish(json.dumps({"kind": "world", "data": world_dict}), "world")
        self.broadcast(msgs)

    def _drain_agent(self, limit_s: float = 120.0) -> None:
        """After the last frame: let second detector passes that are still running report back, so their warnings
        still get a recommendation and an approval card (a short recording would otherwise end first)."""
        pending = getattr(self.agent, "pending_work", None)
        if not callable(pending):
            return
        deadline = time.time() + limit_s
        while not self._stop.is_set() and time.time() < deadline:
            with self.agent_lock:
                if not pending():
                    return
                msgs = list(self.agent.drain() or [])
            self.broadcast(msgs)
            self._stop.wait(0.5)

    def _cloud_tick(self, world: WorldState, msgs: list) -> None:
        """Track snapshot (DynamoDB, <= 1/s) and metrics (CloudWatch, buffered); failures are counted, never fatal."""
        for key, fn in (("tracks", lambda a: a.put_world(world)), ("metrics", lambda a: a.observe_world(world))):
            a = self.cloud.get(key)
            if a is None:
                continue
            try:
                fn(a)
            except Exception as exc:  # noqa: BLE001
                self.cloud_errors[key] = f"{type(exc).__name__}: {exc}"
        m = self.cloud.get("metrics")
        if m is not None:
            for msg in msgs:
                kind = getattr(msg, "kind", None)
                data = getattr(msg, "data", None)
                if kind == "alert" or (kind == "event" and isinstance(data, dict) and data.get("status") == "candidate"):
                    try:
                        m.record_event("alert" if kind == "alert" else str(data.get("type", "event")))
                    except Exception as exc:  # noqa: BLE001
                        self.cloud_errors["metrics"] = f"{type(exc).__name__}: {exc}"

    def cloud_status(self) -> dict:
        ev = self.evidence if isinstance(self.evidence, MirrorEvidenceStore) else None
        return {"adapters": sorted(self.cloud), "errors": dict(self.cloud_errors),
                "evidence_mirrored": ev.mirrored if ev else 0, "evidence_mirror_errors": ev.errors if ev else 0,
                "sns_sent": len(getattr(self.cloud.get("notifier"), "sent", []))}

    def _error(self, where: str, exc: Exception) -> None:
        log.exception("%s failed", where)
        with self._lock:
            self.metrics["errors"] += 1
            self.metrics["last_error"] = f"{where}: {type(exc).__name__}: {exc}"

    # ---- edge privacy ---------------------------------------------------- #

    def _edge_stage(self, depth: int = 3):
        """Decode + face blur in their own thread, one step ahead of vision.

        On a real site each camera's edge device does this before upload; here it
        overlaps with vision so blur cost does not add to the pipeline period.
        """
        q: queue.Queue = queue.Queue(maxsize=depth)
        done = object()

        def produce():
            try:
                for packets in self.source:
                    if self._stop.is_set():
                        break
                    item = self._edge_blur(packets) if packets else packets
                    while not self._stop.is_set():
                        try:
                            q.put(item, timeout=0.2)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:  # surfaced through the consumer
                q.put(exc)
            finally:
                q.put(done)

        threading.Thread(target=produce, name="edge", daemon=True).start()
        while True:
            item = q.get()
            if item is done:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def _blur_one(self, image):
        try:
            img, n = self.blur_fn(image)
            return img, int(n), True
        except Exception as exc:
            self._error("blur_faces", exc)
            return np.zeros_like(image), 0, False  # fail closed: raw pixels must not reach previews/evidence

    def _edge_blur(self, packets: dict[str, FramePacket]) -> dict[str, FramePacket]:
        out = {}
        if self.blur_state == "prefiltered":
            results = {cam_id: (p.image, p.meta.faces_blurred, True) for cam_id, p in packets.items()}
        elif self.blur_fn is not None and self._blur_pool is not None and len(packets) > 1:
            results = dict(zip(packets, self._blur_pool.map(self._blur_one, [p.image for p in packets.values()])))
        elif self.blur_fn is not None:
            results = {cam_id: self._blur_one(p.image) for cam_id, p in packets.items()}
        else:
            results = {cam_id: (np.zeros_like(p.image) if self.cfg.blur != "off" else p.image, 0, False) for cam_id, p in packets.items()}
        for cam_id, p in packets.items():
            img, n, applied = results[cam_id]
            meta = p.meta.model_copy(update={"faces_blurred": int(n), "blur_applied": applied})
            out[cam_id] = FramePacket(meta, img)
            extra = self.cam_extra.setdefault(cam_id, {"faces_blurred_total": 0})
            extra["faces_blurred"] = int(n)
            extra["faces_blurred_total"] += int(n)
            extra["blur_applied"] = applied
            extra["blur_mode"] = self.blur_state  # "prefiltered": heads/faces were pixelated upstream, so the count here stays 0
            extra["width"], extra["height"] = meta.width, meta.height
            extra["last_frame_t"] = meta.t
        return out

    def _thumbnails(self, packets: dict[str, FramePacket], now: float) -> None:
        if not getattr(self.source, "has_video", True):
            return
        for cam_id, p in packets.items():
            if now - self._thumb_wall.get(cam_id, 0.0) < 1.0 / self.cfg.thumb_hz:
                continue
            self._thumb_wall[cam_id] = now
            img = p.image
            h, w = img.shape[:2]
            if w <= 0 or h <= 0:
                continue
            tw = min(self.cfg.thumb_width, w)
            small = cv2.resize(img, (tw, max(1, int(h * tw / w))), interpolation=cv2.INTER_AREA)
            if not p.meta.blur_applied and self.cfg.source == "live":
                # no face blur available on real cameras: pixelate previews instead of leaking faces
                tiny = cv2.resize(small, (max(1, tw // 12), max(1, small.shape[0] // 12)), interpolation=cv2.INTER_AREA)
                small = cv2.resize(tiny, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with self._lock:
                    self.thumbs[cam_id] = buf.tobytes()

    # ---- coverage, history, retention ----------------------------------- #

    def _refresh_coverage(self, force: bool = False) -> None:
        version = None if self.world is None else self.world.coverage_version
        if not force and version == self._coverage_version:
            return
        try:
            cov = to_jsonable(self.vision.coverage())
        except Exception as exc:
            self._error("vision.coverage", exc)
            return
        with self._lock:
            self.coverage = cov
            self._coverage_version = version if version is not None else cov.get("version")
        self.hub.publish(json.dumps({"kind": "coverage", "data": cov}), "coverage")

    def _record_history(self, world: WorldState, world_dict: dict) -> None:
        if world.t - self._last_hist_t < 1.0 / self.cfg.history_hz - 1e-6:
            return
        self._last_hist_t = world.t
        snap = {"t": world.t, "tracks": world_dict.get("tracks", []), "cameras": world_dict.get("cameras", [])}
        max_n = int(self.cfg.history_max_s * self.cfg.history_hz)
        with self._lock:
            self.history_t.append(world.t)
            self.history.append(snap)
            if len(self.history) > max_n:
                k = len(self.history) - max_n
                del self.history[:k]
                del self.history_t[:k]
        try:
            if self._hist_file is None:
                name = time.strftime("%Y%m%d-%H%M%S") + f"-s{self.session}.jsonl"
                self._hist_file = open(os.path.join(self.history_dir, name), "a", encoding="utf-8")
            self._hist_file.write(json.dumps(snap, separators=(",", ":")) + "\n")
            self._hist_file.flush()
        except OSError as exc:
            self._error("history write", exc)

    def _close_history_file(self) -> None:
        f, self._hist_file = self._hist_file, None
        if f is not None:
            try:
                f.close()
            except OSError:
                pass

    def _maybe_purge(self) -> None:
        if time.time() - self._last_purge > 600:
            self._purge()

    def _purge(self) -> dict:
        """Privacy: delete history files and evidence images older than the retention window."""
        self._last_purge = time.time()
        if self.cfg.preserve_evidence:
            return {"history_files": 0, "evidence_images": 0, "preserved": True}
        cutoff = time.time() - self.cfg.retention_s
        removed = 0
        current = getattr(self._hist_file, "name", None)
        for name in os.listdir(self.history_dir):
            path = os.path.join(self.history_dir, name)
            if path != current and os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass
        evidence_removed = self.evidence.purge_expired()
        return {"history_files": removed, "evidence_images": evidence_removed}

    # ---- broadcasting ---------------------------------------------------- #

    def broadcast(self, msgs) -> list[dict]:
        out = []
        for m in msgs or []:
            d = message_dict(m)
            if d is None:
                continue
            out.append(d)
            self.hub.publish(json.dumps(d), d["kind"])
            self._trace_write(d)
        return out

    # ---- trace on disk (judges read it) --------------------------------- #

    TRACE_KINDS = ("event", "agent_step", "approval", "alert", "rule", "interlock")

    def _trace_write(self, d: dict) -> None:
        """Every agent message is appended, as it happens, to runtime_dir/traces/<start>-s<session>.jsonl."""
        if d.get("kind") not in self.TRACE_KINDS:
            return
        try:
            with self._lock:
                if self._trace_file is None:
                    os.makedirs(os.path.join(self.cfg.runtime_dir, "traces"), exist_ok=True)
                    self.trace_path = os.path.join(self.cfg.runtime_dir, "traces", time.strftime("%Y%m%d-%H%M%S") + f"-s{self.session}.jsonl")
                    self._trace_file = open(self.trace_path, "a", encoding="utf-8")
                row = {"wall_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **d}
                self._trace_file.write(json.dumps(row, separators=(",", ":")) + "\n")
                self._trace_file.flush()
        except OSError as exc:
            self._error("trace write", exc)

    def save_snapshot(self, why: str) -> str | None:
        """Full agent snapshot (events, evidence, approvals, alerts, trace) next to the trace file."""
        try:
            os.makedirs(os.path.join(self.cfg.runtime_dir, "traces"), exist_ok=True)
            if self.trace_path:
                base = self.trace_path[: -len(".jsonl")]
            else:
                base = os.path.join(self.cfg.runtime_dir, "traces", time.strftime("%Y%m%d-%H%M%S") + f"-s{self.session}")
            path = base + "-snapshot.json"
            snap = {"saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "why": why, "site": self.site.site_id,
                    "config": {"source": self.cfg.source, "vision": self.vision_label, "agent": self.agent_label, "speed": self.cfg.speed},
                    "privacy": self.privacy(), "agent": self.agent_snapshot(), "interlock": self.interlock_log()}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snap, f, indent=1)
            return path
        except OSError as exc:
            self._error("snapshot write", exc)
            return None

    # ---- API helpers (called from request threads) ----------------------- #

    def interlock_log(self) -> dict:
        lock = getattr(self, "interlock", None)
        if lock is None:
            return {"enabled": False, "commands": [], "counts": {}}
        return {"enabled": lock.enabled, "commands": list(lock.log), "counts": dict(lock.counts)}

    def current_t(self) -> float:
        with self._lock:
            return float(self.world.t) if self.world is not None else 0.0

    def add_rule(self, text: str) -> list[dict]:
        with self.agent_lock:
            msgs = self.agent.add_rule_from_text(text, self.current_t())
        return self.broadcast(msgs)

    def approval(self, approval_id: str) -> dict | None:
        """The agent's own record of one approval (None when it does not know the id)."""
        with self.agent_lock:
            known = getattr(self.agent, "approvals", None)
            if isinstance(known, dict):
                ap = known.get(approval_id)
                return to_jsonable(ap) if ap is not None else None
            snap = self.agent.snapshot() or {}
        return next((a for a in to_jsonable(snap.get("approvals", [])) if a.get("approval_id") == approval_id), None)

    def decide(self, approval_id: str, approved: bool, reviewer: str | None = None) -> list[dict]:
        """KeyError for an unknown approval, ApprovalConflict for one that is no longer pending: a late or second
        click must never look like it changed anything."""
        with self.agent_lock:  # the check and the decision are one step for concurrent reviewers
            current = self.approval(approval_id)
            if current is None:
                raise KeyError(approval_id)
            if current.get("status") != "pending":
                raise ApprovalConflict(current)
            try:
                msgs = self.agent.decide(approval_id, approved, self.current_t(), reviewer=reviewer)
            except TypeError:  # agents without reviewer support (stub, demo)
                msgs = self.agent.decide(approval_id, approved, self.current_t())
        out = self.broadcast(msgs)
        self.save_snapshot(f"decision {approval_id}")
        return out

    def agent_snapshot(self) -> dict:
        try:
            with self.agent_lock:
                snap = self.agent.snapshot() or {}
        except Exception as exc:
            self._error("agent.snapshot", exc)
            snap = {}
        snap = to_jsonable(snap)
        for key in ("rules", "events", "approvals", "alerts", "trace"):
            snap.setdefault(key, [])
        return snap

    def shift_report(self) -> dict:
        try:
            with self.agent_lock:
                return to_jsonable(self.agent.shift_report() or {})
        except Exception as exc:
            self._error("agent.shift_report", exc)
            return {"error": f"shift report failed: {exc}"}

    def camera_list(self) -> list[dict]:
        with self._lock:
            health = {c["camera_id"]: c for c in (self.world_dict or {}).get("cameras", [])}
            out = []
            for cam in self.site.cameras:
                cid = cam.camera_id
                out.append({
                    "camera_id": cid,
                    "health": health.get(cid),
                    "has_video": bool(getattr(self.source, "has_video", True)),
                    "has_thumbnail": cid in self.thumbs,
                    "nominal_fps": cam.fps,
                    **self.cam_extra.get(cid, {}),
                })
            return out

    def thumbnail(self, camera_id: str) -> bytes | None:
        with self._lock:
            return self.thumbs.get(camera_id)

    def history_range(self) -> dict:
        with self._lock:
            if not self.history_t:
                return {"t_min": None, "t_max": None, "count": 0}
            return {"t_min": self.history_t[0], "t_max": self.history_t[-1], "count": len(self.history_t)}

    def replay(self, t0: float | None, t1: float | None, max_points: int = 600) -> dict:
        import bisect

        with self._lock:
            rng = self.history_range()
            if not self.history_t:
                return {"range": rng, "snapshots": []}
            hi_t = self.history_t[-1] if t1 is None else t1
            lo_t = hi_t - 10.0 if t0 is None else t0
            if lo_t > hi_t:
                lo_t, hi_t = hi_t, lo_t
            i0 = bisect.bisect_left(self.history_t, lo_t)
            i1 = bisect.bisect_right(self.history_t, hi_t)
            snaps = self.history[i0:i1]
        if len(snaps) > max_points:
            stride = len(snaps) / max_points
            snaps = [snaps[int(k * stride)] for k in range(max_points - 1)] + [snaps[-1]]
        return {"range": rng, "snapshots": snaps}

    def metrics_snapshot(self) -> dict:
        with self._lock:
            m = dict(self.metrics)
            m["lag_s"] = m.get("lag_s", 0.0)
        m["ws_clients"] = self.hub.client_count
        m["mode"] = self.mode
        m["running"] = self.running
        m["history"] = self.history_range()
        if isinstance(self.source, LiveCameraSource):
            m["live_connected"] = self.source.status()
        return m

    def readiness(self) -> dict:
        """Is the floor actually being watched? /healthz only says the process is up. The route is open (no token),
        so reasons are fixed phrases; error details stay behind /api/metrics."""
        with self._lock:
            mode = self.mode
        reasons = []
        if self.vision_label == "none":
            reasons.append("perception fell back to the null vision: nothing is being detected")
        if mode == "error":
            reasons.append("the runtime stopped with an error")
        elif mode == "starting":
            reasons.append("no frame processed yet: the source is not delivering")
        if mode in ("starting", "live") and not self.running:
            reasons.append("the runtime thread is not running")
        if isinstance(self.source, LiveCameraSource) and not any(self.source.status().values()):
            reasons.append("no live camera is connected")
        # "ended" (recorded footage played to the end without --loop) is a normal finish, not a failure
        return {"ready": not reasons, "reasons": reasons, "mode": mode, "running": self.running,
                "vision": self.vision_label, "agent": self.agent_label}

    def privacy(self) -> dict:
        hours = self.cfg.retention_s / 3600
        return {
            "faces_blurred": self.blur_state in ("on", "prefiltered"),
            "blur": self.blur_state,
            "prefiltered_note": "heads and faces pixelated upstream by the perception stage, which fails closed" if self.blur_state == "prefiltered" else None,
            "positions_only": False,  # positions plus blurred/filtered evidence crops are stored
            "preservation_hold": self.cfg.preserve_evidence,
            "automatic_deletion": not self.cfg.preserve_evidence,
            "retention_h": round(hours, 1) if hours % 1 else int(hours),
        }

    def state(self) -> dict:
        with self._lock:
            world = self.world_dict
            mode = self.mode
        return {
            "site": {"site_id": self.site.site_id, "name": self.site.name},
            "mode": mode,
            "session": self.session,
            "config": {"source": self.cfg.source, "speed": self.cfg.speed, "loop": self.cfg.loop},
            "components": {"vision": self.vision_label, "agent": self.agent_label, "blur": self.blur_state},
            "cloud": self.cloud_status(),
            "warnings": list(self.warnings),
            "world": world,
            "cameras": self.camera_list(),
            "agent": self.agent_snapshot(),
            "metrics": self.metrics_snapshot(),
            "history": self.history_range(),
            "privacy": self.privacy(),
            "interlock": self.interlock_log(),
        }
