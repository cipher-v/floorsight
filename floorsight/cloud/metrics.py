"""CloudWatch custom metrics: per-camera fps, pipeline latency, events. Buffered, batched."""

from __future__ import annotations

import datetime as dt
import threading

from floorsight.cloud._aws import make_client
from floorsight.contracts import WorldState

MAX_DATUMS_PER_CALL = 500  # CloudWatch accepts up to 1000 per PutMetricData; stay well under


class CloudWatchMetrics:
    """Cost note: PutMetricData is billed per call, so datums are buffered (flush_every) and
    observe_world samples at most once per `observe_every_s` of session time. At the defaults
    a 5 Hz world stream costs at most one API call per `max_age_s` (about $0.01 per 1,000 calls)."""

    def __init__(
        self,
        namespace: str = "Floorsight",
        site_id: str = "demo",
        region: str | None = None,
        client=None,
        flush_every: int = 500,
        observe_every_s: float = 10.0,
        max_age_s: float = 60.0,
    ):
        self.namespace = namespace
        self.site_id = site_id
        self.flush_every = flush_every
        self.observe_every_s = observe_every_s
        self.max_age_s = max_age_s
        self._last_observe_t: float | None = None
        self._cw = make_client("cloudwatch", region, client)
        self._buf: list[dict] = []
        self._lock = threading.Lock()

    def _add(self, name: str, value: float, unit: str, **dims: str) -> None:
        d = [{"Name": "Site", "Value": self.site_id}] + [{"Name": k, "Value": v} for k, v in sorted(dims.items())]
        datum = {
            "MetricName": name,
            "Dimensions": d,
            "Timestamp": dt.datetime.now(dt.timezone.utc),
            "Value": float(value),
            "Unit": unit,
        }
        with self._lock:
            self._buf.append(datum)
            age = (datum["Timestamp"] - self._buf[0]["Timestamp"]).total_seconds()
            full = len(self._buf) >= self.flush_every or age >= self.max_age_s
        if full:
            self.flush()

    def record_fps(self, camera_id: str, fps: float) -> None:
        self._add("CameraFps", fps, "Count/Second", Camera=camera_id)

    def record_latency(self, ms: float) -> None:
        self._add("PipelineLatency", ms, "Milliseconds")

    def record_event(self, event_type: str, count: int = 1) -> None:
        self._add("Events", count, "Count", EventType=event_type)

    def observe_world(self, world: WorldState) -> bool:
        """Sample a world tick (rate-limited by session time). Returns True if recorded."""
        if self._last_observe_t is not None and 0 <= world.t - self._last_observe_t < self.observe_every_s:
            return False
        self._last_observe_t = world.t
        for c in world.cameras:
            self.record_fps(c.camera_id, c.fps)
        self.record_latency(world.pipeline_latency_ms)
        self._add("Tracks", len(world.tracks), "Count")
        return True

    def flush(self) -> int:
        with self._lock:
            buf, self._buf = self._buf, []
        for i in range(0, len(buf), MAX_DATUMS_PER_CALL):
            self._cw.put_metric_data(Namespace=self.namespace, MetricData=buf[i : i + MAX_DATUMS_PER_CALL])
        return len(buf)
