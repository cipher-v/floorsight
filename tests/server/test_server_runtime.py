"""Sources, components and runtime pieces without HTTP."""

from __future__ import annotations

import asyncio
import json
import os
import time

import numpy as np

from floorsight.contracts import Track, TrackClass, TrackState
from floorsight.server import components
from floorsight.server.runtime import Hub, Runtime, RuntimeConfig, to_jsonable
from floorsight.server.sources import LiveCameraSource, VideoFileSource, parse_live_spec
from floorsight.site import load_site


def test_video_source_is_synchronised(sim_video_dir):
    src = VideoFileSource(sim_video_dir)
    frames = list(src)
    src.close()
    assert len(frames) == 10  # 1 s at 10 fps
    for i, packets in enumerate(frames):
        assert set(packets) == {"cam1", "cam2", "cam3", "cam4"}
        assert {p.meta.frame_index for p in packets.values()} == {i}
        assert {p.meta.t for p in packets.values()} == {round(i / 10.0, 3)}
        assert all(p.image.shape == (540, 960, 3) for p in packets.values())


def test_video_source_loops(sim_video_dir):
    src = VideoFileSource(sim_video_dir, loop=True)
    idx = []
    for packets in src:
        idx.append(next(iter(packets.values())).meta.frame_index)
        if len(idx) == 25:
            break
    src.close()
    assert idx[:10] == list(range(10)) and idx[10:20] == list(range(10))


def test_live_source_reads_from_a_url(sim_video_dir):
    # a file path is a valid OpenCV "URL": exercises the reader threads without hardware
    spec = parse_live_spec(f"cam1={os.path.join(sim_video_dir, 'cam1.mp4')},cam9=7")
    assert spec["cam9"] == 7
    src = LiveCameraSource({"cam1": spec["cam1"]}, fps=20)
    got = None
    t0 = time.time()
    for packets in src:
        got = packets
        if packets or time.time() - t0 > 5:
            break
    src.close()
    assert got and "cam1" in got and got["cam1"].meta.width == 960


def test_camera_views_from_sim_pose(sim_truth_dir):
    site = load_site(os.path.join(sim_truth_dir, "site.json"))
    truth = components.load_truth(sim_truth_dir)
    views = components.camera_views(site, truth)
    # cam1 sits in the bottom-left corner looking into the floor (towards +x, +y)
    assert 20 < views["cam1"]["yaw_deg"] < 60 and views["cam1"]["source"] == "sim_pose"
    assert all(0 < v["near_m"] < v["far_m"] for v in views.values())
    guessed = components.camera_views(site, None)
    assert guessed["cam1"]["source"] == "guess"


def test_fallbacks_when_real_parts_are_missing(sim_truth_dir, monkeypatch):
    site = load_site(os.path.join(sim_truth_dir, "site.json"))
    monkeypatch.setattr(components, "_try_import", lambda m, a: (None, f"{m}.{a} unavailable (test)"))
    vision, label, warnings = components.build_vision("real", site, evidence_store=None, fake_feed=None)
    assert label == "none" and warnings
    agent, alabel, awarn = components.build_agent("real", site, vision)
    assert alabel == "stub" and awarn and agent.on_world(vision.process({})) == []
    blur, err = components.load_blur()
    assert blur is None and err


def test_history_is_persisted_and_preserved(sim_truth_dir, tmp_path):
    cfg = RuntimeConfig(source="fake", sim_dir=sim_truth_dir, speed=50.0, runtime_dir=str(tmp_path / "rt"), retention_s=3600)
    rt = Runtime(cfg)
    old = os.path.join(rt.history_dir, "old.jsonl")
    with open(old, "w") as f:
        f.write("{}\n")
    past = time.time() - 7200
    os.utime(old, (past, past))
    rt.start()
    t0 = time.time()
    while rt.metrics_snapshot()["history"]["count"] < 4 and time.time() - t0 < 10:
        time.sleep(0.05)
    rt.stop()
    assert os.path.exists(old)  # owner preservation hold, including startup
    assert rt.privacy()["automatic_deletion"] is False
    files = [f for f in os.listdir(rt.history_dir) if f.endswith(".jsonl") and f != "old.jsonl"]
    assert files
    with open(os.path.join(rt.history_dir, files[0])) as f:
        snap = json.loads(f.readline())
    assert {"t", "tracks", "cameras"} <= set(snap)


def test_hub_drops_world_messages_for_slow_clients():
    async def scenario():
        hub = Hub(queue_size=2)
        hub.attach_loop(asyncio.get_running_loop())
        q = hub.register()
        for i in range(5):
            hub.publish(f"w{i}", "world")
        hub.publish("event", "event")
        await asyncio.sleep(0.05)
        out = []
        while not q.empty():
            out.append(q.get_nowait())
        return out

    out = asyncio.run(scenario())
    assert out[-1] == "event" and len(out) == 2


def test_jsonable_handles_numpy_and_models():
    tr = Track(track_id="T1", cls=TrackClass.person, x=1, y=2, vx=0, vy=0, sigma_m=0.1, camera_ids=[],
               first_seen_t=0, last_seen_t=0, state=TrackState.seen)
    out = to_jsonable({"a": np.float32(1.5), "b": np.arange(3), "t": tr})
    assert out["a"] == 1.5 and out["b"] == [0, 1, 2] and out["t"]["cls"] == "person"
    json.dumps(out)
