"""API smoke tests: fake source + fake vision, stub (or demo/real) agent."""

from __future__ import annotations

import importlib.util
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from floorsight.contracts import ApprovalRequest, CoverageMap, LiveMessage, SitePlan, WorldState
from floorsight.server.app import create_app
from floorsight.server.runtime import RuntimeConfig


def make_client(sim_dir, tmp_path, token="", **overrides) -> TestClient:
    cfg = RuntimeConfig(
        source="fake", sim_dir=sim_dir, vision="fake", agent="stub", speed=20.0, loop=True,
        runtime_dir=str(tmp_path / "runtime"),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return TestClient(create_app(cfg, token=token))


def wait_for(client, pred, timeout=10.0, path="/api/metrics"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        data = client.get(path).json()
        if pred(data):
            return data
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting on {path}: {data}")


def receive_kind(ws, kind, limit=500):
    for _ in range(limit):
        msg = ws.receive_json()
        if msg.get("kind") == kind:
            return msg
    raise AssertionError(f"no {kind!r} message received")


# --------------------------------------------------------------------------- #


def test_healthz_and_site(sim_truth_dir, tmp_path):
    with make_client(sim_truth_dir, tmp_path) as client:
        r = client.get("/healthz")
        assert r.status_code == 200 and r.json()["ok"] is True
        site = client.get("/api/site").json()
        plan = SitePlan.model_validate(site["site"])
        assert plan.zones and plan.cameras
        assert set(site["camera_views"]) == {c.camera_id for c in plan.cameras}
        for v in site["camera_views"].values():
            assert {"x", "y", "yaw_deg", "hfov_deg", "near_m", "far_m"} <= set(v)


def test_dashboard_served(sim_truth_dir, tmp_path):
    with make_client(sim_truth_dir, tmp_path) as client:
        r = client.get("/")
        assert r.status_code == 200 and "Floorsight" in r.text
        for asset in ("/static/app.js", "/static/styles.css"):
            assert client.get(asset).status_code == 200


def test_websocket_streams_world(sim_truth_dir, tmp_path):
    with make_client(sim_truth_dir, tmp_path) as client:
        with client.websocket_connect("/ws") as ws:
            first = receive_kind(ws, "world")
            second = receive_kind(ws, "world")
        LiveMessage.model_validate(first)
        w1, w2 = WorldState.model_validate(first["data"]), WorldState.model_validate(second["data"])
        assert w2.t >= w1.t or w2.t < 0.5  # monotonic, unless the 6 s fixture looped
        assert w1.tracks and w1.cameras


def test_state_metrics_coverage_replay_report(sim_truth_dir, tmp_path):
    with make_client(sim_truth_dir, tmp_path) as client:
        wait_for(client, lambda m: m["frames"] > 20 and m["history"]["count"] >= 3)
        st = client.get("/api/state").json()
        for key in ("site", "mode", "components", "warnings", "world", "cameras", "agent", "metrics", "history", "privacy"):
            assert key in st
        assert st["components"] == {"vision": "fake", "agent": "stub", "blur": st["components"]["blur"]}
        assert set(st["agent"]) >= {"rules", "events", "approvals", "alerts", "trace"}
        WorldState.model_validate(st["world"])
        assert {c["camera_id"] for c in st["cameras"]} == {"cam1", "cam2", "cam3", "cam4"}

        m = client.get("/api/metrics").json()
        for key in ("fps", "latency_ms", "t", "dropped", "frames"):
            assert key in m

        CoverageMap.model_validate(client.get("/api/coverage").json())

        rep = client.get("/api/replay", params={"t0": 0, "t1": 1000}).json()
        assert rep["range"]["count"] >= 3 and rep["snapshots"]
        ts = [s["t"] for s in rep["snapshots"]]
        assert ts == sorted(ts)
        assert client.get("/api/replay").status_code == 200  # defaults: the last 10 s

        report = client.get("/api/report").json()
        assert {"agent", "facts", "events", "privacy"} <= set(report)


def test_rules_and_approvals_with_stub(sim_truth_dir, tmp_path):
    with make_client(sim_truth_dir, tmp_path) as client:
        r = client.post("/api/rules", json={"text": "Keep Exit B clear for 30 seconds"})
        assert r.status_code == 200
        assert r.json() == {"ok": True, "agent": "stub", "messages": []}
        assert client.post("/api/rules", json={"text": ""}).status_code == 422
        assert client.post("/api/rules", json={"text": "   "}).status_code == 422
        # the stub has no approvals: an id it never issued is unknown, not a silent success
        r = client.post("/api/approvals/AP-1", json={"approved": True})
        assert r.status_code == 404
        assert client.post("/api/approvals/AP-1", json={}).status_code == 422


def test_rule_approval_round_trip_with_demo_agent(sim_truth_dir, tmp_path):
    with make_client(sim_truth_dir, tmp_path, agent="demo") as client:
        wait_for(client, lambda m: m["frames"] > 2)
        with client.websocket_connect("/ws") as ws:
            r = client.post("/api/rules", json={"text": "Nobody should stay in the cold room longer than 2 minutes"})
            assert r.status_code == 200
            msgs = r.json()["messages"]
            for m in msgs:
                LiveMessage.model_validate(m)
            approval = ApprovalRequest.model_validate(next(m["data"] for m in msgs if m["kind"] == "approval"))
            assert approval.action == "confirm_rule" and approval.status == "pending"
            pushed = receive_kind(ws, "approval")  # the same request is broadcast to dashboards
            assert pushed["data"]["approval_id"] == approval.approval_id

        r = client.post(f"/api/approvals/{approval.approval_id}", json={"approved": True})
        assert r.status_code == 200
        decided = next(m["data"] for m in r.json()["messages"] if m["kind"] == "approval")
        assert decided["status"] == "approved"
        rule = next(m["data"] for m in r.json()["messages"] if m["kind"] == "rule")
        assert rule["confirmed"] is True
        assert client.post("/api/approvals/nope", json={"approved": False}).status_code == 404


@pytest.mark.skipif(importlib.util.find_spec("floorsight.agent") is None, reason="real agent not built yet")
def test_rules_shape_with_real_agent(sim_truth_dir, tmp_path, monkeypatch):
    monkeypatch.delenv("FLOORSIGHT_LLM", raising=False)  # local planner only: no network
    with make_client(sim_truth_dir, tmp_path, agent="real") as client:
        st = client.get("/api/state").json()
        if st["components"]["agent"] != "real":
            pytest.skip(f"real agent did not start: {st['warnings']}")
        r = client.post("/api/rules", json={"text": "Keep Exit B clear for more than 30 seconds"})
        assert r.status_code == 200
        for m in r.json()["messages"]:
            LiveMessage.model_validate(m)


def test_camera_frames_for_fake_feed(sim_truth_dir, tmp_path):
    with make_client(sim_truth_dir, tmp_path) as client:
        wait_for(client, lambda m: m["frames"] > 2)
        assert client.get("/api/cameras/cam1/frame.jpg").status_code == 204  # fake feed has no video
        assert client.get("/api/cameras/nope/frame.jpg").status_code == 404
        assert client.get("/api/evidence/does-not-exist").status_code == 404
        assert client.get("/api/evidence/..%2F..%2Fsecret").status_code == 404


def test_token_enforcement(sim_truth_dir, tmp_path):
    with make_client(sim_truth_dir, tmp_path, token="judge-123") as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/api/site").status_code == 401
        assert client.get("/api/site", params={"token": "wrong"}).status_code == 401
        page = client.get("/")
        assert page.status_code == 401 and "access token" in page.text.lower()
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws") as ws:
                ws.receive_json()

        r = client.get("/api/site", params={"token": "judge-123"}, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/api/site"  # token leaves the URL at once
        assert "floorsight_token" in r.headers.get("set-cookie", "")
        # the cookie set above now authorises plain requests and the websocket
        assert client.get("/api/state").status_code == 200
        with client.websocket_connect("/ws") as ws:
            assert receive_kind(ws, "world")["kind"] == "world"

    with make_client(sim_truth_dir, tmp_path, token="judge-123") as client:
        assert client.get("/api/state", headers={"Authorization": "Bearer judge-123"}).status_code == 200
        with client.websocket_connect("/ws?token=judge-123") as ws:
            assert receive_kind(ws, "world")["kind"] == "world"


def test_sim_video_source_thumbnails(sim_video_dir, tmp_path):
    with make_client(sim_video_dir, tmp_path, source="sim", speed=2.0) as client:
        wait_for(client, lambda m: m["frames"] >= 3)
        wait_for(client, lambda st: all(c["has_thumbnail"] for c in st["cameras"]), path="/api/state")
        r = client.get("/api/cameras/cam2/frame.jpg")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert r.content[:2] == b"\xff\xd8"
        st = client.get("/api/state").json()
        assert all(c["has_video"] and c["width"] == 960 for c in st["cameras"])
