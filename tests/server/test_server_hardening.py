"""Audit fixes (26 Oct): readiness (C3), retention flag (C6), approval conflicts (C7), token-in-URL redirect (R2),
camera reader stop/join (R6). Fake source and fake/null vision only: no video is rendered."""

from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from floorsight.contracts import ApprovalRequest, ApprovalStatus
from floorsight.server import components
from floorsight.server.__main__ import build_parser
from floorsight.server.app import _without_token, create_app
from floorsight.server.runtime import Runtime, RuntimeConfig
from floorsight.server.sources import _CameraReader

TOKEN = "judge-123-judge-123-judge"


def _cfg(sim_dir, tmp_path, **overrides) -> RuntimeConfig:
    cfg = RuntimeConfig(source="fake", sim_dir=sim_dir, vision="fake", agent="stub", speed=20.0, loop=True,
                        runtime_dir=str(tmp_path / "runtime"))
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _poll(client, path, pred, timeout=10.0):
    t0 = time.time()
    while True:
        r = client.get(path)
        if pred(r) or time.time() - t0 > timeout:
            return r
        time.sleep(0.05)


# ---- C3: liveness vs readiness ------------------------------------------------ #


def test_readyz_is_200_while_perception_runs(sim_truth_dir, tmp_path):
    with TestClient(create_app(_cfg(sim_truth_dir, tmp_path), token="", origin_secret="")) as c:
        r = _poll(c, "/readyz", lambda r: r.status_code == 200)
        assert r.status_code == 200 and r.json()["ready"] is True and r.json()["reasons"] == []
        h = c.get("/healthz").json()
        assert h["ok"] is True and h["vision"] == "fake"


def test_readyz_is_503_when_vision_fell_back_to_null(sim_truth_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(components, "_try_import", lambda m, a: (None, f"{m}.{a} unavailable (test)"))
    with TestClient(create_app(_cfg(sim_truth_dir, tmp_path, vision="real"), token=TOKEN, origin_secret="")) as c:
        _poll(c, "/healthz", lambda r: r.json()["mode"] != "starting")
        h = c.get("/healthz")
        assert h.status_code == 200 and h.json()["ok"] is True and h.json()["vision"] == "none"  # liveness unchanged
        r = c.get("/readyz")  # open like /healthz: no token
        assert r.status_code == 503
        body = r.json()
        assert body["ready"] is False and body["vision"] == "none"
        assert any("null vision" in reason for reason in body["reasons"])


class _BrokenSource:
    realtime = False
    has_video = False
    fps = 10.0

    def __iter__(self):
        raise OSError("camera unplugged at /dev/video0")

    def close(self):
        pass


def test_readyz_is_503_after_a_runtime_error_without_leaking_it(sim_truth_dir, tmp_path):
    rt = Runtime(_cfg(sim_truth_dir, tmp_path))
    rt.source = _BrokenSource()
    with TestClient(create_app(runtime=rt, token="", origin_secret="")) as c:
        r = _poll(c, "/readyz", lambda r: r.json()["mode"] == "error")
        assert r.status_code == 503 and "the runtime stopped with an error" in r.json()["reasons"]
        assert "/dev/video0" not in r.text  # the open route carries no error details
        assert c.get("/healthz").status_code == 200


# ---- C6: preservation hold ------------------------------------------------------ #


def test_preservation_hold_flag_env_and_default(monkeypatch):
    parser = build_parser()

    def hold(*argv):
        return RuntimeConfig.from_args(parser.parse_args(list(argv))).preserve_evidence

    monkeypatch.delenv("FLOORSIGHT_PRESERVATION_HOLD", raising=False)
    assert hold() is True  # default unchanged: the owner hold is on
    assert hold("--preservation-hold", "off") is False
    assert hold("--preservation-hold", "on") is True
    monkeypatch.setenv("FLOORSIGHT_PRESERVATION_HOLD", "off")
    assert hold() is False
    assert hold("--preservation-hold", "on") is True  # the flag wins over the environment
    monkeypatch.setenv("FLOORSIGHT_PRESERVATION_HOLD", "maybe")
    with pytest.raises(ValueError):
        hold()


def test_hold_off_purges_history_past_retention(sim_truth_dir, tmp_path):
    rt = Runtime(_cfg(sim_truth_dir, tmp_path, preserve_evidence=False, retention_s=3600))
    old = os.path.join(rt.history_dir, "old.jsonl")
    with open(old, "w") as f:
        f.write("{}\n")
    past = time.time() - 7200
    os.utime(old, (past, past))
    try:
        assert rt._purge()["history_files"] == 1 and not os.path.exists(old)
        assert rt.privacy()["automatic_deletion"] is True and rt.privacy()["preservation_hold"] is False
    finally:
        rt.stop()


# ---- C7: approval decisions answer with the server's state --------------------------- #


def test_decisions_return_the_authoritative_state(sim_truth_dir, tmp_path, monkeypatch):
    monkeypatch.delenv("FLOORSIGHT_LLM", raising=False)  # local planner and rule parser: no network
    cfg = _cfg(sim_truth_dir, tmp_path, agent="real", loop=False)  # no loop: the agent (and its approvals) persists
    with TestClient(create_app(cfg, token="", origin_secret="")) as c:
        rt = c.app.state.runtime
        if rt.agent_label != "real":
            pytest.skip(f"real agent did not start: {rt.warnings}")
        msgs = c.post("/api/rules", json={"text": "Keep Exit B clear for more than 30 seconds"}).json()["messages"]
        ap_id = next(m["data"]["approval_id"] for m in msgs if m["kind"] == "approval")

        r = c.post(f"/api/approvals/{ap_id}", json={"approved": True, "reviewer": "first"})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True and body["status"] == "approved" and body["approved"] is True
        assert body["approval"]["approval_id"] == ap_id and body["approval"]["status"] == "approved"

        # a second reviewer rejects the already-approved request: refused, and told what actually holds
        r = c.post(f"/api/approvals/{ap_id}", json={"approved": False, "reviewer": "second"})
        assert r.status_code == 409
        assert r.json()["status"] == "approved" and r.json()["approval"]["status"] == "approved"
        assert rt.approval(ap_id)["status"] == "approved"

        assert c.post("/api/approvals/AP-9999", json={"approved": True}).status_code == 404

        # stale: the incident closed and the request expired before anyone answered
        with rt.agent_lock:
            rt.agent.approvals["AP-OLD"] = ApprovalRequest(approval_id="AP-OLD", event_id=None, action="dispatch_check", summary="Check the area",
                                                            status=ApprovalStatus.expired, created_t=0.0, decided_t=1.0)
        r = c.post("/api/approvals/AP-OLD", json={"approved": True})
        assert r.status_code == 409 and r.json()["status"] == "expired"
        assert rt.approval("AP-OLD")["status"] == "expired"


# ---- R2: token in the URL ------------------------------------------------------------ #


def test_token_link_is_swapped_for_a_cookie_and_removed_from_the_url(sim_truth_dir, tmp_path):
    with TestClient(create_app(_cfg(sim_truth_dir, tmp_path), token=TOKEN, origin_secret="")) as c:
        r = c.get("/api/replay", params={"t0": "1", "token": TOKEN}, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/api/replay?t0=1"
        cookie = r.headers["set-cookie"]
        assert f"floorsight_token={TOKEN}" in cookie and "HttpOnly" in cookie and "SameSite=Lax" in cookie
        assert "Secure" not in cookie  # plain http (local run, SSM port-forward): a Secure cookie would never return
        assert c.get("/api/state").status_code == 200  # the cookie authorises plain requests now

        page = c.get("/", params={"token": TOKEN})  # what the login form submits
        assert page.status_code == 200 and "token" not in str(page.url) and page.history[0].status_code == 303

        # a POST keeps its method and body: no redirect, the cookie rides on the response
        r = c.post(f"/api/rules?token={TOKEN}", json={"text": "Keep Exit B clear for 30 seconds"})
        assert r.status_code == 200 and "floorsight_token" in r.headers.get("set-cookie", "")


def test_token_cookie_is_secure_over_https_and_behind_cloudfront(sim_truth_dir, tmp_path):
    with TestClient(create_app(_cfg(sim_truth_dir, tmp_path / "a"), token=TOKEN, origin_secret=""), base_url="https://testserver") as c:
        r = c.get("/", params={"token": TOKEN}, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/" and "Secure" in r.headers["set-cookie"]
    # CloudFront -> instance is plain HTTP, but viewers always reach CloudFront over HTTPS
    with TestClient(create_app(_cfg(sim_truth_dir, tmp_path / "b"), token=TOKEN, origin_secret="s3cret")) as c:
        r = c.get("/", params={"token": TOKEN}, headers={"X-Floorsight-Origin": "s3cret"}, follow_redirects=False)
        assert r.status_code == 303 and "Secure" in r.headers["set-cookie"]


def test_redirect_target_stays_on_this_site():
    assert _without_token({"raw_path": b"//evil.example/x", "query_string": b"token=a&b=2"}) == "/evil.example/x?b=2"
    assert _without_token({"raw_path": b"/\\evil.example", "query_string": b"token=a"}) == "/evil.example"
    assert _without_token({"path": "/", "query_string": b"token=a"}) == "/"


# ---- R6: camera reader lifecycle ----------------------------------------------------- #


def test_camera_reader_starts_stops_and_joins_without_a_camera(tmp_path):
    reader = _CameraReader("cam9", str(tmp_path / "no-such-camera.mp4"), retry_s=0.05)
    reader.start()
    time.sleep(0.2)  # a few failed connection attempts
    reader.stop()
    reader.join(timeout=5)  # Thread.join calls self._stop(): an Event stored under that name raised TypeError here
    assert not reader.is_alive() and reader.connected is False
