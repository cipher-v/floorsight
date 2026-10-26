"""Public mode: only requests carrying CloudFront's X-Floorsight-Origin secret reach the app; /healthz stays open."""

from __future__ import annotations

from fastapi.testclient import TestClient

from floorsight.server.app import create_app
from floorsight.server.runtime import RuntimeConfig


def test_direct_hits_are_refused_in_public_mode(sim_truth_dir, tmp_path):
    cfg = RuntimeConfig(source="fake", sim_dir=sim_truth_dir, vision="fake", agent="stub", speed=20.0, loop=True,
                        runtime_dir=str(tmp_path / "runtime"))
    with TestClient(create_app(cfg, token="", origin_secret="s3cret")) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/api/site").status_code == 403
        assert c.get("/api/site", headers={"X-Floorsight-Origin": "wrong"}).status_code == 403
        assert c.get("/api/site", headers={"X-Floorsight-Origin": "s3cret"}).status_code == 200


def test_private_mode_has_no_origin_check(sim_truth_dir, tmp_path):
    cfg = RuntimeConfig(source="fake", sim_dir=sim_truth_dir, vision="fake", agent="stub", speed=20.0, loop=True,
                        runtime_dir=str(tmp_path / "runtime"))
    with TestClient(create_app(cfg, token="", origin_secret="")) as c:
        assert c.get("/api/site").status_code == 200
