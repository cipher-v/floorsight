"""Replay admission (audit R3): opt-in, fail-closed check of a replay folder's metadata, privacy review and file hashes.

Tiny synthetic folders; admission never decodes video, so the "video" is placeholder bytes. The raw source clip is
bound by hash only and is never present, as in a real replay folder.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from floorsight.contracts import FrameMeta
from floorsight.interfaces import FramePacket
from floorsight.realclip import ADMISSION_ENV, RealClipSource, ReplayNotAdmitted, ReplayVision, admit_replay, load_playlist
from floorsight.site import load_site
from floorsight.storage import LocalEvidenceStore

K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
R = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]  # looking straight down
H = 4.0
OFF = [-5.0, -5.0]
RAW_SHA = hashlib.sha256(b"raw source clip: stays on the perception worker").hexdigest()


def _put(root: Path, rel: str, payload: bytes) -> dict:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(payload)
    return {"path": rel, "sha256": hashlib.sha256(payload).hexdigest()}


def make_folder(root: Path, with_manifest: bool = True, manifest_edit=None) -> Path:
    d = root / "replay"
    d.mkdir()
    site = {"schema_version": "1.3", "site_id": "test-real", "name": "test", "width_m": 10.0, "depth_m": 10.0, "zones": [], "obstacles": [], "markers": [],
            "cameras": [{"camera_id": "camA", "width": 1920, "height": 1080, "fps": 25.0, "K": K, "dist": [0.0] * 5, "nominal_position": [5.0, 5.0, H]}],
            "floor_stage": True}
    (d / "site.json").write_text(json.dumps(site))
    rec = {"f": 0, "t": 0.0, "persons": [{"id": "T1:P1", "xy": [5.5, 5.0], "sigma": 0.08}], "vehicles": [], "zones": {}, "alerts": []}
    (d / "T1.frames.jsonl").write_text(json.dumps(rec) + "\n")
    video = _put(d, "T1-filtered.mp4", b"privacy-filtered frames (placeholder bytes)")
    evidence = _put(d, "evidence/T1-privacy-review.json", b'{"fixture": true}')
    manifest = {"schema": "floorsight.real-clip.v1", "camera_id": "camera9",
                "source": {"url": "https://data.mendeley.com/datasets/xjmtb22pff/1", "license": "CC-BY-4.0", "attribution": "Onal and Dandil"},
                "original": {"path": "quarantine/T1.mp4", "sha256": RAW_SHA},
                "privacy": {"status": "reviewed", "filtered": video, "review_evidence": evidence, "fail_closed_check": True}}
    if manifest_edit:
        manifest_edit(manifest)
    clip = {"name": "T1", "code": "T1", "fps": 25.0, "frames": 1, "offset_frames": 0, "offset_t": 0.0, "video": "T1-filtered.mp4",
            "records": "T1.frames.jsonl", "clipSha256": RAW_SHA}
    if with_manifest:
        clip["manifest"] = _put(d, "T1.manifest.json", json.dumps(manifest).encode())
    playlist = {"camera_id": "camA", "calibration": {"K": K, "R": R, "h": H}, "offset_m": OFF, "frame_size": [1920, 1080], "clips": [clip]}
    (d / "playlist.json").write_text(json.dumps(playlist))
    return d


def _edit_clip(d: Path, **changes) -> None:
    pl = json.loads((d / "playlist.json").read_text())
    pl["clips"][0].update(changes)
    (d / "playlist.json").write_text(json.dumps(pl))


def _refused(d: Path) -> tuple[str, ...]:
    with pytest.raises(ReplayNotAdmitted) as info:
        load_playlist(d, admission=True)
    return info.value.reasons


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ADMISSION_ENV, raising=False)
    monkeypatch.delenv("FLOORSIGHT_REDETECT_CMD", raising=False)


def test_off_by_default_so_existing_folders_still_load(tmp_path):
    d = make_folder(tmp_path, with_manifest=False)
    assert load_playlist(d)["clips"][0]["name"] == "T1"
    RealClipSource(str(d))
    ReplayVision(load_site(str(d / "site.json")), LocalEvidenceStore(str(tmp_path / "ev"), retention_s=3600), str(d)).close()


def test_reviewed_and_hash_bound_folder_is_admitted_and_plays(tmp_path, monkeypatch):
    d = make_folder(tmp_path)
    assert admit_replay(d, load_playlist(d)) == ()
    monkeypatch.setenv(ADMISSION_ENV, "enforce")
    RealClipSource(str(d))
    vision = ReplayVision(load_site(str(d / "site.json")), LocalEvidenceStore(str(tmp_path / "ev"), retention_s=3600), str(d))
    meta = FrameMeta(camera_id="camA", frame_index=0, t=0.0, width=192, height=108, faces_blurred=0, blur_applied=True)
    world = vision.process({"camA": FramePacket(meta, np.zeros((108, 192, 3), np.uint8))})
    vision.close()
    assert [t.track_id for t in world.tracks] == ["T1:P1"]


def test_missing_metadata_is_refused(tmp_path):
    d = make_folder(tmp_path, with_manifest=False)
    assert _refused(d) == ("T1:metadata_missing",)
    _edit_clip(d, manifest={"path": "T1.manifest.json", "sha256": "a" * 64})
    assert _refused(d) == ("T1:manifest:missing_file",)


def test_altered_manifest_is_refused(tmp_path):
    d = make_folder(tmp_path)
    m = json.loads((d / "T1.manifest.json").read_text())
    m["privacy"]["status"] = "reviewed "  # any byte change breaks the playlist's binding
    (d / "T1.manifest.json").write_text(json.dumps(m))
    assert _refused(d) == ("T1:manifest:hash_mismatch",)


def test_altered_video_hash_mismatch_is_refused(tmp_path):
    d = make_folder(tmp_path)
    with (d / "T1-filtered.mp4").open("ab") as f:
        f.write(b" appended alteration")
    assert _refused(d) == ("T1:privacy.filtered:hash_mismatch",)


def test_altered_privacy_review_evidence_is_refused(tmp_path):
    d = make_folder(tmp_path)
    (d / "evidence/T1-privacy-review.json").write_bytes(b'{"fixture": false}')
    assert _refused(d) == ("T1:privacy.review_evidence:hash_mismatch",)


@pytest.mark.parametrize("privacy", [
    {"status": "pending"},  # review not finished
    {"fail_closed_check": False},  # filter did not prove it fails closed
])
def test_failed_privacy_review_is_refused(tmp_path, privacy):
    d = make_folder(tmp_path, manifest_edit=lambda m: m["privacy"].update(privacy))
    assert "T1:privacy_not_reviewed" in _refused(d)


def test_zero_detected_faces_is_not_a_privacy_review(tmp_path):
    d = make_folder(tmp_path, manifest_edit=lambda m: m.update(privacy={"faces_detected": 0, "status": "processed"}))
    reasons = _refused(d)
    assert "T1:privacy_not_reviewed" in reasons and "T1:video_is_not_the_reviewed_file" in reasons


def test_manifest_for_a_different_video_is_refused(tmp_path):
    d = make_folder(tmp_path)
    (d / "T1-other.mp4").write_bytes(b"some other video")
    _edit_clip(d, video="T1-other.mp4")
    assert _refused(d) == ("T1:video_is_not_the_reviewed_file",)


def test_records_from_another_source_clip_are_refused(tmp_path):
    d = make_folder(tmp_path)
    _edit_clip(d, clipSha256="b" * 64)
    assert _refused(d) == ("T1:records_not_from_the_reviewed_source",)


def test_app_constructors_fail_closed_when_enforced(tmp_path, monkeypatch):
    d = make_folder(tmp_path, with_manifest=False)
    site = load_site(str(d / "site.json"))
    monkeypatch.setenv(ADMISSION_ENV, "enforce")
    with pytest.raises(ReplayNotAdmitted, match="T1:metadata_missing"):
        RealClipSource(str(d), site=site)
    with pytest.raises(ReplayNotAdmitted, match="T1:metadata_missing"):
        ReplayVision(site, LocalEvidenceStore(str(tmp_path / "ev"), retention_s=3600), str(d))
    monkeypatch.setenv(ADMISSION_ENV, "on")  # an unrecognised value enforces too (fail closed)
    with pytest.raises(ReplayNotAdmitted):
        RealClipSource(str(d), site=site)
    monkeypatch.setenv(ADMISSION_ENV, "0")
    RealClipSource(str(d), site=site)
