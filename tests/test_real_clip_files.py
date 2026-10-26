"""GCP-only tests. All bytes are fixtures, never real privacy evidence."""
import hashlib
import os
import tempfile
from pathlib import Path

import pytest

from floorsight.real_clip_files import verify_file_bindings


@pytest.fixture
def tmp_path(tmp_path_factory):
    # Deliberately override pytest's cleanup-managed fixture. Preserve every file.
    root = Path(os.environ.get("FLOORSIGHT_TEST_ARTIFACT_ROOT") or tmp_path_factory.mktemp("file-bindings"))
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="file-bindings-", dir=root))


def put(root, path, payload):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return dict(path=path, sha256=hashlib.sha256(payload).hexdigest())


def fixture(root):
    return dict(original=put(root, "quarantine/raw.mp4", b"raw fixture"), privacy=dict(
        filtered=put(root, "filtered/clip.webm", b"filtered fixture"),
        review_evidence=put(root, "evidence/review.json", b'{"fixture": true}')))


def test_file_integrity_does_not_claim_media_or_privacy_validity(tmp_path):
    result = verify_file_bindings(tmp_path, fixture(tmp_path))
    assert result.verified and len(result.checked) == 3
    # Bytes are not playable video, and no privacy reviewer approved this fixture.
    assert not hasattr(result, "preview") and not hasattr(result, "metric_events")


def test_changed_bytes_rejected(tmp_path):
    manifest = fixture(tmp_path)
    with (tmp_path / "filtered/clip.webm").open("ab") as stream:
        stream.write(b" appended alteration")
    assert "privacy.filtered:hash_mismatch" in verify_file_bindings(tmp_path, manifest).reasons


def test_missing_file_rejected(tmp_path):
    manifest = fixture(tmp_path)
    manifest["privacy"]["review_evidence"]["path"] = "evidence/missing.json"
    assert "privacy.review_evidence:missing_file" in verify_file_bindings(tmp_path, manifest).reasons


def test_matching_empty_file_still_rejected(tmp_path):
    manifest = fixture(tmp_path)
    manifest["privacy"]["filtered"] = put(tmp_path, "filtered/empty.webm", b"")
    assert "privacy.filtered:empty_file" in verify_file_bindings(tmp_path, manifest).reasons


def test_hardlink_cannot_relabel_raw_as_filtered(tmp_path):
    manifest = fixture(tmp_path)
    os.link(tmp_path / "quarantine/raw.mp4", tmp_path / "filtered/raw-link.webm")
    manifest["privacy"]["filtered"] = dict(path="filtered/raw-link.webm", sha256=manifest["original"]["sha256"])
    assert "raw_and_filtered_same_file" in verify_file_bindings(tmp_path, manifest).reasons


@pytest.mark.parametrize("target", ["quarantine/raw.mp4", "../outside.mp4"])
def test_symlink_rejected_even_inside_root(tmp_path, target):
    manifest = fixture(tmp_path)
    make_symlink(tmp_path / "link.webm", target)
    manifest["privacy"]["filtered"] = dict(path="link.webm", sha256=manifest["original"]["sha256"])
    assert "privacy.filtered:symlink" in verify_file_bindings(tmp_path, manifest).reasons


def test_parent_symlink_rejected(tmp_path):
    manifest = fixture(tmp_path)
    make_symlink(tmp_path / "alias", "filtered", target_is_directory=True)
    manifest["privacy"]["filtered"]["path"] = "alias/clip.webm"
    assert "privacy.filtered:symlink" in verify_file_bindings(tmp_path, manifest).reasons


def make_symlink(path, target, **kwargs):
    try:
        path.symlink_to(target, **kwargs)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows account cannot create symlinks; covered by Linux CI")
        raise


def test_directory_not_accepted_as_media(tmp_path):
    manifest = fixture(tmp_path)
    manifest["privacy"]["filtered"]["path"] = "filtered"
    assert "privacy.filtered:not_regular_file" in verify_file_bindings(tmp_path, manifest).reasons


def test_declared_geometry_requires_actual_files(tmp_path):
    manifest = fixture(tmp_path)
    manifest["calibration"] = {"status": "validated"}
    result = verify_file_bindings(tmp_path, manifest)
    assert not result.verified
    assert all("calibration." + name + ":invalid_reference" in result.reasons
               for name in ("file", "heldout_evidence", "scale_measurement"))
