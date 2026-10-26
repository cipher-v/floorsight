"""Synthetic acquisition boundary tests. GCP only; all output fixtures preserved."""
import hashlib
import io
import os
import tempfile
from datetime import datetime, timedelta, timezone
from email.message import Message
from pathlib import Path

import pytest

from scripts import acquire_real_clip as acquire


@pytest.fixture
def preserved_path():
    root = Path(os.environ["FLOORSIGHT_TEST_ARTIFACT_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="acquisition-", dir=root))


def response(monkeypatch, payload, *, status=200, content_type="video/mp4", length=None):
    class Reply(io.BytesIO):
        def geturl(self):
            return "https://data.mendeley.com/fixture"
    reply = Reply(payload)
    reply.status = status
    reply.headers = Message()
    reply.headers["Content-Type"] = content_type
    if length is not None:
        reply.headers["Content-Length"] = str(length)
    class Opener:
        def open(self, request, timeout):
            return reply
    monkeypatch.setattr(acquire, "build_opener", lambda *handlers: Opener())


def future():
    return datetime.now(timezone.utc) + timedelta(minutes=5)


def test_matching_bytes_record_integrity_only(monkeypatch, preserved_path):
    payload = b"fixture, deliberately not video"
    response(monkeypatch, payload, length=len(payload))
    evidence = {}
    acquire.transfer("https://data.mendeley.com/fixture", preserved_path / "clip.mp4",
                     hashlib.sha256(payload).hexdigest(), 1024, evidence, future())
    assert evidence["publisher_sha256_match"] is True
    assert evidence["bytes"] == len(payload)
    assert "decode_result" not in evidence and "privacy_reviewed" not in evidence


def test_checksum_failure_preserves_download(monkeypatch, preserved_path):
    response(monkeypatch, b"fixture")
    target = preserved_path / "clip.mp4"
    evidence = {}
    with pytest.raises(ValueError, match="publisher_checksum_mismatch"):
        acquire.transfer("https://data.mendeley.com/fixture", target, "0" * 64,
                         1024, evidence, future())
    assert target.read_bytes() == b"fixture"
    assert evidence["publisher_sha256_match"] is False


def test_existing_output_is_not_overwritten(monkeypatch, preserved_path):
    response(monkeypatch, b"new fixture")
    target = preserved_path / "clip.mp4"
    target.write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        acquire.transfer("https://data.mendeley.com/fixture", target, "0" * 64,
                         1024, {}, future())
    assert target.read_bytes() == b"preserved"


@pytest.mark.parametrize("options,reason", [
    ({"status": 206}, "unexpected_response"),
    ({"content_type": "text/html"}, "non_media_response"),
    ({"length": 2048}, "invalid_content_length"),
])
def test_bad_response_rejected(monkeypatch, preserved_path, options, reason):
    response(monkeypatch, b"fixture", **options)
    with pytest.raises(ValueError, match=reason):
        acquire.transfer("https://data.mendeley.com/fixture", preserved_path / "clip.mp4",
                         "0" * 64, 1024, {}, future())


def test_stream_byte_limit(monkeypatch, preserved_path):
    response(monkeypatch, b"123456789")
    with pytest.raises(ValueError, match="download_byte_limit"):
        acquire.transfer("https://data.mendeley.com/fixture", preserved_path / "clip.mp4",
                         "0" * 64, 8, {}, future())
    assert (preserved_path / "clip.mp4").exists()


def test_short_body_rejected(monkeypatch, preserved_path):
    response(monkeypatch, b"short", length=100)
    with pytest.raises(ValueError, match="empty_or_incomplete_response"):
        acquire.transfer("https://data.mendeley.com/fixture", preserved_path / "clip.mp4",
                         "0" * 64, 1024, {}, future())


def test_expired_lease_fails_before_metadata_query(monkeypatch):
    monkeypatch.setattr(acquire, "metadata", lambda field: pytest.fail("metadata queried"))
    with pytest.raises(ValueError, match="lease_expired_or_not_ready"):
        acquire.validate_lease({"status": "ready", "validUntilUtc": "2000-01-01T00:00:00Z"})


def test_wrong_cloud_resource_rejected(monkeypatch):
    monkeypatch.setattr(acquire, "metadata", lambda field: "unassigned")
    with pytest.raises(ValueError, match="lease_resource_mismatch"):
        acquire.validate_lease({"status": "ready", "validUntilUtc": future().isoformat(),
                               "project": "peakai-pipeline-dev", "instance": "assigned", "zone": "assigned"})


def test_insecure_redirect_rejected():
    with pytest.raises(ValueError, match="non_https_redirect"):
        acquire.HTTPSRedirects().redirect_request(None, None, 302, "", {}, "http://example.test/file")
