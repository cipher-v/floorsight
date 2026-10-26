"""Lazy boto3 client creation shared by the adapters."""

from __future__ import annotations


def make_client(service: str, region: str | None = None, client=None):
    """Return `client` if given (tests pass a stubbed one), else a boto3 client with sane retries."""
    if client is not None:
        return client
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:  # pragma: no cover
        raise ImportError("boto3 is required for floorsight.cloud (pip install -r requirements.txt)") from e
    cfg = Config(
        retries={"max_attempts": 5, "mode": "standard"},
        connect_timeout=3,
        read_timeout=10,
        signature_version="s3v4" if service == "s3" else None,  # SigV4 presigned URLs
    )
    return boto3.client(service, region_name=region, config=cfg)


def safe_ref(ref: str) -> bool:
    return bool(ref) and len(ref) <= 128 and all(ch.isalnum() or ch in "-_" for ch in ref)
