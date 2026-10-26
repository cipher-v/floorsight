"""Metadata admission policy for real footage; not a detector or evidence verifier.

The caller must independently verify referenced files and hashes on the GCP worker.
The app replay (floorsight.realclip) applies it at startup only when opted in with
FLOORSIGHT_REPLAY_ADMISSION=enforce: export_app_replay.py does not write manifests yet.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Admission:
    preview: bool
    pixel_observations: bool
    metric_events: bool
    reasons: tuple[str, ...]


def _reference(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    path = value.get("path")
    digest = value.get("sha256")
    if (not isinstance(path, str) or not path or "\\" in path or ":" in path
            or any(ord(char) < 32 or ord(char) == 127 for char in path)
            or any(part in ("", ".", "..") for part in path.split("/"))):
        return False
    parsed = PurePosixPath(path)
    return (not parsed.is_absolute() and ".." not in parsed.parts
            and isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest) is not None)


def assess(manifest: dict) -> Admission:
    """Fail closed on missing provenance/privacy; metric claims require calibration.

    Accepted metadata establishes eligibility for verification, not privacy coverage
    or geometric accuracy. A face detector returning zero is never a review pass.
    """
    reasons = []
    if manifest.get("schema") != "floorsight.real-clip.v1":
        reasons.append("unsupported_schema")
    source = manifest.get("source")
    if not isinstance(source, dict):
        source = {}
    url = source.get("url")
    try:
        valid_url = isinstance(url, str) and urlsplit(url).scheme == "https" and bool(urlsplit(url).hostname)
    except ValueError:
        valid_url = False
    if not valid_url or not source.get("attribution") or source.get("license") != "CC-BY-4.0":
        reasons.append("source_rights_not_supported")
    if not manifest.get("camera_id") or not _reference(manifest.get("original")):
        reasons.append("original_not_bound")
    privacy = manifest.get("privacy")
    if not isinstance(privacy, dict):
        privacy = {}
    if (privacy.get("status") != "reviewed" or not _reference(privacy.get("filtered"))
            or not _reference(privacy.get("review_evidence"))
            or privacy.get("fail_closed_check") is not True):
        reasons.append("privacy_not_reviewed")
    else:
        original = manifest.get("original")
        if isinstance(original, dict) and privacy["filtered"]["path"] == original.get("path"):
            reasons.append("raw_and_filtered_paths_identical")
    basic = not reasons
    calibration = manifest.get("calibration")
    metric = False
    if isinstance(calibration, dict):
        error = calibration.get("heldout_reprojection_p95_px")
        limit = calibration.get("acceptance_limit_px")
        numeric = (type(error) in (int, float) and type(limit) in (int, float)
                   and math.isfinite(error) and math.isfinite(limit) and 0 <= error <= limit and limit > 0)
        metric = (basic and calibration.get("status") == "validated"
                  and calibration.get("units") == "metres" and numeric
                  and _reference(calibration.get("file"))
                  and _reference(calibration.get("heldout_evidence"))
                  and _reference(calibration.get("scale_measurement")))
    if not metric:
        reasons.append("metric_geometry_not_validated")
    return Admission(basic, basic, metric, tuple(reasons))
