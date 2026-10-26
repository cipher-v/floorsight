"""Read-only artifact integrity checks. Raw-media checks execute on the coordinated GCP worker only.

This checks file bindings, NOT the truth or adequacy of a privacy/geometry review.
It grants no preview or metric permissions. The app replay (floorsight.realclip) runs it
opt-in at startup with skip=("original",): a replay folder never holds the raw clip.
Use a private, immutable job directory; concurrent mutation is unsupported.
"""
from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from .real_clip_contract import _reference


@dataclass(frozen=True)
class FileBindings:
    verified: bool
    checked: tuple[str, ...]
    reasons: tuple[str, ...]


def _bound_file(root: Path, reference: object) -> Path:
    if not _reference(reference):
        raise ValueError("invalid_reference")
    current = root
    for part in reference["path"].split("/"):
        current = current / part
        # Reject every symlink, including ones that happen to point inside root.
        if stat.S_ISLNK(current.lstat().st_mode):
            raise ValueError("symlink")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("outside_root")
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ValueError("not_regular_file")
    return resolved


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if before.st_size == 0:
            raise ValueError("empty_file")
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if signature(before) != signature(after) or signature(after) != signature(path.stat()):
        raise ValueError("file_changed_during_check")
    return digest.hexdigest()


def verify_file_bindings(root: Path, manifest: dict, skip: tuple[str, ...] = ()) -> FileBindings:
    """Check required artifact bytes without decoding, copying or exposing raw media.

    Call alongside metadata assessment. Even both passing still require review of
    evidence contents and decoded media before admitting a clip. Hashes provide
    integrity, not independent proof that an assertion is true. `skip` names
    references deliberately absent from root (e.g. "original" in an app replay folder).
    """
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("root_not_directory")
    privacy = manifest.get("privacy")
    privacy = privacy if isinstance(privacy, dict) else {}
    references = {
        "original": manifest.get("original"),
        "privacy.filtered": privacy.get("filtered"),
        "privacy.review_evidence": privacy.get("review_evidence"),
    }
    calibration = manifest.get("calibration")
    if isinstance(calibration, dict) and calibration.get("status") == "validated":
        for name in ("file", "heldout_evidence", "scale_measurement"):
            references["calibration." + name] = calibration.get(name)
    checked, reasons, paths = [], [], {}
    for name, reference in references.items():
        if name in skip:
            continue
        try:
            path = _bound_file(root, reference)
            if _digest(path) != reference["sha256"]:
                raise ValueError("hash_mismatch")
            paths[name] = path
            checked.append(name)
        except FileNotFoundError:
            reasons.append(name + ":missing_file")
        except ValueError as error:
            reasons.append(name + ":" + str(error))
        except OSError:
            reasons.append(name + ":unreadable_file")
    raw, filtered = paths.get("original"), paths.get("privacy.filtered")
    if raw is not None and filtered is not None:
        try:
            if raw.samefile(filtered):
                reasons.append("raw_and_filtered_same_file")
        except OSError:
            reasons.append("raw_filtered_identity_unreadable")
    return FileBindings(not reasons, tuple(checked), tuple(reasons))
