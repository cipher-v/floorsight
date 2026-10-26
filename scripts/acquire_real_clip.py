"""Acquire ONE pinned source into private GCP quarantine. UNTESTED preparation.

Run only through the coordinated lease runner after renewal and test validation.
No cleanup, overwrite, retries, media decoding, privacy approval or publishing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


USER_AGENT = "floorsight-research-intake/1.0 (private CC BY 4.0 dataset evaluation)"


class HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise ValueError("non_https_redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def metadata(field: str) -> str:
    request = Request("http://metadata.google.internal/computeMetadata/v1/" + field,
                      headers={"Metadata-Flavor": "Google"})
    with build_opener(ProxyHandler({})).open(request, timeout=3) as response:
        if response.headers.get("Metadata-Flavor") != "Google":
            raise ValueError("gcp_metadata_header_missing")
        return response.read(1024).decode("utf-8").strip()


def validate_lease(lease: dict) -> dict:
    deadline = datetime.fromisoformat(lease["validUntilUtc"].replace("Z", "+00:00"))
    if (deadline.tzinfo is None or deadline <= datetime.now(timezone.utc)
            or lease.get("status") != "ready"):
        raise ValueError("lease_expired_or_not_ready")
    actual = {"project": metadata("project/project-id"),
              "instance": metadata("instance/name"),
              "zone": metadata("instance/zone").rsplit("/", 1)[-1]}
    if actual["project"] != "peakai-pipeline-dev" or any(
            actual[key] != lease.get(key) for key in actual):
        raise ValueError("lease_resource_mismatch")
    if lease.get("threadId") != "0c0b53bd-0cb1-410d-93e6-ae66eb8fac24":
        raise ValueError("lease_thread_mismatch")
    return actual


def transfer(url: str, target: Path, expected: str, limit: int, evidence: dict,
             deadline_utc: datetime) -> None:
    digest, count = hashlib.sha256(), 0
    started = time.monotonic()
    # The publisher's CDN returns 403 for the default Python-urllib agent (GCP run
    # 20261019T060736Z, preserved); identify the intake honestly instead.
    request = Request(url, headers={"Accept-Encoding": "identity", "User-Agent": USER_AGENT})
    with build_opener(HTTPSRedirects()).open(request, timeout=15) as response:
        evidence.update(final_response_url=response.geturl(), response_status=response.status)
        if response.status != 200 or urlsplit(response.geturl()).scheme != "https":
            raise ValueError("unexpected_response")
        length = response.headers.get("Content-Length")
        length = int(length) if length is not None else None
        if length is not None and (length <= 0 or length > limit):
            raise ValueError("invalid_content_length")
        if response.headers.get_content_type() in ("text/html", "application/json"):
            raise ValueError("non_media_response")
        with target.open("xb") as output:
            while True:
                if time.monotonic() - started > 120 or datetime.now(timezone.utc) >= deadline_utc:
                    raise TimeoutError("transfer_deadline_or_lease_expiry")
                chunk = response.read(min(64 * 1024, limit - count + 1))
                if time.monotonic() - started > 120 or datetime.now(timezone.utc) >= deadline_utc:
                    raise TimeoutError("transfer_deadline_or_lease_expiry")
                if not chunk:
                    break
                if count + len(chunk) > limit:
                    raise ValueError("download_byte_limit")
                output.write(chunk)
                digest.update(chunk)
                count += len(chunk)
                evidence["bytes"] = count
                evidence["partial_sha256"] = digest.hexdigest()
        if count == 0 or (length is not None and count != length):
            raise ValueError("empty_or_incomplete_response")
    evidence["computed_sha256"] = digest.hexdigest()
    evidence["publisher_sha256_match"] = digest.hexdigest() == expected
    if not evidence["publisher_sha256_match"]:
        raise ValueError("publisher_checksum_mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lease = json.loads(args.lease.read_text(encoding="utf-8-sig"))
    # Fail before any filesystem mutation or media processing on the wrong host.
    actual = validate_lease(lease)
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    selected, settings = plan["selection"], plan["execution"]
    expected, url = selected["publisherSha256"], selected["downloadUrl"]
    # Neutral name only: the class-bearing publisher filename never reaches processing.
    neutral_name = selected.get("neutralProcessingName", "clip-dev-001.mp4")
    if (plan.get("schema") != "floorsight.acquisition-plan.v1"
            or not re.fullmatch(r"clip-dev-[0-9]{3}\.mp4", neutral_name)
            or settings.get("project") != actual["project"]
            or settings.get("maximumClips") != 1
            or settings.get("bulkArchiveAllowedForThisJob") is not False
            or plan.get("source", {}).get("license") != "CC-BY-4.0"
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
            or urlsplit(url).scheme != "https"
            or urlsplit(url).hostname != "data.mendeley.com"
            or "/public-files/datasets/" not in urlsplit(url).path):
        raise ValueError("unsupported_acquisition_plan")
    limit = settings["maximumDownloadBytes"]
    if type(limit) is not int or not 0 < limit <= 32 * 1024 * 1024:
        raise ValueError("invalid_byte_limit")
    allowed = Path(lease["outputRoot"]).resolve(strict=True)
    destination = args.output.resolve()
    if destination == allowed or not destination.is_relative_to(allowed):
        raise ValueError("output_outside_lease")
    destination.mkdir(parents=True, exist_ok=False)
    evidence = {"schema": "floorsight.acquisition-result.v1", "actual_resource": actual,
                "startedAt": datetime.now(timezone.utc).isoformat(),
                "source_plan": plan, "status": "started", "bytes": 0,
                "publisher_sha256_match": False, "decode_result": "not_run",
                "privacy_reviewed": False, "review_bundle_eligible": False,
                "receipt_note": "Bind to the enclosing GCP runner execution receipt; this file is not that receipt."}
    try:
        transfer(url, destination / neutral_name, expected, limit, evidence,
                 datetime.fromisoformat(lease["validUntilUtc"].replace("Z", "+00:00")))
        evidence["status"] = "download_hash_verified_only"
        result = 0
    except Exception as error:
        evidence.update(status="failed_preserved", error_type=type(error).__name__, error=str(error))
        result = 1
    finally:
        evidence["finishedAt"] = datetime.now(timezone.utc).isoformat()
        with (destination / "acquisition-result.json").open("x", encoding="utf-8") as output:
            json.dump(evidence, output, indent=2, ensure_ascii=False)
            output.write("\n")
    print(json.dumps({"status": evidence["status"], "output": str(destination)}))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
