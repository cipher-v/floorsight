"""AWS adapters for the project protocols.

    EvidenceStore -> S3EvidenceStore      (floorsight.cloud.s3)
    Notifier      -> SnsNotifier          (floorsight.cloud.sns)
    world sink    -> DynamoTrackSink      (floorsight.cloud.dynamo)
    metrics       -> CloudWatchMetrics    (floorsight.cloud.metrics)

boto3 is imported lazily, so the rest of Floorsight runs without it. On the EC2
instance, credentials come from the instance role (no keys on disk). Settings
come from environment variables written by the Terraform user-data into
/etc/floorsight.env; see `from_env()`. The server only builds them when
FLOORSIGHT_CLOUD=aws (`server_adapters()`), so tests and local runs never call AWS.
"""

from __future__ import annotations

import os


def from_env(env: dict | None = None) -> dict:
    """Build whichever adapters are configured. Keys of the result: evidence, notifier, tracks, metrics."""
    env = dict(os.environ if env is None else env)
    region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
    out: dict = {}
    if env.get("FLOORSIGHT_S3_BUCKET"):
        from floorsight.cloud.s3 import S3EvidenceStore

        out["evidence"] = S3EvidenceStore(env["FLOORSIGHT_S3_BUCKET"], prefix=env.get("FLOORSIGHT_S3_PREFIX", "evidence/"), region=region)
    if env.get("FLOORSIGHT_SNS_TOPIC_ARN"):
        from floorsight.cloud.sns import SnsNotifier

        # every agent alert is published (they are created with channel "dashboard"); set the variable to narrow it
        out["notifier"] = SnsNotifier(env["FLOORSIGHT_SNS_TOPIC_ARN"], region=region, only_channel=env.get("FLOORSIGHT_SNS_ONLY_CHANNEL") or None)
    if env.get("FLOORSIGHT_DDB_TABLE"):
        from floorsight.cloud.dynamo import DynamoTrackSink

        out["tracks"] = DynamoTrackSink(
            env["FLOORSIGHT_DDB_TABLE"],
            session_id=env.get("FLOORSIGHT_SESSION_ID", "default"),
            ttl_s=float(env.get("FLOORSIGHT_DDB_TTL_S", 86400)),
            region=region,
        )
    if env.get("FLOORSIGHT_CW_NAMESPACE"):
        from floorsight.cloud.metrics import CloudWatchMetrics

        out["metrics"] = CloudWatchMetrics(namespace=env["FLOORSIGHT_CW_NAMESPACE"], site_id=env.get("FLOORSIGHT_SITE_ID", "demo"), region=region)
    return out


def server_adapters(env: dict | None = None) -> dict:
    """Adapters for the server: empty unless FLOORSIGHT_CLOUD=aws."""
    env = dict(os.environ if env is None else env)
    if env.get("FLOORSIGHT_CLOUD", "").strip().lower() != "aws":
        return {}
    return from_env(env)


class MirrorEvidenceStore:
    """Local store first (the dashboard and the trace read it), then the same JPEG bytes to S3 under the same ref.

    An S3 failure never loses evidence: the local copy stays, and the error is counted for the health view."""

    def __init__(self, local, s3):
        self.local, self.s3 = local, s3
        self.mirrored, self.errors, self.last_error = 0, 0, None

    def put(self, image, label: str) -> str:
        ref = self.local.put(image, label)
        data = self.local.get(ref)
        if data is not None:
            try:
                self.s3.put_bytes(ref, data)
                self.mirrored += 1
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                self.errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
        return ref

    def get(self, ref: str):
        data = self.local.get(ref)
        return data if data is not None else self.s3.get(ref)

    def __getattr__(self, name):  # purge(), root, ... stay the local store's
        return getattr(self.local, name)
