"""The server's AWS wiring: off unless FLOORSIGHT_CLOUD=aws; evidence mirrored to S3 under the same ref; alerts to SNS."""

from __future__ import annotations

import numpy as np

from floorsight.cloud import MirrorEvidenceStore, server_adapters
from floorsight.cloud.s3 import S3EvidenceStore
from floorsight.cloud.sns import SnsNotifier
from floorsight.contracts import Alert
from floorsight.storage import LocalEvidenceStore


class FakeS3:
    def __init__(self, fail=False):
        self.objects, self.fail = {}, fail

    def put_object(self, Bucket, Key, Body, **kw):
        if self.fail:
            raise RuntimeError("s3 down")
        self.objects[(Bucket, Key)] = Body


class FakeSns:
    def __init__(self):
        self.published = []

    def publish(self, **kw):
        self.published.append(kw)
        return {"MessageId": f"m{len(self.published)}"}


def test_adapters_are_off_without_the_switch():
    env = {"FLOORSIGHT_S3_BUCKET": "b", "FLOORSIGHT_SNS_TOPIC_ARN": "arn:aws:sns:us-east-1:1:t", "AWS_REGION": "us-east-1"}
    assert server_adapters(env) == {}


def test_mirror_store_uses_the_same_ref(tmp_path):
    local = LocalEvidenceStore(str(tmp_path), retention_s=3600)
    fake = FakeS3()
    s3 = S3EvidenceStore("bucket", client=fake)
    store = MirrorEvidenceStore(local, s3)
    ref = store.put(np.zeros((8, 8, 3), np.uint8), "crop-camA")
    assert (("bucket", s3.key(ref))) in fake.objects
    assert fake.objects[("bucket", s3.key(ref))] == local.get(ref)
    assert store.get(ref) == local.get(ref) and store.mirrored == 1


def test_s3_failure_keeps_the_local_copy(tmp_path):
    local = LocalEvidenceStore(str(tmp_path), retention_s=3600)
    store = MirrorEvidenceStore(local, S3EvidenceStore("bucket", client=FakeS3(fail=True)))
    ref = store.put(np.zeros((8, 8, 3), np.uint8), "crop")
    assert local.get(ref) is not None and store.errors == 1 and "s3 down" in store.last_error


def test_dashboard_alerts_reach_sns_when_wired_like_the_server():
    fake = FakeSns()
    n = SnsNotifier("arn:aws:sns:us-east-1:1:t", client=fake, only_channel=None)
    n.send(Alert(alert_id="AL-1", event_id="EV-1", t=12.5, channel="dashboard", audience="floor_lead", message="S002:F1 will pass within 0.8 m"))
    assert len(fake.published) == 1 and fake.published[0]["MessageAttributes"]["audience"]["StringValue"] == "floor_lead"
