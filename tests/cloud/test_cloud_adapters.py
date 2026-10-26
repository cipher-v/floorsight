"""AWS adapters, tested ONLY with botocore Stubber. Any real HTTP send fails the test."""

from __future__ import annotations

import datetime as dt
import json

import cv2
import numpy as np
import pytest

boto3 = pytest.importorskip("boto3")
from botocore.response import StreamingBody  # noqa: E402
from botocore.stub import ANY, Stubber  # noqa: E402

from floorsight.cloud import from_env  # noqa: E402
from floorsight.cloud.dynamo import DynamoTrackSink  # noqa: E402
from floorsight.cloud.metrics import CloudWatchMetrics  # noqa: E402
from floorsight.cloud.s3 import S3EvidenceStore  # noqa: E402
from floorsight.cloud.sns import SnsNotifier  # noqa: E402
from floorsight.contracts import Alert, CameraHealth, CameraStatus, Track, TrackClass, TrackState, WorldState  # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("real network call attempted")

    monkeypatch.setattr("botocore.httpsession.URLLib3Session.send", boom)
    for k in ("AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def client(service):
    from botocore.config import Config

    return boto3.client(
        service, region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing",
        config=Config(signature_version="s3v4") if service == "s3" else None,
    )


def world(t=1.0, n=2):
    return WorldState(
        t=t,
        tracks=[
            Track(track_id=f"T{i}", cls=TrackClass.person, x=1.0 + i, y=2.0, vx=0, vy=0, sigma_m=0.2,
                  camera_ids=["cam1"], first_seen_t=0, last_seen_t=t, state=TrackState.seen)
            for i in range(n)
        ],
        cameras=[CameraHealth(camera_id="cam1", status=CameraStatus.ok, fps=10.0, last_frame_t=t, markers_visible=4, calib_error_m=0.05)],
        coverage_version=1,
        pipeline_latency_ms=42.0,
    )


# ----------------------------------------------------------------------- S3


def test_s3_put_get_roundtrip():
    s3 = client("s3")
    store = S3EvidenceStore("floorsight-evidence-test", client=s3)
    img = np.zeros((40, 60, 3), np.uint8)
    img[:, :30] = 255
    with Stubber(s3) as st:
        st.add_response(
            "put_object",
            {"ETag": '"x"'},
            {"Bucket": "floorsight-evidence-test", "Key": ANY, "Body": ANY, "ContentType": "image/jpeg",
             "ServerSideEncryption": "AES256", "Metadata": {"label": "blocked_exit_cam3", "blurred": "true"}},
        )
        ref = store.put(img, "blocked exit/cam3")
        assert ref.endswith("blocked_exit_cam3")
        jpeg = cv2.imencode(".jpg", img)[1].tobytes()
        import io

        st.add_response(
            "get_object",
            {"Body": StreamingBody(io.BytesIO(jpeg), len(jpeg))},
            {"Bucket": "floorsight-evidence-test", "Key": f"evidence/{ref}.jpg"},
        )
        assert store.get(ref) == jpeg
        st.add_client_error("get_object", service_error_code="NoSuchKey", http_status_code=404)
        assert store.get("missing-ref") is None
        st.assert_no_pending_responses()


def test_s3_rejects_bad_refs_and_presigns_offline():
    s3 = client("s3")
    store = S3EvidenceStore("b", client=s3)
    with Stubber(s3):
        assert store.get("../etc/passwd") is None
    url = store.presigned_url("123-00001-x", expiry_s=60)
    assert "evidence/123-00001-x.jpg" in url
    assert "X-Amz-Expires=60" in url and "X-Amz-Signature=" in url


# ----------------------------------------------------------------------- SNS


def test_sns_publishes_only_sns_channel():
    sns = client("sns")
    arn = "arn:aws:sns:us-east-1:123456789012:floorsight-alerts"
    n = SnsNotifier(arn, client=sns)
    a = Alert(alert_id="A1", event_id="E1", t=80.5, channel="sns", audience="floor_lead", message="Exit B blocked by a pallet")
    with Stubber(sns) as st:
        st.add_response(
            "publish",
            {"MessageId": "m-1"},
            {"TopicArn": arn, "Subject": "[Floorsight] floor_lead: Exit B blocked by a pallet", "Message": ANY, "MessageAttributes": ANY},
        )
        n.send(a)
        n.send(a.model_copy(update={"channel": "dashboard"}))  # ignored, no stubbed call needed
        st.assert_no_pending_responses()
    assert n.sent == ["m-1"]


def test_sns_subject_truncated():
    sns = client("sns")
    n = SnsNotifier("arn:aws:sns:us-east-1:123456789012:t", client=sns)
    a = Alert(alert_id="A", event_id="E", t=0, channel="sns", audience="manager", message="x" * 300)
    with Stubber(sns) as st:
        st.add_response("publish", {"MessageId": "m"}, {"TopicArn": ANY, "Subject": "[Floorsight] manager: " + "x" * 78, "Message": ANY, "MessageAttributes": ANY})
        n.send(a)
        st.assert_no_pending_responses()


# ------------------------------------------------------------------ DynamoDB


def test_dynamo_put_with_ttl_and_throttle():
    ddb = client("dynamodb")
    sink = DynamoTrackSink("floorsight-tracks", session_id="demo_001", ttl_s=3600, client=ddb, clock=lambda: 1_700_000_000)
    w = world(t=12.345)
    expected = {
        "TableName": "floorsight-tracks",
        "Item": {
            "session_id": {"S": "demo_001"},
            "t_ms": {"N": "12345"},
            "world": {"S": json.dumps(w.model_dump(mode="json"), separators=(",", ":"))},
            "n_tracks": {"N": "2"},
            "expires_at": {"N": str(1_700_000_000 + 3600)},
        },
    }
    with Stubber(ddb) as st:
        st.add_response("put_item", {}, expected)
        assert sink.put_world(w) is True
        assert sink.put_world(world(t=12.6)) is False  # < 1 s later: skipped
        st.assert_no_pending_responses()


def test_dynamo_query_paginates():
    ddb = client("dynamodb")
    sink = DynamoTrackSink("floorsight-tracks", session_id="s", client=ddb)
    item = lambda t: {"session_id": {"S": "s"}, "t_ms": {"N": str(int(t * 1000))}, "world": {"S": world(t).model_dump_json()}}  # noqa: E731
    with Stubber(ddb) as st:
        st.add_response("query", {"Items": [item(1.0)], "LastEvaluatedKey": {"session_id": {"S": "s"}, "t_ms": {"N": "1000"}}})
        st.add_response("query", {"Items": [item(2.0)]})
        got = sink.query(0, 5)
    assert [g.t for g in got] == [1.0, 2.0]


# ---------------------------------------------------------------- CloudWatch


def test_cloudwatch_batches_and_flushes():
    cw = client("cloudwatch")
    m = CloudWatchMetrics(namespace="Floorsight", site_id="demo", client=cw, flush_every=1000)
    with Stubber(cw) as st:
        st.add_response("put_metric_data", {}, {"Namespace": "Floorsight", "MetricData": ANY})
        assert m.observe_world(world(t=1.0))
        assert not m.observe_world(world(t=5.0))  # sampled at most every 10 s of session time
        m.record_event("blocked_exit")
        assert m.flush() == 4  # fps(cam1) + latency + tracks + event
        st.assert_no_pending_responses()


def test_cloudwatch_datum_shape():
    cw = client("cloudwatch")
    m = CloudWatchMetrics(client=cw, flush_every=1000)
    m.record_latency(12.5)
    d = m._buf[0]  # noqa: SLF001
    assert d["MetricName"] == "PipelineLatency" and d["Unit"] == "Milliseconds"
    assert d["Dimensions"] == [{"Name": "Site", "Value": "demo"}]
    assert isinstance(d["Timestamp"], dt.datetime)
    with Stubber(cw) as st:
        st.add_response("put_metric_data", {}, {"Namespace": "Floorsight", "MetricData": [d]})
        m.flush()


def test_from_env_builds_configured_adapters(monkeypatch):
    made = from_env({"AWS_REGION": "us-east-1", "FLOORSIGHT_S3_BUCKET": "b", "FLOORSIGHT_CW_NAMESPACE": "Floorsight"})
    assert set(made) == {"evidence", "metrics"}
    assert from_env({}) == {}
