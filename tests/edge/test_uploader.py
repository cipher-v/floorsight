"""Uploader against a local stub HTTP server on 127.0.0.1 only (no real network)."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from floorsight.contracts import FrameMeta
from floorsight.edge.uploader import META_HEADER, EdgeUploader
from floorsight.interfaces import FramePacket


class Stub:
    def __init__(self, script=None, delay_s=0.0):
        self.script = list(script or [])  # status codes to return first, then 200
        self.delay_s = delay_s
        self.received: list[tuple[dict, np.ndarray, dict]] = []
        self.lock = threading.Lock()
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if stub.delay_s:
                    time.sleep(stub.delay_s)
                with stub.lock:
                    code = stub.script.pop(0) if stub.script else 200
                    if code == 200:
                        meta = json.loads(self.headers[META_HEADER])
                        img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
                        stub.received.append((meta, img, dict(self.headers)))
                self.send_response(code)
                if code in (429, 503):
                    self.send_header("Retry-After", "0")
                self.send_header("Content-Length", "0")
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/ingest"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def frame():
    img = np.zeros((90, 160, 3), np.uint8)
    cv2.rectangle(img, (20, 20), (80, 70), (0, 200, 255), -1)
    return img


def packet(img, i, blur=True):
    return FramePacket(FrameMeta(camera_id="cam1", frame_index=i, t=i / 10, width=160, height=90, faces_blurred=0, blur_applied=blur), img)


def test_posts_jpeg_and_meta(frame):
    stub = Stub()
    up = EdgeUploader(stub.url, auth_token="tok")
    for i in range(3):
        up.submit(packet(frame, i))
    assert up.flush(5)
    up.close()
    stub.close()
    assert up.stats.sent == 3
    metas = sorted(m["frame_index"] for m, _, _ in stub.received)
    assert metas == [0, 1, 2]
    meta, img, headers = stub.received[0]
    FrameMeta.model_validate(meta)
    assert img.shape == frame.shape
    assert headers["Authorization"] == "Bearer tok"
    assert headers["Content-Type"] == "image/jpeg"


def test_retries_then_succeeds(frame):
    stub = Stub(script=[503, 500])
    up = EdgeUploader(stub.url, backoff_s=0.01, max_retries=3)
    up.submit(packet(frame, 0))
    assert up.flush(5)
    up.close()
    stub.close()
    assert up.stats.sent == 1 and up.stats.retries == 2


def test_gives_up_after_retries(frame):
    stub = Stub(script=[500] * 10)
    up = EdgeUploader(stub.url, backoff_s=0.01, max_retries=2)
    up.submit(packet(frame, 0))
    assert up.flush(5)
    up.close()
    stub.close()
    assert up.stats.sent == 0 and up.stats.dropped_after_retries == 1


def test_bad_request_not_retried(frame):
    stub = Stub(script=[400])
    up = EdgeUploader(stub.url, backoff_s=0.01)
    up.submit(packet(frame, 0))
    assert up.flush(5)
    up.close()
    stub.close()
    assert up.stats.dropped_rejected == 1 and up.stats.retries == 0


def test_drops_oldest_when_link_is_slow(frame):
    stub = Stub(delay_s=0.05)
    up = EdgeUploader(stub.url, max_queue=2)
    for i in range(20):
        up.submit(packet(frame, i))
    assert up.flush(10)
    up.close()
    stub.close()
    got = [m["frame_index"] for m, _, _ in stub.received]
    assert up.stats.dropped_backpressure >= 10
    assert up.stats.sent + up.stats.dropped_backpressure == 20
    assert got[-1] == 19  # newest frame always arrives
    assert got == sorted(got)


def test_unreachable_server_does_not_block(frame):
    up = EdgeUploader("http://127.0.0.1:9/ingest", max_queue=2, backoff_s=0.01, max_retries=0, timeout_s=0.5)
    t0 = time.monotonic()
    for i in range(50):
        up.submit(packet(frame, i))
    assert time.monotonic() - t0 < 1.0  # submit never waits on the network
    up.flush(10)
    up.close()
    assert up.stats.sent == 0


def test_refuses_unblurred(frame):
    stub = Stub()
    up = EdgeUploader(stub.url)
    with pytest.raises(ValueError):
        up.submit(packet(frame, 0, blur=False))
    up.close()
    stub.close()


def test_submit_raw_blurs_first(frame):
    from floorsight.edge import privacy

    if not __import__("os").path.exists(privacy.DEFAULT_MODEL):
        pytest.skip("YuNet model missing")
    stub = Stub()
    up = EdgeUploader(stub.url)
    meta = up.submit_raw(frame, "cam1", 5, 0.5)
    assert up.flush(5)
    up.close()
    stub.close()
    assert meta.blur_applied and meta.faces_blurred == 0
    assert stub.received[0][0]["faces_blurred"] == 0
