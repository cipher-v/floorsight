"""GCP-only control-flow tests; no image processing or detector accuracy claim."""
import json
import threading
from types import SimpleNamespace

import pytest

from floorsight.contracts import CameraStatus
from floorsight.vision import pipeline


@pytest.mark.parametrize("error", [float("nan"), float("inf"), -float("inf")])
def test_recalibration_rejects_nonfinite_error(monkeypatch, error):
    cam = SimpleNamespace(last_image=object(), status=CameraStatus.ok,
                          check_obs=[], info=object())
    vision = object.__new__(pipeline.RealVision)
    vision.lock = threading.RLock()
    vision.cams = {"cam1": cam}
    vision.t = 10.0
    vision.floor_corners = {}
    vision.markers = SimpleNamespace(detect=lambda *a, **kw: {})
    monkeypatch.setattr(pipeline, "fit_calibration",
                        lambda *a, **kw: SimpleNamespace(error_m=error))
    result = vision.recalibrate("cam1")
    assert result["ok"] is False
    assert result["calib_error_m"] is None
    assert cam.status == CameraStatus.needs_human
    json.dumps(result, allow_nan=False)


def test_nonfinite_drift_does_not_clear_suspicion():
    cam = SimpleNamespace(check_obs=[], calib=SimpleNamespace(
        floor_error=lambda *args: float("nan")), bad_streak=1,
        status=CameraStatus.ok)
    vision = object.__new__(pipeline.RealVision)
    vision.floor_corners = {}
    vision.markers = SimpleNamespace(detect=lambda image: {1: object()})
    vision._drift_check(cam, object(), 10.0)
    assert cam.bad_streak == 2
    assert cam.next_check_t == 10.0 + pipeline.RECHECK_S


def test_startup_does_not_install_nonfinite_calibration(monkeypatch):
    cam = SimpleNamespace(frames=pipeline.CALIB_FRAMES, calib_hist=[],
                          calib=None, info=SimpleNamespace(camera_id="cam1"), plate=None)
    vision = object.__new__(pipeline.RealVision)
    vision.floor_corners = {}
    vision.cams = {"cam1": cam}
    vision.markers = SimpleNamespace(detect=lambda *a, **kw: {})
    monkeypatch.setattr(pipeline, "fit_calibration",
                        lambda *a, **kw: SimpleNamespace(error_m=float("nan")))
    vision._calibration_frame(cam, object(), 10.0)
    assert cam.calib is None
    assert cam.status == CameraStatus.needs_human
