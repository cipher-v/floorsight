import cv2
import numpy as np
from scripts.real_clip_perception import background_samples, temporal_median


def test_sequential_calibration_samples_match_seeks_including_duplicates(tmp_path):
    path = str(tmp_path/'sample.avi')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 25, (64, 48))
    assert writer.isOpened()
    for i in range(17):
        writer.write(np.full((48, 64, 3), i*10, np.uint8))
    writer.release()
    cap = cv2.VideoCapture(path)
    try:
        old = background_samples(cap, 17, mode='seek')
        new = background_samples(cap, 17, mode='auto')
        assert len(old) == len(new) == 25
        assert all(np.array_equal(a, b) for a, b in zip(old, new))
    finally:
        cap.release()


def test_compact_median_matches_numpy_without_changing_samples():
    rng = np.random.default_rng(8304)
    for count in [1, 2, 17, 24, 25]:
        frames = rng.integers(0, 256, (count, 48, 64, 3), dtype=np.uint8)
        original = frames.copy()
        assert np.array_equal(temporal_median(frames), temporal_median(frames, mode='numpy'))
        assert np.array_equal(frames, original)
