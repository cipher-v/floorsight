"""Privacy filter on a real face photo from the OpenCV samples repo.

The image is downloaded once into data/test-data/ (gitignored). The simulated
warehouse footage has no real faces (people are rendered as coloured shapes),
so it can only prove the filter does not invent faces there.
"""

from __future__ import annotations

import os
import threading
import urllib.request

import cv2
import numpy as np
import pytest

from floorsight.edge import privacy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DATA = os.path.join(ROOT, "data", "test-data")
SAMPLES = {
    "messi5.jpg": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/messi5.jpg",
    "lena.jpg": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg",
}


def sample(name: str) -> np.ndarray:
    path = os.path.join(TEST_DATA, name)
    if not os.path.exists(path):
        os.makedirs(TEST_DATA, exist_ok=True)
        try:
            urllib.request.urlretrieve(SAMPLES[name], path)
        except Exception as e:  # offline
            pytest.skip(f"cannot download {name}: {e}")
    img = cv2.imread(path)
    assert img is not None
    return img


pytestmark = pytest.mark.skipif(not os.path.exists(privacy.DEFAULT_MODEL), reason="YuNet model missing")


@pytest.mark.parametrize("name", ["messi5.jpg", "lena.jpg"])
def test_faces_found_and_destroyed(name):
    img = sample(name)
    faces = privacy.get_detector().detect(img)
    assert len(faces) >= 1
    out, n = privacy.blur_faces(img)
    assert n == len(faces)
    assert out.shape == img.shape and out is not img

    for (x, y, w, h), _ in faces:
        a = cv2.cvtColor(img[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY).astype(np.float64)
        b = cv2.cvtColor(out[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY).astype(np.float64)
        # Strong change in the face box
        assert np.abs(a - b).mean() > 15
        # Fine detail is gone: edge energy drops by more than 95 %
        lap_a = cv2.Laplacian(a, cv2.CV_64F).var()
        lap_b = cv2.Laplacian(b, cv2.CV_64F).var()
        assert lap_b < 0.05 * lap_a

    # The detector itself can no longer find a face in the filtered frame
    assert privacy.get_detector().detect(out) == []


def test_outside_faces_untouched():
    img = sample("messi5.jpg")
    out, _ = privacy.blur_faces(img)
    h, w = img.shape[:2]
    # bottom quarter of the picture (pitch, ball) holds no face
    assert np.array_equal(img[3 * h // 4 :], out[3 * h // 4 :])


def test_input_not_modified():
    img = sample("messi5.jpg")
    before = img.copy()
    privacy.blur_faces(img)
    assert np.array_equal(img, before)


def test_detector_input_follows_frame_size():
    img = sample("messi5.jpg")
    for size in [(640, 360), (1280, 720), (548, 342)]:
        n = privacy.blur_faces(cv2.resize(img, size))[1]
        assert n >= 1


def test_thread_safe_singleton():
    img = sample("messi5.jpg")
    results = []

    def work():
        for _ in range(3):
            results.append(privacy.blur_faces(img)[1])

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [1] * 12
    assert privacy.get_detector() is privacy.get_detector()


def test_people_heads_fallback():
    img = np.full((200, 200, 3), 128, np.uint8)
    cv2.randu(img, 0, 255)
    out, n = privacy.blur_people_heads(img, [(50, 20, 60, 160)])
    assert n == 1
    head = out[20 : 20 + 40, 50:110].astype(float)
    assert cv2.Laplacian(head, cv2.CV_64F).var() < 0.05 * cv2.Laplacian(img[20:60, 50:110].astype(float), cv2.CV_64F).var()
    assert np.array_equal(out[150:], img[150:])  # feet untouched


def test_no_faces_in_simulated_footage():
    path = os.path.join(ROOT, "data", "sim", "bench_dev", "cam1.mp4")
    if not os.path.exists(path):
        pytest.skip("simulated footage not rendered")
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    assert ok
    out, n = privacy.blur_faces(frame)
    assert n == 0 and np.array_equal(out, frame)
