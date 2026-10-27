"""Concurrent inference preserves all stage results and drains failures."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import pytest
from scripts.open_detectors import OpenDetectors


def detector(workers):
    d = OpenDetectors.__new__(OpenDetectors)
    d._pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    d.timing = {'person_s': 0., 'privacy_extra_cv2dnn_s': 0.}
    return d


@pytest.mark.parametrize('workers', [1, 2])
def test_every_detector_result_is_preserved(workers):
    d = detector(workers)
    try:
        tasks = {'person_s': (lambda: ['person', 'vehicle'], ()),
                 'privacy_extra_cv2dnn_s': (lambda: ['additional person'], ())}
        assert d._run_tasks(tasks) == {'person_s': ['person', 'vehicle'],
                                     'privacy_extra_cv2dnn_s': ['additional person']}
        assert all(t >= 0 for t in d.timing.values())
    finally:
        d.close()


def test_failed_privacy_model_waits_for_other_inflight_model():
    d = detector(2)
    together = threading.Barrier(2)
    completed = threading.Event()
    def privacy():
        together.wait(timeout=2)
        raise RuntimeError('privacy model failed')
    def person():
        together.wait(timeout=2)
        time.sleep(.02)
        completed.set()
        return []
    try:
        with pytest.raises(RuntimeError, match='privacy model failed'):
            d._run_tasks({'privacy_extra_cv2dnn_s': (privacy, ()), 'person_s': (person, ())})
        assert completed.is_set()  # next frame cannot race this frame's model state
        assert d._run_tasks({'person_s': (lambda: ['next frame'], ())})['person_s'] == ['next frame']
    finally:
        d.close()
