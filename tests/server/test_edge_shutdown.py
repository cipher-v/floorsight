"""A full edge queue must not leave a producer blocked after shutdown."""
import threading

import pytest

from floorsight.server.runtime import Runtime


def test_stop_unblocks_full_edge_queue():
    rt = object.__new__(Runtime)
    rt._stop = threading.Event()
    rt._edge_blur = lambda packets: packets
    third = threading.Event()

    def source():
        yield {"frame": 1}
        yield {"frame": 2}
        third.set()
        yield {"frame": 3}

    rt.source = source()
    stream = rt._edge_stage(depth=1)
    assert next(stream) == {"frame": 1}
    assert third.wait(2)
    rt._stop.set()
    stream.close()
    rt._edge_thread.join(2)
    assert not rt._edge_thread.is_alive()


def test_edge_source_error_reaches_consumer():
    rt = object.__new__(Runtime)
    rt._stop = threading.Event()
    rt._edge_blur = lambda packets: packets

    def source():
        yield {"frame": 1}
        raise RuntimeError("decoder failed")

    rt.source = source()
    stream = rt._edge_stage(depth=1)
    assert next(stream) == {"frame": 1}
    with pytest.raises(RuntimeError, match="decoder failed"):
        next(stream)
    rt._stop.set()
    rt._edge_thread.join(2)
    assert not rt._edge_thread.is_alive()


def test_feedback_source_waits_for_command_before_advancing():
    class Source:
        feedback_driven = True

        def __init__(self):
            self.command = "moving"

        def __iter__(self):
            yield {"frame": self.command}
            # In a prefetched source this second frame would already be moving.
            yield {"frame": self.command}

    rt = object.__new__(Runtime)
    rt._stop = threading.Event()
    rt._edge_blur = lambda packets: packets
    rt.source = Source()
    stream = rt._edge_stage()
    assert next(stream) == {"frame": "moving"}
    rt.source.command = "stopped"
    assert next(stream) == {"frame": "stopped"}
    with pytest.raises(StopIteration):
        next(stream)


def test_shutdown_does_not_close_a_decoder_still_in_use():
    rt = object.__new__(Runtime)
    rt._stop = threading.Event()
    rt._thread = None
    rt._blur_pool = None
    rt.vision = object()
    rt.source = object()
    rt._close_history_file = lambda: None
    closed = []
    rt._close_source = lambda: closed.append(True)
    release = threading.Event()
    rt._edge_thread = threading.Thread(target=lambda: release.wait(2))
    rt._edge_thread.start()
    try:
        rt.stop(timeout=0.01)
        assert not closed
    finally:
        release.set()
        rt._edge_thread.join(2)
    rt.stop()
    assert closed == [True]
