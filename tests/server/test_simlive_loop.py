import pytest

from floorsight.sim.live import SimLiveSource


def meta(packets):
    return next(iter(packets.values())).meta


def test_loop_restarts_frames_and_clears_braking_and_truth():
    source = SimLiveSource(duration_s=0.3, loop=True)
    frames = iter(source)
    first = next(frames)
    original = source.sim
    source.command({"command": "stop", "vehicle": "truck", "t": -1})
    assert meta(next(frames)).frame_index == 1
    assert source.sim.fork.mode == "stop"
    assert meta(next(frames)).frame_index == 2
    wrapped = next(frames)
    assert meta(wrapped).frame_index == meta(first).frame_index == 0
    assert meta(wrapped).t == 0
    assert source.sim is not original
    assert source.sim.fork.mode == "normal"
    assert not source.sim.fork.requests
    assert all(len(gaps) == 1 for gaps in source.sim.truth.gaps.values())
    source.close()
    with pytest.raises(StopIteration):
        next(frames)


def test_default_simulation_still_ends():
    assert [meta(p).frame_index for p in SimLiveSource(duration_s=0.3)] == [0, 1, 2]


def test_loop_rejects_a_duration_without_a_detectable_frame_reset():
    with pytest.raises(ValueError, match="at least two frames"):
        SimLiveSource(duration_s=0.1, loop=True)
