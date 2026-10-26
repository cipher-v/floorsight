"""Forklift interlock (24 Oct): the machine slows or stops the truck by itself; people review afterwards."""

from __future__ import annotations

import math

from floorsight.contracts import FloorAlert, FloorHazards, Track, TrackClass, TrackState, WorldState
from floorsight.interlock import RESUME_CLEAR_S, ForkliftInterlock, stopping_distance
from floorsight.sim.live import FPS, ClosedLoopSim


def _track(tid, cls, x, y, vx=0.0, vy=0.0, t=0.0):
    return Track(track_id=tid, cls=cls, x=x, y=y, vx=vx, vy=vy, sigma_m=0.1, camera_ids=["cam1"], first_seen_t=0.0,
                 last_seen_t=t, state=TrackState.seen)


def _world(t, tracks, hazards=None):
    return WorldState(t=t, tracks=tracks, cameras=[], coverage_version=0, pipeline_latency_ms=0.0, hazards=hazards)


# ------------------------------------------------------------------ floor-stage input (real clips)
def _alert(level, ttc=2.0, gap=4.0):
    return FloorAlert(person_id="P1", vehicle_id="F1", level=level, confirmed=True, streak=3, clearance_m={"median": gap},
                      ttc_s={"median": ttc}, vehicle_moving=True, reasons=["in the stopping zone"] if level == "danger" else [])


def test_a_confirmed_floor_warning_slows_the_forklift_at_once_and_says_where_it_would_stop():
    lock = ForkliftInterlock()
    fork = _track("F1", TrackClass.forklift, 0.0, 0.0, vx=2.0)
    cmds = lock.update(_world(1.0, [fork], FloorHazards(alerts=[_alert("warning", ttc=2.2, gap=4.2)])))
    assert [c["command"] for c in cmds] == ["slow"]
    c = cmds[0]
    assert abs(c["stoppingDistance_m"] - stopping_distance(2.0)) < 0.006
    assert abs(c["stopsShortBy_m"] - (4.2 - stopping_distance(2.0))) < 0.006


def test_danger_stops_and_the_truck_resumes_only_after_the_person_is_seen_clear():
    # The floor stage only warns about a moving truck, so its alerts stop the moment the truck stands: an empty alert
    # list is not "clear". The truck waits until the person is seen out of its path (audit C1, 27 Oct).
    lock = ForkliftInterlock()
    fork = _track("F1", TrackClass.forklift, 0.0, 0.0, vx=2.0)
    in_path = _track("P1", TrackClass.person, 4.0, 0.0)
    assert [c["command"] for c in lock.update(_world(1.0, [fork, in_path], FloorHazards(alerts=[_alert("danger", gap=0.6)])))] == ["stop"]
    stood = _track("F1", TrackClass.forklift, 0.0, 0.0)
    for t in (1.5, 2.5, 3.5, 6.0):                                                       # alerts gone, person still in the path
        assert lock.update(_world(t, [stood, in_path], FloorHazards())) == []
    clear = _track("P1", TrackClass.person, 4.0, 3.0)                                     # stepped 3 m aside
    assert lock.update(_world(6.5, [stood, clear], FloorHazards())) == []                # first clear moment
    assert lock.update(_world(6.5 + RESUME_CLEAR_S / 2, [stood, clear], FloorHazards())) == []  # not clear long enough
    cmds = lock.update(_world(6.5 + RESUME_CLEAR_S + 0.05, [stood, clear], FloorHazards()))
    assert [c["command"] for c in cmds] == ["resume"]


def test_a_floor_stage_stop_waits_for_a_person_last_seen_in_the_path_who_drops_out_of_view():
    lock = ForkliftInterlock()
    fork = _track("F1", TrackClass.forklift, 0.0, 0.0, vx=2.0)
    in_path = _track("P1", TrackClass.person, 4.0, 0.0)
    assert [c["command"] for c in lock.update(_world(1.0, [fork, in_path], FloorHazards(alerts=[_alert("danger", gap=0.6)])))] == ["stop"]
    stood = _track("F1", TrackClass.forklift, 0.0, 0.0)
    for k in range(1, 40):                                                               # a detector dropout: no person track
        assert lock.update(_world(1.0 + 0.25 * k, [stood], FloorHazards())) == []
    assert lock.vehicles["F1"]["state"] == "stop"


def test_a_floor_stage_stop_for_a_person_not_on_the_floor_map_stays_latched():
    lock = ForkliftInterlock()
    fork = _track("F1", TrackClass.forklift, 0.0, 0.0, vx=2.0)
    assert [c["command"] for c in lock.update(_world(1.0, [fork], FloorHazards(alerts=[_alert("danger", gap=0.6)])))] == ["stop"]
    for k in range(1, 40):
        assert lock.update(_world(1.0 + 0.25 * k, [fork], FloorHazards())) == []
    assert lock.vehicles["F1"]["state"] == "stop"


def test_a_floor_stage_slow_still_returns_to_normal_once_the_warning_is_gone():
    lock = ForkliftInterlock()
    fork = _track("F1", TrackClass.forklift, 0.0, 0.0, vx=2.0)
    person = _track("P1", TrackClass.person, 5.0, 1.0)
    assert [c["command"] for c in lock.update(_world(1.0, [fork, person], FloorHazards(alerts=[_alert("warning", ttc=2.2, gap=4.2)])))] == ["slow"]
    assert lock.update(_world(1.5, [fork, person], FloorHazards())) == []
    cmds = lock.update(_world(1.5 + RESUME_CLEAR_S + 0.05, [fork, person], FloorHazards()))
    assert [c["command"] for c in cmds] == ["resume"]


def test_an_unconfirmed_warning_does_nothing():
    lock = ForkliftInterlock()
    fork = _track("F1", TrackClass.forklift, 0.0, 0.0, vx=2.0)
    a = _alert("warning")
    a.confirmed = False
    assert lock.update(_world(1.0, [fork], FloorHazards(alerts=[a]))) == []


# ------------------------------------------------------------------ track input (simulated warehouse)
def test_a_person_standing_in_the_path_stops_the_forklift_after_two_ticks():
    lock = ForkliftInterlock()
    out = []
    for k in range(4):
        t = k / FPS
        fork = _track("F1", TrackClass.forklift, 0.0 + 2.0 * t, 5.0, vx=2.0, t=t)
        person = _track("P1", TrackClass.person, 3.5, 5.0, t=t)
        out += lock.update(_world(t, [fork, person]))
    assert [c["command"] for c in out] == ["stop"]
    assert out[0]["t"] >= 0.1


def test_a_person_walking_alongside_beyond_08_m_causes_nothing():
    lock = ForkliftInterlock()
    out = []
    for k in range(30):
        t = k / FPS
        fork = _track("F1", TrackClass.forklift, 2.0 * t, 1.2, vx=2.0, t=t)
        person = _track("P1", TrackClass.person, 6.0 - 1.1 * t, 2.9, vx=-1.1, t=t)
        out += lock.update(_world(t, [fork, person]))
    assert out == []


def test_a_person_walking_into_a_forklift_that_drives_away_gets_a_horn_not_a_stop():
    lock = ForkliftInterlock()
    out = []
    for k in range(6):
        t = k / FPS
        fork = _track("F1", TrackClass.forklift, 5.0 - 2.0 * t, 1.2, vx=-2.0, t=t)  # driving west, away
        person = _track("P1", TrackClass.person, 5.6, 2.6 - 1.4 * t, vy=-1.4, t=t)  # walking into its back half
        out += lock.update(_world(t, [fork, person]))
    assert out and out[0]["command"] == "horn"
    assert not any(c["command"] == "stop" for c in out)


# ------------------------------------------------------------------ closed loop (true positions, no rendering)
def _oracle_run(enabled: bool, duration: float = 40.0):
    sim = ClosedLoopSim(render=False)
    lock = ForkliftInterlock(enabled=enabled)
    prev, dt = {}, 1.0 / FPS
    for i in range(int(duration * FPS)):
        t = i * dt
        st = sim.advance(t, dt)
        f = st["forklift"]
        tracks = [_track("F1", TrackClass.forklift, f["x"], f["y"], f["v"] * math.cos(f["heading"]), f["v"] * math.sin(f["heading"]), t)]
        for pid, p in st["people"].items():
            px, py = prev.get(pid, (p["x"], p["y"]))
            tracks.append(_track(pid, TrackClass.person, p["x"], p["y"], (p["x"] - px) * FPS, (p["y"] - py) * FPS, t))
            prev[pid] = (p["x"], p["y"])
        for c in lock.update(_world(t, tracks)):
            sim.fork.command(c, t)
    return sim.truth.summary(sim.walkers), lock


def test_closed_loop_the_interlock_prevents_both_in_path_contacts():
    off, _ = _oracle_run(False)
    on, lock = _oracle_run(True)
    assert off["S1 in path"]["contact"] and off["S2 crossing"]["contact"]
    assert not on["S1 in path"]["contact"] and not on["S2 crossing"]["contact"]
    assert not any(c["person"] == "w3" and c["command"] in ("slow", "stop") for c in lock.log)  # no needless stop for S3
    first_w4 = next(c for c in lock.log if c["person"] == "w4")
    assert first_w4["command"] == "horn"  # walking into a truck that drives away: warn the person


def test_a_stopped_truck_found_again_under_a_new_track_id_stays_stopped():
    """The vision can lose a standing truck and find it again as a new track: absent is not clear, and the new track is
    the same truck (interlock-sim-03: the brakes were released while the picker still stood in the path)."""
    lock = ForkliftInterlock()
    out = []
    for k in range(4):  # T1 drives at the person standing in its path -> stop
        t = k / FPS
        out += lock.update(_world(t, [_track("T1", TrackClass.forklift, 2.0 * t, 5.0, vx=2.0, t=t), _track("P1", TrackClass.person, 3.5, 5.0, t=t)]))
    assert [c["command"] for c in out] == ["stop"]
    for k in range(4, 12):  # the truck fades out of the tracker; the person is still there
        t = k / FPS
        out += lock.update(_world(t, [_track("P1", TrackClass.person, 3.5, 5.0, t=t)]))
    for k in range(12, 60):  # found again, standing, under a new id
        t = k / FPS
        out += lock.update(_world(t, [_track("T9", TrackClass.forklift, 0.7, 5.0, t=t), _track("P1", TrackClass.person, 3.5, 5.0, t=t)]))
    assert not any(c["command"] == "resume" for c in out)
    assert lock.vehicles[out[0]["vehicle"]]["state"] == "stop"


def test_a_truck_stopped_for_a_person_waits_until_that_person_is_seen_clear():
    """'No longer visible' is not 'gone' (interlock-sim-04): a picker standing still faded from the background model and
    the truck drove on. Now it resumes only after the person is seen again out of its path."""
    lock = ForkliftInterlock()
    out = []
    fork_x = 0.0
    for k in range(4):  # driving at a person standing in the path -> stop
        t = k / FPS
        fork_x = 2.0 * t
        out += lock.update(_world(t, [_track("T1", TrackClass.forklift, fork_x, 5.0, vx=2.0, t=t), _track("P1", TrackClass.person, 3.5, 5.0, t=t)]))
    assert [c["command"] for c in out] == ["stop"]
    for k in range(4, 80):  # the person fades from the tracker (still standing there, unseen) for 7.6 s
        t = k / FPS
        out += lock.update(_world(t, [_track("T1", TrackClass.forklift, fork_x, 5.0, t=t)]))
    assert not any(c["command"] == "resume" for c in out)
    vid = out[0]["vehicle"]
    assert lock.vehicles[vid]["waiting"]
    for k in range(80, 120):  # seen again, stepped back 1.4 m to the rack face, 0.85 m off the path
        t = k / FPS
        out += lock.update(_world(t, [_track("T1", TrackClass.forklift, fork_x, 5.0, t=t), _track("P7", TrackClass.person, 3.5, 6.4, t=t)]))
    assert any(c["command"] == "resume" for c in out)


def test_a_person_last_seen_walking_out_of_the_path_need_not_be_seen_again():
    """interlock-sim-05: the truck waited 35 s for a person who had crossed and then stood still out of view."""
    lock = ForkliftInterlock()
    out = []
    for k in range(4):  # a person crossing the lane ahead -> stop
        t = k / FPS
        out += lock.update(_world(t, [_track("T1", TrackClass.forklift, 2.0 * t, 5.0, vx=2.0, t=t),
                                      _track("P1", TrackClass.person, 3.0, 5.6 - 1.4 * t, vy=-1.4, t=t)]))
    assert out and out[-1]["command"] == "stop"
    y = 5.6 - 1.4 * 0.3
    for k in range(4, 18):  # seen walking on, out of the lane, until about 1 m clear of it
        t = k / FPS
        y -= 1.4 / FPS
        out += lock.update(_world(t, [_track("T1", TrackClass.forklift, 0.6, 5.0, t=t), _track("P1", TrackClass.person, 3.0, y, vy=-1.4, t=t)]))
    for k in range(18, 70):  # then out of view
        t = k / FPS
        out += lock.update(_world(t, [_track("T1", TrackClass.forklift, 0.6, 5.0, t=t)]))
    assert any(c["command"] == "resume" for c in out)
