"""Motion smoothing (v9, 24 Oct): Kalman-filtered floor tracks, sticky decisions, short-gap coasting, person association,
and the parked-object rule. Bugs found reviewing the 23 Oct AWS demo video and its per-frame records.
v10 (24 Oct, second pass over the smoothed video): re-anchoring keeps the speed, a turn-rate filter decides "turning",
and a truck that reverses or pulls away from its parked axis keeps its footprint."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mf = _load("motion_filter")
dzm = _load("danger_zone")
rcp = _load("real_clip_perception")
CAL = ROOT / "tests" / "data" / "floor-calibration-camA.json"


# ------------------------------------------------------------------ Kalman filter
def test_filter_follows_a_steady_forklift_and_removes_most_of_the_jitter():
    rng = np.random.default_rng(1)
    kf = mf.FloorKF(accel_sigma=3.0, meas_sigma=0.35)
    raw_err, filt_err = [], []
    for i in range(75):
        t = i / 25
        truth = np.array([[2.0 * t, 1.0]])
        z = truth + rng.normal(0, 0.35, (1, 2))
        kf.update(t, z)
        if i >= 25:
            raw_err.append(np.linalg.norm(z - truth))
            filt_err.append(np.linalg.norm(kf.pos - truth))
    assert np.mean(filt_err) < 0.6 * np.mean(raw_err)
    assert abs(kf.vel[0, 0] - 2.0) < 0.5 and abs(kf.vel[0, 1]) < 0.5


def test_a_single_frame_jump_of_the_outline_is_ignored():
    kf = mf.FloorKF(accel_sigma=3.0, meas_sigma=0.35)
    for i in range(30):
        kf.update(i / 25, [[0.08 * i, 0.0]])
    before = kf.pos.copy()
    assert kf.update(30 / 25, [[0.08 * 30 + 3.0, 0.0]]) == "outlier"
    assert np.linalg.norm(kf.pos - before) < 0.2


def test_a_real_jump_restarts_the_filter_after_four_frames():
    kf = mf.FloorKF(accel_sigma=3.0, meas_sigma=0.35)
    for i in range(30):
        kf.update(i / 25, [[0.0, 0.0]])
    states = [kf.update((30 + j) / 25, [[5.0, 0.0]]) for j in range(4)]
    assert states == ["outlier", "outlier", "outlier", "reset"]
    assert abs(kf.pos[0, 0] - 5.0) < 1e-9


def test_all_draws_share_one_gain():
    kf = mf.FloorKF(accel_sigma=3.0, meas_sigma=0.35)
    for i in range(20):
        kf.update(i / 25, [[0.1 * i, 0.0], [0.105 * i, 0.0]])  # second draw: 5 % larger floor scale
    assert abs(kf.pos[1, 0] / kf.pos[0, 0] - 1.05) < 0.01


def test_hysteresis_needs_three_frames_each_way():
    h = mf.Hysteresis(0.2, 0.1, 3)
    assert [h.update(v) for v in (0.3, 0.3, 0.05, 0.3, 0.3, 0.3)] == [False, False, False, False, False, True]
    assert [h.update(v) for v in (0.15, 0.05, 0.05, 0.05)] == [True, True, True, False]


# ------------------------------------------------------------------ danger zone with smoothing
def _centre(poly):
    """Area centre (the mean of the corners moves when the corner count changes)."""
    x, y = np.asarray(poly, float).T
    c = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = c.sum() / 2
    return np.array([((x + np.roll(x, -1)) * c).sum(), ((y + np.roll(y, -1)) * c).sum()]) / (6 * a)


def _rect(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


def _drive(smoothing: bool, frames: int = 50, gap=(), xpath=None):
    dzm.SMOOTHING = smoothing
    try:
        dz = dzm.DangerZone(CAL, (1920, 1080), 25.0, draws=8)
        rng = np.random.default_rng(3)
        centres, coasting, headings, turning = [], [], [], []
        for i in range(frames):
            x = 700 + 6 * i if xpath is None else xpath(i)
            j = rng.normal(0, 18, 4)  # the detector's outline jitters by ~20 px each frame
            v = {"id": "F1", "polygons": [_rect(x + j[0], 430 + j[1], x + 380 + j[2], 720 + j[3])], "loadPolygons": [],
                 "moving": True, "estimated": i in gap}
            if i in gap:
                v["polygons"] = []
            out = dz.step(i / 25, [v], [])
            vs = {s["id"]: s for s in out["vehicles"]}
            centres.append(_centre(vs["F1"]["footprint"]) if "F1" in vs else None)
            coasting.append(vs["F1"]["coasting"] if "F1" in vs else None)
            headings.append(vs["F1"]["heading_rad"] if "F1" in vs else None)
            turning.append(vs["F1"].get("turning") if "F1" in vs else None)
        if xpath is not None:
            return centres, headings, turning, dz
        return centres, coasting, dz
    finally:
        dzm.SMOOTHING = True


def test_smoothing_steadies_the_footprint():
    def jumps(cs):
        cs = [c for c in cs[15:] if c is not None]
        return np.median([np.linalg.norm(b - a) for a, b in zip(cs, cs[1:])])
    raw, _, _ = _drive(False)
    smooth, _, _ = _drive(True)
    assert jumps(smooth) < 0.6 * jumps(raw)


def test_a_short_detector_miss_keeps_the_zone_and_a_long_one_drops_it():
    centres, coasting, dz = _drive(True, frames=60, gap=set(range(30, 34)) | set(range(40, 60)))
    assert all(c is True for c in coasting[30:34])          # 4 missed frames (0.16 s): zone kept, marked coasting
    assert coasting[34] is False                              # measured again
    assert centres[59] is None                                 # 20 missed frames (0.8 s) > COAST_MAX_S: no zone
    assert dz.motion_stats["coasted"] >= 4 + 12


def test_a_straight_drive_is_never_called_a_turn():
    _, _, turning, _ = _drive(True, frames=60, xpath=lambda i: 700 + 6 * i)
    assert sum(1 for x in turning if x) == 0


def test_a_reversing_truck_points_the_other_way_without_swinging_sideways():
    path = lambda i: 700 + 6 * i if i < 40 else 940 - 6 * (i - 40)
    centres, headings, _, _ = _drive(True, frames=80, xpath=path)
    before, after = headings[35], headings[75]
    assert np.cos(after - before) < -0.9               # v9 turned the heading at 1 rad/s: still sideways 1.4 s later
    cs = [c for c in centres[20:] if c is not None]
    assert max(np.linalg.norm(b - a) for a, b in zip(cs, cs[1:])) < 0.6


def test_a_parked_truck_pulling_away_keeps_its_footprint():
    path = lambda i: 900 if i < 30 else 900 - 6 * (i - 30)  # parked 1.2 s, then drives off to the left
    centres, _, _, _ = _drive(True, frames=70, xpath=path)
    cs = [c for c in centres[5:] if c is not None]
    assert max(np.linalg.norm(b - a) for a, b in zip(cs, cs[1:])) < 0.6


def test_a_two_frame_reversed_reading_does_not_flip_the_zone():
    # forward at 6 px/frame; frames 40-41 read 30 px back (a 1 s fit can briefly read the truck reversed), then forward
    path = lambda i: 700 + 6 * i - (30 if i in (40, 41) else 0)
    _, headings, _, _ = _drive(True, frames=60, xpath=path)
    hs = [h for h in headings[30:] if h is not None]
    assert min(np.cos(b - a) for a, b in zip(hs, hs[1:])) > 0.5


def test_a_truck_that_stops_and_drives_off_the_other_way_points_the_new_way_at_once():
    def path(i):
        if i < 30:
            return 700 + 6 * i
        if i < 60:
            return 880           # stopped 1.2 s
        return 880 - 6 * (i - 60)
    _, headings, _, _ = _drive(True, frames=95, xpath=path)
    before = headings[25]
    first_new = next(i for i in range(60, 95) if headings[i] is not None and np.cos(headings[i] - before) < -0.9)
    assert first_new - 60 <= 30      # the 1 s fit needs to see the new motion; no extra wait once it does


def test_a_person_lost_and_seen_again_far_away_is_not_carried_on_as_a_ghost():
    dzm.SMOOTHING = True
    dz = dzm.DangerZone(CAL, (1920, 1080), 25.0, draws=8)
    def person(x):
        return [{"id": "P1", "box": [x, 500, x + 60, 680], "foot": [x + 30, 680], "operator": False}]
    for i in range(10):
        out = dz.step(i / 25, [], person(400))
    for i in range(10, 30):
        out = dz.step(i / 25, [], [])            # lost 0.8 s
    first = dz.step(30 / 25, [], person(900))["persons"]["P1"]["xy_m"]
    for i in range(31, 45):
        out = dz.step(i / 25, [], [])            # lost again
    again = dz.step(45 / 25, [], person(900))["persons"]["P1"]["xy_m"]
    assert np.linalg.norm(np.subtract(again, first)) < 0.3   # standing still at the new place, not walking on


# ------------------------------------------------------------------ v10 filters
def test_a_reanchor_keeps_the_speed_and_reports_the_shift():
    kf = mf.FloorKF(accel_sigma=3.0, meas_sigma=0.35, reset_after=6, keep_velocity=True)
    for i in range(40):
        kf.update(i / 25, [[2.5 * i / 25, 0.0]])
    states = [kf.update((40 + j) / 25, [[2.5 * (40 + j) / 25 + 2.0, 0.0]]) for j in range(6)]
    assert states == ["outlier"] * 5 + ["reset"]
    assert abs(kf.vel[0, 0] - 2.5) < 0.3 and abs(kf.jump[0, 0] - 2.0) < 0.3


def test_a_reanchor_is_not_thrown_by_one_spiky_reading():
    kf = mf.FloorKF(accel_sigma=3.0, meas_sigma=0.35, reset_after=6, keep_velocity=True)
    for i in range(40):
        kf.update(i / 25, [[2.5 * i / 25, 0.0]])
    spikes = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # the last reading of the run is itself 1 m off
    for j, sp in enumerate(spikes):
        kf.update((40 + j) / 25, [[2.5 * (40 + j) / 25 + 2.0 + sp, 0.0]])
    assert abs(kf.pos[0, 0] - (2.5 * 45 / 25 + 2.0)) < 0.25   # mean of the run: 1/6 of the spike, not all of it


def test_an_outline_switch_of_four_frames_does_not_move_the_track():
    kf = mf.FloorKF(accel_sigma=3.0, meas_sigma=0.35, reset_after=6, keep_velocity=True)
    for i in range(40):
        kf.update(i / 25, [[2.5 * i / 25, 0.0]])
    states = [kf.update((40 + j) / 25, [[2.5 * (40 + j) / 25 + 1.5, 0.0]]) for j in range(4)]
    assert states == ["outlier"] * 4
    assert kf.update(44 / 25, [[2.5 * 44 / 25, 0.0]]) == "update"
    assert abs(kf.pos[0, 0] - 2.5 * 44 / 25) < 0.3


def _turns(path, n, seed):
    rng = np.random.default_rng(seed)
    kf, h, on, w = mf.TurnKF(3.0, 1.0, 0.35), mf.Hysteresis(0.2, 0.1, 3), [], []
    for i in range(n):
        kf.update(i / 25, path(i / 25) + rng.normal(0, 0.35, 2))
        m = abs(kf.omega) if abs(kf.speed) >= 0.6 and 2 * kf.omega_se <= abs(kf.omega) and kf.updates >= 3 else 0.0
        on.append(h.update(m))
        w.append(kf.omega)
    return on, w


def test_turn_filter_does_not_call_a_noisy_straight_run_a_turn():
    frames_on = sum(sum(_turns(lambda t: np.array([2.5 * t, 1.0]), 150, s)[0]) for s in range(15))
    assert frames_on <= 0.01 * 15 * 150   # the 1 s line fit (v9) said "turning" in ~73 % of these frames


def test_turn_filter_confirms_a_real_turn_and_measures_it():
    W, v = 0.8, 2.5
    path = lambda t: np.array([v / W * np.sin(W * t), v / W * (1 - np.cos(W * t))])
    on, w = _turns(path, 100, 2)
    first = next(i for i, x in enumerate(on) if x)
    assert first / 25 < 2.0 and abs(np.mean(w[-25:]) - W) < 0.25


def test_braking_and_reversing_is_not_a_turn():
    def rev(t):  # 2 m/s, brakes at 2 m/s2, reverses to -2 m/s
        if t < 2:
            return np.array([2 * t, 0.0])
        if t < 4:
            s_ = t - 2
            return np.array([4 + 2 * s_ - s_ * s_, 0.0])
        return np.array([4 - 2 * (t - 4), 0.0])
    frames_on = sum(sum(_turns(rev, 175, s)[0]) for s in range(10))
    assert frames_on <= 0.02 * 10 * 175


# ------------------------------------------------------------------ people: association
def test_a_person_whose_box_jumps_keeps_one_id_with_prediction():
    def ids(tracker):
        out = []
        for i in range(12):
            x = 100 + 45 * i  # 45 px per frame on a 60 px wide box: IoU with the last box is below 0.2
            live = tracker.update(np.array([[x, 300, x + 60, 480]], float))
            out.append(live[0]["id"])
        return set(out)
    assert len(ids(rcp.IouTracker("P", max_missed=25))) > 1
    assert ids(rcp.IouTracker("P", max_missed=25, predict=True, dist_max=0.6)) == {"P1"}


def test_two_people_side_by_side_are_not_merged():
    tr = rcp.IouTracker("P", max_missed=25, predict=True, dist_max=0.6)
    for i in range(6):
        live = tr.update(np.array([[100, 300, 160, 480], [400, 300, 460, 480]], float))
    assert {t["id"] for t in live} == {"P1", "P2"}


# ------------------------------------------------------------------ parked objects
def test_a_never_moving_object_needs_a_typical_confidence_of_half_to_be_shown():
    fg = np.zeros((540, 960), np.uint8)
    def shown(conf):
        tracker = rcp.IouTracker("F", iou_min=0.15, max_missed=rcp.MOTION_MAX_FRAMES)
        last = []
        for i in range(30):
            c = conf if i % 6 else 0.46  # an occasional higher score, like the press machine in screen-002
            last = rcp.track_vehicles(tracker, [(np.array([500, 2, 680, 260], float), c, None, "forklift")], fg, 0.5, 0.5, i)
        return bool(last)
    assert not shown(0.33)
    assert shown(0.68)
