"""Rules engine on hand-built WorldStates."""

from floorsight.agent.rules import RulesEngine, default_rules
from floorsight.contracts import CameraHealth, CameraStatus, Track, TrackClass, TrackState, WorldState


def trk(tid, cls, x, y, vx=0.0, vy=0.0, cams=("cam2",), state=TrackState.seen, first=0.0, sigma=0.1):
    return Track(track_id=tid, cls=cls, x=x, y=y, vx=vx, vy=vy, sigma_m=sigma, camera_ids=list(cams), first_seen_t=first, last_seen_t=0.0, state=state)


def world(t, tracks, cams=()):
    return WorldState(t=t, tracks=tracks, cameras=list(cams), coverage_version=1, pipeline_latency_ms=1.0)


def test_default_rules(site):
    ids = {r.rule_id for r in default_rules(site)}
    assert ids == {"R-exit_b-blocked", "R-near-miss", "R-cold_room-unseen"}


def test_blocked_after_20s_once_per_object(site):
    eng = RulesEngine(site)
    news = []
    for i in range(0, 260):
        t = i * 0.1
        new, _ = eng.update(world(t, [trk("T-box", TrackClass.object, 15.0, 7.9)]))
        news += [(t, c) for c in new]
    assert len(news) == 1
    t, c = news[0]
    assert abs(t - 20.0) < 0.11 and c.type.value == "blocked_exit" and c.zone_id == "exit_b"


def test_moving_or_outside_objects_do_not_block(site):
    eng = RulesEngine(site)
    for i in range(0, 260):
        new, _ = eng.update(world(i * 0.1, [trk("T-a", TrackClass.object, 15.0, 7.9, vx=0.5), trk("T-b", TrackClass.object, 10.0, 7.9), trk("T-p", TrackClass.person, 15.0, 7.9)]))
        assert not new


def test_spot_memory_raises_threshold_for_the_same_camera_only(site):
    eng = RulesEngine(site)
    eng.remember_false_alarm("exit_b", (14.95, 7.6), ["cam3"], 48.0, "EV-x", "glare")
    first = {}
    for i in range(0, 400):
        t = i * 0.1
        new, _ = eng.update(world(t, [trk("T-glare2", TrackClass.object, 14.9, 7.2, cams=["cam3"]), trk("T-real", TrackClass.object, 15.4, 8.6, cams=["cam2", "cam3"])]))
        for c in new:
            first.setdefault(c.track_ids[0], t)
    assert abs(first["T-real"] - 20.0) < 0.11
    assert abs(first["T-glare2"] - 30.0) < 0.11


def _forklift(t, x, vx=1.5):
    return trk("T-fk", TrackClass.forklift, x, 5.0, vx=vx, cams=["cam1"])


def test_near_miss_needs_distance_and_closing(site):
    eng = RulesEngine(site)
    # person walking down toward the lane while the forklift passes: closing, 0.5 m from the body
    p = trk("T-p", TrackClass.person, 12.0, 6.05, vy=-1.4, cams=["cam1"])
    new, _ = eng.update(world(1.0, [_forklift(1.0, 12.0), p]))
    assert [c.type.value for c in new] == ["near_miss"]
    # same gap but moving away: no hit
    eng2 = RulesEngine(site)
    away = trk("T-p", TrackClass.person, 12.0, 6.05, vy=1.4, cams=["cam1"])
    new, _ = eng2.update(world(1.0, [_forklift(1.0, 12.0), away]))
    assert not new


def test_near_miss_merges_within_3s_per_pair(site):
    eng = RulesEngine(site)
    p = trk("T-p", TrackClass.person, 12.0, 6.05, vy=-1.4, cams=["cam1"])
    other = trk("T-q", TrackClass.person, 12.4, 6.05, vy=-1.4, cams=["cam1"])
    n1, _ = eng.update(world(1.0, [_forklift(1.0, 12.0), p, other]))
    n2, c2 = eng.update(world(3.5, [_forklift(3.5, 12.0), p]))
    n3, _ = eng.update(world(7.0, [_forklift(7.0, 12.0), p]))
    assert len(n1) == 2  # two different pairs
    assert not n2 and len(c2) == 1  # merged into the same incident
    assert len(n3) == 1  # > 3 s gap: a new incident


def test_forklift_heading_kept_when_slow(site):
    eng = RulesEngine(site)
    eng.update(world(0.0, [trk("T-fk", TrackClass.forklift, 13.0, 3.0, vy=-1.5)]))
    h = eng._heading["T-fk"]
    eng.update(world(0.1, [trk("T-fk", TrackClass.forklift, 13.0, 2.9, vy=-0.05)]))
    assert eng._heading["T-fk"] == h


def test_unseen_in_cold_room_after_10s(site):
    eng = RulesEngine(site)
    hits = []
    for i in range(0, 150):
        t = i * 0.1
        new, _ = eng.update(world(t, [trk("T-p2", TrackClass.person, 1.6, 8.8, cams=[], state=TrackState.unseen, sigma=0.6)]))
        hits += [(t, c) for c in new]
    assert len(hits) == 1 and abs(hits[0][0] - 10.0) < 0.11 and hits[0][1].type.value == "unseen_in_zone"


def test_camera_moved_once_per_episode(site):
    eng = RulesEngine(site)
    moved = CameraHealth(camera_id="cam4", status=CameraStatus.moved, fps=10, last_frame_t=0, markers_visible=6, calib_error_m=0.35)
    ok = moved.model_copy(update={"status": CameraStatus.ok})
    a, _ = eng.update(world(100.0, [], [moved]))
    b, _ = eng.update(world(100.1, [], [moved]))
    eng.update(world(100.2, [], [ok]))
    c, _ = eng.update(world(140.0, [], [moved]))
    assert len(a) == 1 and not b and len(c) == 1


def test_agent_slow_zone_rule_fires_in_the_crossing_only(site):
    from floorsight.contracts import Rule, RuleType

    eng = RulesEngine(site)
    eng.add_rule(Rule(rule_id="R-A001", text="slow zone", type=RuleType.proximity, zone_id="aisle4_crossing", classes=[TrackClass.person, TrackClass.forklift], distance_m=5.0, source="agent"))
    fk = trk("T-fk", TrackClass.forklift, 9.0, 5.0, vx=1.5, cams=["cam1"])
    inside = trk("T-p", TrackClass.person, 11.8, 5.0, cams=["cam1"])  # standing still in the crossing, 1.8 m from the body
    outside = trk("T-q", TrackClass.person, 8.0, 7.5, cams=["cam1"])
    new, _ = eng.update(world(1.0, [fk, inside, outside]))
    assert [(c.rule.rule_id, c.track_ids[0], c.zone_id) for c in new] == [("R-A001", "T-p", "aisle4_crossing")]


def test_parked_forklift_blocks_the_exit(site):
    eng = RulesEngine(site)
    news = []
    for i in range(0, 260):
        new, _ = eng.update(world(i * 0.1, [trk("T-fl", TrackClass.forklift, 15.0, 7.9)]))
        news += new
    assert len(news) == 1 and news[0].type.value == "blocked_exit" and "parked forklift" in news[0].summary
