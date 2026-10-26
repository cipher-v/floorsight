"""Failures observed in the preserved demo and boundary audit; no external services."""
import numpy as np


def test_second_view_presence_cannot_override_missing_pair_geometry():
    from floorsight.agent.planner import has_support
    view = {"event":{"type":"near_miss","evidence":[
        {"kind":"trajectory","supports":False},
        {"kind":"second_view","supports":True,"camera_id":"a"},
        {"kind":"second_view","supports":True,"camera_id":"b"}]},
        "facts":{"detect_cameras":[],"geometry_supported":False}}
    assert not has_support(view)
    view['facts']['geometry_supported'] = True
    assert has_support(view)


def test_temporal_clearance_requires_recent_observed_displacement():
    from floorsight.agent.rules import RulesEngine
    from floorsight.contracts import Track, WorldState
    engine = RulesEngine(load_site('data/sim/demo_001/site.json'))
    def sample(t, x, state=TrackState.seen, person=False):
        tracks = [Track(track_id='fork', cls=TrackClass.forklift, x=x, y=5., vx=1.5, vy=0.,
                        sigma_m=.1, camera_ids=['a'], first_seen_t=0., last_seen_t=t, state=state)]
        if person:
            tracks.append(Track(track_id='person', cls=TrackClass.person, x=x, y=6.05, vx=0., vy=-1.4,
                                sigma_m=.1, camera_ids=['b'], first_seen_t=0., last_seen_t=t, state=TrackState.seen))
        return WorldState(t=t, tracks=tracks, cameras=[], coverage_version=1, pipeline_latency_ms=1.)
    engine.evaluate(sample(0., 11., TrackState.estimated))
    assert not engine.evaluate(sample(.6,12.,person=True))[0].data['geometry_supported']
    supported = engine.evaluate(sample(1.2,13.,person=True))[0]
    assert supported.data['geometry_supported']
    assert supported.data['heading_baseline_m'] == 1.
    assert not engine.evaluate(sample(2.3,14.,person=True))[0].data['geometry_supported']

from floorsight.contracts import TrackClass, TrackState
from floorsight.server import components
from floorsight.server.runtime import Runtime
from floorsight.site import load_site
from floorsight.vision.tracker import Tracker, _Track, FloorDet, cluster


def confirmed(tracker, xy=(2., 2.), sigma=.1):
    tr = _Track(1, "mover", np.array(xy), sigma, 0.)
    tr.confirmed = True
    tr.votes[TrackClass.person] = 20
    tracker.tracks = [tr]
    return tr


def test_restricted_position_expires_without_claiming_observation():
    tracker = Tracker(lambda x, y: 0, (16., 10.), restricted=lambda xy: True)
    confirmed(tracker)
    assert tracker.step(10., [], [])[0].state == TrackState.unseen
    assert tracker.step(20.1, [], []) == []


def test_stationary_disappearance_does_not_teleport_into_blind_space():
    tracker = Tracker(lambda x, y: 1, (16., 10.), nearest_blind=lambda xy, r: np.array([2.5, 2.]))
    confirmed(tracker)
    result = tracker.step(1., [], [])[0]
    assert result.state == TrackState.lost
    assert result.x == 2.


def test_uncertain_measurement_and_coasting_are_not_seen():
    tracker = Tracker(lambda x, y: 1, (16., 10.))
    tr = confirmed(tracker, sigma=.6)
    assert tracker._to_contract(tr, 0.).state == TrackState.estimated
    tr.measurement_sigma = .1
    assert tracker._to_contract(tr, .3).state == TrackState.estimated


def test_conflicting_cameras_do_not_manufacture_precision():
    ds = [FloorDet(cid, np.array([x, 2.]), .1, 1.2, "moving", 0., 1., (0, 0, 20, 20), (10., 20.))
          for cid, x in [("a", 1.), ("b", 2.)]]
    merged = cluster(ds, 1.5)[0]
    assert merged.sigma > .5


def test_established_person_cannot_jump_to_single_camera_vehicle_cluster():
    tracker = Tracker(lambda x, y: 1, (16., 10.))
    tr = confirmed(tracker)
    broad = FloorDet("cam1", np.array([2.1,2.]), .1, 1.5, "moving", 0., 1., (0,0,20,20), (10.,20.))
    assert tracker._assign([tr], cluster([broad], .9), lambda _: 1.5) == []


def test_narrow_fragment_of_known_vehicle_is_estimated_until_full_observation():
    tracker = Tracker(lambda x, y: 1, (16., 10.))
    tr = confirmed(tracker)
    tr.votes.clear()
    tr.votes[TrackClass.forklift] = 100
    def detection(width):
        return FloorDet("cam1", np.array([2., 2.]), .1, width, "moving", 0., 1., (0,0,20,20), (10.,20.))
    partial = tracker.step(.1, cluster([detection(.49)], .9), [])[0]
    assert partial.cls == TrackClass.forklift
    assert partial.state == TrackState.estimated and partial.sigma_m >= .6
    complete = tracker.step(.2, cluster([detection(1.4)], .9), [])[0]
    assert complete.state == TrackState.seen and complete.sigma_m < .35


def test_person_beside_vehicle_does_not_inherit_vehicle_fusion_radius():
    # Observed failure: two broad vehicle views agree, a narrow adjacent-person
    # view is swallowed by the 1.6m vehicle gate and poisons position/confidence.
    def det(cid, xy, width, sigma):
        return FloorDet(cid, np.array(xy), sigma, width, "moving", 0., 1., (0,0,20,20), (10.,20.))
    ds = [det("person_view", (11.78,6.06), .68, .106),
          det("vehicle_a", (11.41,4.91), 1.98, .115),
          det("vehicle_b", (11.14,4.95), 1.32, .170)]
    groups = cluster(ds, .9, 1.6)
    assert len(groups) == 2
    vehicle = next(g for g in groups if "vehicle_a" in g.camera_ids)
    assert vehicle.camera_ids == ["vehicle_a", "vehicle_b"]
    assert vehicle.sigma < .35


def test_real_vision_failure_never_falls_back_to_truth(monkeypatch):
    monkeypatch.setattr(components, "_try_import", lambda *a: (None, "deliberate failure"))
    site = load_site("data/sim/demo_001/site.json")
    vision, label, warnings = components.build_vision("real", site, None, fake_feed=object())
    assert label == "none" and isinstance(vision, components.NullVision) and warnings


def test_blur_exception_does_not_expose_raw_pixels():
    rt = object.__new__(Runtime)
    rt.blur_fn = lambda image: (_ for _ in ()).throw(RuntimeError("detector unavailable"))
    rt._error = lambda *a: None
    image, count, applied = rt._blur_one(np.full((10, 10, 3), 255, np.uint8))
    assert not image.any() and count == 0 and not applied


def test_owner_hold_never_enters_retention_paths():
    from types import SimpleNamespace
    rt = object.__new__(Runtime)
    rt.cfg = SimpleNamespace(preserve_evidence=True)
    assert rt._purge()["preserved"] is True


def test_closed_case_expires_operational_approval_and_ignores_late_click():
    from floorsight.agent.runtime import FloorsightAgent, Case
    from floorsight.agent.planner import Action
    from floorsight.contracts import Event, EventStatus, EventType, ApprovalRequest, ApprovalStatus
    site = load_site("data/sim/demo_001/site.json")
    agent = FloorsightAgent(site, components.NullVision(site))
    event = Event(event_id="E1", type=EventType.unseen_in_zone, rule_id=None,
                  t_start=0., t_detect=10., location=(1., 8.), severity="warning", status=EventStatus.verifying, summary="Unseen person")
    case = Case(event, ("unseen", "cold"), None, [], {}, 10.)
    ap = ApprovalRequest(approval_id="A1", event_id="E1", action="dispatch_check",
                         summary="Check area", status=ApprovalStatus.pending, created_t=10.)
    agent.approvals[ap.approval_id] = ap
    agent.cases[event.event_id] = case
    agent.t = 25.
    updates = []
    agent._do_resolve_event(case, Action("resolve_event", {}, "track expired"), updates)
    assert ap.status == ApprovalStatus.expired and len(updates) == 1
    assert agent.decide("A1", True, 26.) == []


def test_missing_camera_loses_coverage_and_stale_frames_cannot_revive_it(monkeypatch):
    from floorsight.vision.pipeline import RealVision
    from floorsight.contracts import FrameMeta, CameraStatus
    from floorsight.interfaces import FramePacket
    site = load_site("data/sim/demo_001/site.json")
    vision = RealVision(site, workers=1)
    # Exercise lifecycle independently of image detectors and calibration compute.
    vision._plates_checked = True
    def tick(cam, p):
        cam.last_t = p.meta.t
        cam.frames += 1
    monkeypatch.setattr(vision, "_camera_step", tick)
    def packet(cid, t):
        return FramePacket(FrameMeta(camera_id=cid, frame_index=int(t*10), t=t, width=960, height=540, faces_blurred=0), np.zeros((2,2,3), np.uint8))
    try:
        vision.process({"cam1": packet("cam1", 0.), "cam2": packet("cam2", 0.)})
        world = vision.process({"cam1": packet("cam1", 1.), "cam2": packet("cam2", 0.)})
        assert next(c for c in world.cameras if c.camera_id == "cam2").status == CameraStatus.offline
        assert vision.cams['cam2'].frames == 1
        assert vision.sharper_crop('cam2', (2.,2.)).supports is None
    finally:
        vision.close()
