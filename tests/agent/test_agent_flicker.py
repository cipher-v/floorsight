"""Real tracking flickers and re-numbers people; incidents must not multiply."""

from floorsight.agent.runtime import FloorsightAgent
from floorsight.contracts import CameraHealth, CameraStatus, Track, TrackClass, TrackState, WorldState


class NoCoverVision:
    def cameras_covering(self, xy):
        return []

    def second_view(self, xy, exclude, kind):
        return []

    def sharper_crop(self, camera_id, xy, radius_m=1.0):
        raise AssertionError("no camera covers the cold room")

    def recalibrate(self, camera_id):
        return {"ok": True, "calib_error_m": 0.0, "markers_used": 4}

    def coverage(self):
        return None


def _world(t, tracks, site):
    cams = [CameraHealth(camera_id=c.camera_id, status=CameraStatus.ok, fps=10, last_frame_t=t, markers_visible=6, calib_error_m=0.01) for c in site.cameras]
    return WorldState(t=t, tracks=tracks, cameras=cams, coverage_version=1, pipeline_latency_ms=1.0)


def _person(tid, t, state=TrackState.unseen):
    return Track(
        track_id=tid, cls=TrackClass.person, x=1.6, y=8.8, vx=0, vy=0, sigma_m=0.8,
        camera_ids=[], first_seen_t=0.0, last_seen_t=t, state=state,
    )


def test_cold_room_renumbered_and_flickering_person_asks_once(site):
    agent = FloorsightAgent(site, NoCoverVision())
    t = 0.0
    while t < 60.0:
        if 25.0 <= t < 26.0:
            tracks = []  # one second of flicker: the track vanishes
        elif t < 30.0:
            tracks = [_person("T6", t)]
        else:
            tracks = [_person("T21", t)]  # tracker re-numbers the same person
        agent.on_world(_world(round(t, 1), tracks, site))
        t += 0.1
    snap = agent.snapshot()
    unseen = [e for e in snap["events"] if e["type"] == "unseen_in_zone"]
    checks = [a for a in snap["approvals"] if a["action"] == "dispatch_check"]
    assert len(unseen) == 1, unseen
    assert len(checks) == 1, checks
