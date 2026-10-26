"""A near-miss whose tracks are too new gets a second look instead of an instant rejection."""

from floorsight.agent.planner import LocalPlanner


def _view(evidence):
    return {
        "event": {
            "type": "near_miss",
            "status": "verifying",
            "location": [11.8, 5.2],
            "track_ids": ["T30", "T25"],
            "camera_ids": ["cam3"],
            "evidence": evidence,
            "summary": "Near-miss",
        },
        "facts": {"detect_cameras": ["cam3"]},
        "trigger": "new",
        "done": [],
        "approvals": [],
        "alerts": [],
    }


WEAK_TRAJ = {"kind": "trajectory", "camera_id": None, "supports": False, "note": "track 0.4 s old"}
CROP = {"kind": "crop", "camera_id": "cam3", "supports": None, "note": "close-up"}


def _sv(cam, supports):
    return {"kind": "second_view", "camera_id": cam, "supports": supports, "note": ""}


def test_weak_tracks_trigger_a_second_view():
    a = LocalPlanner().next_action(_view([WEAK_TRAJ]))
    assert a.tool == "second_view"
    assert a.input["kind"] == "moving"


def test_two_cameras_confirm_only_with_supported_pair_geometry():
    view = _view([WEAK_TRAJ, _sv("cam2", True), _sv("cam3", True), CROP])
    assert LocalPlanner().next_action(view).tool == "reject_event"
    view['facts']['geometry_supported'] = True
    a = LocalPlanner().next_action(view)
    assert a.tool == "confirm_event"


def test_one_camera_is_not_enough():
    a = LocalPlanner().next_action(_view([WEAK_TRAJ, _sv("cam2", False), _sv("cam3", True), CROP]))
    assert a.tool == "reject_event"
