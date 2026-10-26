"""Danger zones with more than one vehicle in view (bugs found 23 Oct on the AWS run with the new detectors)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("danger_zone", ROOT / "scripts" / "danger_zone.py")
dzm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dzm)
CAL = ROOT / "tests" / "data" / "floor-calibration-camA.json"


def rect(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


NEAR = {"id": "F1", "polygons": [rect(900, 420, 1300, 760)], "loadPolygons": [], "moving": True, "estimated": False}
PARKED = {"id": "F2", "polygons": [rect(200, 600, 480, 760)], "loadPolygons": [], "moving": False, "estimated": False}
PERSON = [{"id": "P1", "box": [1340.0, 560.0, 1400.0, 770.0], "foot": [1370.0, 770.0], "operator": False}]


def clearance(vehicles):
    dz = dzm.DangerZone(CAL, (1920, 1080), 25.0, draws=8)
    out = dz.step(1.0, vehicles, PERSON)
    return out["persons"]["P1"]


def test_a_parked_second_vehicle_does_not_change_the_gap_to_the_moving_one():
    alone = clearance([NEAR])
    both = clearance([NEAR, PARKED])
    assert both["vehicle"] == "F1"
    assert abs(both["clearance_m"]["median"] - alone["clearance_m"]["median"]) < 1e-6
    assert len(both["clearance"]) == len(alone["clearance"]) == 8  # one sample per draw, not per draw x vehicle
    assert "F2" in both["otherVehicles"]


def test_a_vehicle_cut_by_the_side_edge_gets_no_zone():
    dz = dzm.DangerZone(CAL, (1920, 1080), 25.0, draws=8)
    cut = {"id": "F3", "polygons": [rect(1700, 400, 1919, 700)], "loadPolygons": [], "moving": True, "estimated": False}
    out = dz.step(1.0, [cut], PERSON)
    assert out["vehicles"] == [] and getattr(dz, "skipped_side_cut", 0) == 1
