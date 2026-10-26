from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

from floorsight.contracts import SitePlan
from floorsight.edge import intrinsics, markers

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _detect(img):
    det = cv2.aruco.ArucoDetector(markers.get_dictionary())
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return corners, ([] if ids is None else ids.ravel().tolist())


@pytest.mark.parametrize("paper,size", [("A4", (2480, 3508)), ("A3", (3508, 4961))])
def test_sheet_page_size_and_markers_detectable(paper, size):
    pages = markers.marker_sheets([0, 1, 2, 3, 4, 5], 80.0, paper=paper, dpi=300)
    assert pages[0].shape[1] == size[0] and pages[0].shape[0] == size[1]
    found = []
    for p in pages:
        corners, ids = _detect(p)
        found += ids
        for c in corners:
            side_px = np.linalg.norm(c[0][0] - c[0][1])
            assert abs(side_px - markers.mm_to_px(80.0, 300)) <= 2  # exact physical size
    assert sorted(found) == [0, 1, 2, 3, 4, 5]


def test_big_marker_is_tiled():
    pages = markers.marker_sheets([7], 600.0, paper="A3", dpi=100)
    assert len(pages) > 1  # 600 mm cannot fit an A3 page
    for p in pages:
        assert p.shape[:2] == (markers.mm_to_px(420, 100), markers.mm_to_px(297, 100))


def test_png_has_dpi(tmp_path):
    paths = markers.write_sheets(markers.marker_sheets([0], 100.0), str(tmp_path))
    with open(paths[0], "rb") as f:
        assert b"pHYs" in f.read(200)
    assert cv2.imread(paths[0]) is not None


def _site(tmp_path):
    src = os.path.join(ROOT, "data", "sim", "bench_dev", "site.json")
    if os.path.exists(src):
        with open(src, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {
            "site_id": "t", "name": "t", "width_m": 16.0, "depth_m": 10.0, "zones": [], "obstacles": [],
            "markers": [{"marker_id": 1, "center": [1, 1], "size_m": 0.6}],
            "cameras": [{"camera_id": "cam1", "width": 960, "height": 540, "fps": 10, "K": [[800, 0, 480], [0, 800, 270], [0, 0, 1]]}],
        }
    path = tmp_path / "site.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_place_measured_markers(tmp_path):
    site_path = _site(tmp_path)
    csv_path = tmp_path / "measured.csv"
    csv_path.write_text("marker_id,x_m,y_m,size_m\n1,4.25,2.55,\n42,7.0,7.5,0.2\n")
    site = markers.update_site_file(site_path, str(csv_path))
    by_id = {m.marker_id: m for m in site.markers}
    assert by_id[1].center == (4.25, 2.55) and by_id[1].size_m == 0.6  # size kept
    assert by_id[42].center == (7.0, 7.5) and by_id[42].size_m == 0.2  # added
    SitePlan.model_validate(json.loads(open(site_path, encoding="utf-8").read()))


def test_place_rejects_outside_plan(tmp_path):
    site_path = _site(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"marker_id": 1, "x_m": 99, "y_m": 1}]))
    with pytest.raises(ValueError):
        markers.update_site_file(site_path, str(bad))


def test_parse_ids():
    assert markers.parse_ids("0-3,7") == [0, 1, 2, 3, 7]
    with pytest.raises(ValueError):
        markers.parse_ids("49-50")


# --------------------------------------------------------------------------- #
# ChArUco calibration on synthetic views of a known camera
# --------------------------------------------------------------------------- #


def _synthetic_views(board, square_mm, K, size=(960, 540), n=12, seed=0):
    sx, sy = board.getChessboardSize()
    px_per_sq = 105  # marker = 78 px = 6 modules x 13 px
    bimg = board.generateImage((sx * px_per_sq, sy * px_per_sq), marginSize=0, borderBits=1)
    s = (square_mm / 1000.0) / px_per_sq  # metres per board-image pixel
    S = np.diag([s, s, 1.0])
    rng = np.random.default_rng(seed)
    w_m, h_m = sx * square_mm / 1000.0, sy * square_mm / 1000.0
    views = []
    for _ in range(n):
        rvec = rng.uniform([-0.5, -0.5, -0.3], [0.5, 0.5, 0.3])
        R, _ = cv2.Rodrigues(rvec)
        z = rng.uniform(0.35, 0.6)
        # keep the board centre near the image centre, shifted a little
        c = np.array([w_m / 2, h_m / 2, 0.0])
        t = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.03, 0.03), z]) - R @ c
        H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
        img = cv2.warpPerspective(bimg, H, size, flags=cv2.INTER_LINEAR, borderValue=200)
        views.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return views


def test_charuco_calibration_recovers_known_camera(tmp_path):
    board = intrinsics.make_board()
    K = np.array([[820.0, 0, 478.0], [0, 820.0, 268.0], [0, 0, 1]])
    views = _synthetic_views(board, intrinsics.SQUARE_MM, K)
    res = intrinsics.calibrate(views, board)
    assert res["views_used"] >= 8
    assert res["rms_px"] < 0.5
    Kr = np.array(res["K"])
    assert abs(Kr[0, 0] - 820) / 820 < 0.03
    assert abs(Kr[1, 1] - 820) / 820 < 0.03
    assert abs(Kr[0, 2] - 478) < 15 and abs(Kr[1, 2] - 268) < 15

    site_path = _site(tmp_path)
    site = intrinsics.write_camera_intrinsics(site_path, "cam1", res)
    cam = next(c for c in site.cameras if c.camera_id == "cam1")
    assert np.allclose(cam.K, res["K"]) and cam.width == 960


def test_board_page_is_printable_size():
    page = intrinsics.board_image(intrinsics.make_board(), paper="A4", dpi=300)
    assert page.shape == (2480, 3508)
