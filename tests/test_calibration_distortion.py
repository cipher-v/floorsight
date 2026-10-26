"""Lens distortion is applied around the image<->floor homography (audit R4); zero distortion is bit-identical.

Synthetic camera only: floor points are projected with cv2.projectPoints (the lens model solvePnP uses) and must map
back to the same floor spot. No detector, no rendering.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from floorsight.contracts import CameraInfo
from floorsight.vision.calib import CORNER_OFFSETS, fit_calibration, floor_error, undistort_px

K = np.array([[800.0, 0.0, 480.0], [0.0, 800.0, 270.0], [0.0, 0.0, 1.0]])
SIZE = (960, 540)
BARREL = np.array([-0.28, 0.09, 0.001, -0.0005, 0.0])  # wide CCTV lens: up to ~50 px shift near the corners
CAM_AT, LOOK_AT = np.array([0.0, -2.0, 4.0]), np.array([6.0, 6.0, 0.0])
CENTRES = [(2.0, 3.0), (5.0, 3.0), (8.0, 4.0), (3.0, 6.0), (6.0, 7.0), (9.0, 8.0), (4.0, 10.0), (8.0, 11.0)]


def _pose():
    fwd = (LOOK_AT - CAM_AT) / np.linalg.norm(LOOK_AT - CAM_AT)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    R = np.stack([right, np.cross(fwd, right), fwd])  # world -> camera (x right, y down, z forward)
    return cv2.Rodrigues(R)[0], (-R @ CAM_AT).reshape(3, 1)


RVEC, TVEC = _pose()


def _project(xyz, dist):
    return cv2.projectPoints(np.asarray(xyz, float).reshape(-1, 3), RVEC, TVEC, K, dist)[0].reshape(-1, 2)


def _fit(dist):
    floor = {i: np.asarray(c) + CORNER_OFFSETS * 0.3 for i, c in enumerate(CENTRES)}
    obs = {i: _project(np.column_stack([c, np.zeros(4)]), dist) for i, c in floor.items()}
    info = CameraInfo(camera_id="cam1", width=SIZE[0], height=SIZE[1], fps=10.0, K=K.tolist(), dist=list(dist))
    calib = fit_calibration(obs, floor, info)
    assert calib is not None
    return calib, obs, floor


def _floor_points_in_view(dist, height=0.0):
    """Floor grid points the camera really sees (inside the image with and without the lens), and their pixels."""
    g = np.array([(x, y) for x in np.arange(-6.0, 20.0, 0.5) for y in np.arange(0.0, 30.0, 0.5)])
    xyz = np.column_stack([g, np.full(len(g), height)])
    px, ideal = _project(xyz, dist), _project(xyz, np.zeros(5))
    in_front = (xyz @ cv2.Rodrigues(RVEC)[0].T + TVEC.ravel())[:, 2] > 0
    inside = lambda p: (p >= 0).all(1) & (p < SIZE).all(1)
    keep = in_front & inside(px) & inside(ideal)
    return xyz[keep], px[keep]


def test_distorted_pixels_map_back_to_their_floor_spot():
    calib, _, _ = _fit(BARREL)
    assert calib.error_m < 0.005 and calib.homography_error_m < 0.005
    xyz, px = _floor_points_in_view(BARREL)
    assert len(xyz) > 500
    err = np.linalg.norm(calib.to_floor(px) - xyz[:, :2], axis=1)
    assert err.max() < 0.003  # within a few mm everywhere in view, image corners included
    # The same pixels through the bare homography (the old behaviour) land metres off near the edges.
    bare = cv2.perspectiveTransform(px.reshape(-1, 1, 2), calib.H_img2floor).reshape(-1, 2)
    assert np.linalg.norm(bare - xyz[:, :2], axis=1).max() > 0.5


def test_floor_to_pixel_uses_the_same_lens_model():
    calib, obs, floor = _fit(BARREL)
    xyz, px = _floor_points_in_view(BARREL)
    uv, z = calib.project(xyz)
    assert (z > 0).all() and np.abs(uv - px).max() < 0.05
    assert calib.floor_error(obs, floor) < 0.005  # drift check sees an unmoved camera as unmoved
    # Raised plane (e.g. 1 m): plane_to_floor undistorts as well.
    xyz1, px1 = _floor_points_in_view(BARREL, height=1.0)
    assert np.linalg.norm(calib.plane_to_floor(px1, 1.0) - xyz1[:, :2], axis=1).max() < 0.003
    m = calib.metres_per_pixel((900.0, 500.0))
    assert np.isfinite(m) and m > 0


def test_zero_distortion_is_bit_identical_to_the_plain_homography():
    calib, obs, floor = _fit(np.zeros(5))
    assert not calib.distorted
    xyz, px = _floor_points_in_view(np.zeros(5))
    assert np.array_equal(undistort_px(px, K, np.zeros(5)), px)
    plain = cv2.perspectiveTransform(np.asarray(px, np.float64).reshape(-1, 1, 2), calib.H_img2floor).reshape(-1, 2)
    assert np.array_equal(calib.to_floor(px), plain)
    cam = xyz @ calib.R.T + calib.tvec.ravel()
    uv = cam @ calib.K.T
    uv, z = uv[:, :2] / np.where(np.abs(cam[:, 2]) < 1e-9, 1e-9, cam[:, 2])[:, None], cam[:, 2]
    got_uv, got_z = calib.project(xyz)
    assert np.array_equal(got_uv, uv) and np.array_equal(got_z, z)
    H = calib.K @ np.column_stack([calib.R[:, 0], calib.R[:, 1], calib.R[:, 2] * 1.0 + calib.tvec.ravel()])
    assert np.array_equal(calib.plane_to_floor(px, 1.0), cv2.perspectiveTransform(px.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2))
    assert calib.floor_error(obs, floor) == floor_error(calib.H_img2floor, obs, floor)


@pytest.mark.parametrize("dist", [None, np.zeros(5), np.zeros(8)])
def test_no_distortion_leaves_pixels_untouched(dist):
    px = np.array([[0.0, 0.0], [959.5, 539.5], [480.25, 270.75]])
    assert np.array_equal(undistort_px(px, K, dist), px)
    assert undistort_px(np.zeros((0, 2)), K, BARREL).shape == (0, 2)
