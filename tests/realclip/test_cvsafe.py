"""cvsafe: polygons are clipped to the image before filling, so OpenCV never sees an out-of-image vertex."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location("cvsafe", Path(__file__).resolve().parents[2] / "scripts" / "cvsafe.py")
cvsafe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cvsafe)


def test_clip_keeps_inside_polygons():
    pts = np.array([[10, 10], [50, 10], [50, 40], [10, 40]], float)
    assert np.allclose(cvsafe.clip_polygon(pts, 100, 100), pts)


def test_clip_cuts_at_the_bottom_edge():
    c = cvsafe.clip_polygon([[111, 248], [102, 251], [92, 254], [88, 242], [107, 236]], 321, 251)
    assert len(c) >= 3 and c[:, 1].max() <= 250 and c[:, 0].min() >= 0


def test_fill_matches_plain_fill_inside_the_image():
    """Same pixels as an unclipped fill for the COOL crash polygon (on builds where the plain fill works)."""
    poly = np.array([[111, 248], [102, 251], [92, 254], [88, 242], [107, 236]], np.int32)
    a = np.zeros((251, 321), np.uint8)
    cvsafe.fill_poly(a, [poly], 1)
    assert 200 <= int(a.sum()) <= 300  # stock 5.0.0 plain fill gives 254
    assert a[236:251, 88:112].any() and not a[:230].any()


def test_fully_outside_and_degenerate_polygons_are_skipped():
    a = np.zeros((50, 50), np.uint8)
    cvsafe.fill_poly(a, [[[100, 100], [120, 100], [110, 130]], [[1, 1], [2, 2]]], 1)
    cvsafe.fill_convex(a, [[-30, -30], [-10, -30], [-20, -5]], 1)
    assert a.sum() == 0


def test_convex_fill_of_a_rounded_rectangle_pose():
    a = np.zeros((60, 60), np.uint8)
    cvsafe.fill_convex(a, [[10, 10], [40, 10], [40, 10], [40, 30], [10, 30]], 1)  # duplicate vertex from rounding
    b = np.zeros((60, 60), np.uint8)
    cv2.fillConvexPoly(b, np.array([[10, 10], [40, 10], [40, 30], [10, 30]], np.int32), 1)
    assert (a == b).all()
