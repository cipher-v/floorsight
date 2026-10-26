"""Polygon filling that never hands OpenCV a vertex outside the image.

Why: on 23 Oct 2026 the COOL build (OpenCV 5.1.0-dev, AWS Graviton4 Marketplace AMI) crashed (SIGSEGV) in cv2.fillPoly
for a 5-point polygon whose lowest vertices lay 4 rows below a 251 x 321 uint8 image; stock OpenCV 5.0.0 fills it
(254 px). Reproduction: eval/aws-cool-fillpoly-crash-20261023.txt. Clipping every polygon to the image rectangle first
(Sutherland-Hodgman) gives the same visible fill on any build and avoids that code path.
"""

from __future__ import annotations

import cv2
import numpy as np


def clip_polygon(pts, w: int, h: int) -> np.ndarray:
    """Clip a polygon (N x 2, any order) to [0, w-1] x [0, h-1]; returns float points (possibly empty)."""
    poly = [tuple(map(float, p)) for p in np.asarray(pts, np.float64).reshape(-1, 2) if np.isfinite(p).all()]
    for axis, lim, keep_le in ((0, 0.0, False), (0, w - 1.0, True), (1, 0.0, False), (1, h - 1.0, True)):
        if not poly:
            break
        inside = (lambda p: p[axis] <= lim) if keep_le else (lambda p: p[axis] >= lim)
        out = []
        for i, cur in enumerate(poly):
            prev = poly[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(_cross(prev, cur, axis, lim))
                out.append(cur)
            elif inside(prev):
                out.append(_cross(prev, cur, axis, lim))
        poly = out
    return np.array(poly, np.float64).reshape(-1, 2)


def _cross(a, b, axis, lim):
    t = (lim - a[axis]) / (b[axis] - a[axis])
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def fill_poly(img, polys, color) -> None:
    """cv2.fillPoly over polygons clipped to the image; polygons that vanish are skipped."""
    h, w = img.shape[:2]
    parts = []
    for p in polys:
        c = clip_polygon(p, w, h)
        if len(c) >= 3:
            parts.append(np.round(c).astype(np.int32))
    if parts:
        cv2.fillPoly(img, parts, color)


def fill_convex(img, pts, color) -> None:
    """cv2.fillConvexPoly of the clipped convex hull (for shapes convex by construction)."""
    h, w = img.shape[:2]
    c = clip_polygon(pts, w, h)
    if len(c) >= 3:
        hull = cv2.convexHull(np.round(c).astype(np.int32).reshape(-1, 1, 2)).reshape(-1, 2)
        cv2.fillConvexPoly(img, hull, color)
