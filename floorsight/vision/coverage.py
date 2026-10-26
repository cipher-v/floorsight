"""Which floor cells each camera can see, from its fitted pose and the site's obstacles."""

from __future__ import annotations

import numpy as np

from floorsight.contracts import CameraInfo, SitePlan
from floorsight.site import sight_line_blocked
from floorsight.vision.calib import Calibration


def _obstacle_boxes(site: SitePlan) -> list[tuple[float, float, float, float, float]]:
    out = []
    for ob in site.obstacles:
        xs = [p[0] for p in ob.polygon]
        ys = [p[1] for p in ob.polygon]
        out.append((min(xs), min(ys), max(xs), max(ys), ob.height_m))
    return out


def _segment_may_hit(boxes, a: np.ndarray, b: np.ndarray) -> bool:
    """Cheap pre-filter: does the segment's bounding box overlap any obstacle volume's bounding box?"""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return any(lo[0] <= x1 and hi[0] >= x0 and lo[1] <= y1 and hi[1] >= y0 and lo[2] < h for x0, y0, x1, y1, h in boxes)


def sees(site: SitePlan, calib: Calibration, cam: CameraInfo, xyz: tuple[float, float, float], margin_px: float = 10.0, boxes=None) -> bool:
    """Point inside the image (with margin), in front of the camera, and not hidden behind an obstacle."""
    uv, z = calib.project(np.asarray([xyz], float))
    u, v = uv[0]
    if z[0] <= 0.2 or not (margin_px <= u <= cam.width - margin_px and margin_px <= v <= cam.height - margin_px):
        return False
    c = calib.position
    boxes = _obstacle_boxes(site) if boxes is None else boxes
    if not _segment_may_hit(boxes, c, np.asarray(xyz, float)):
        return True
    return not sight_line_blocked(site, tuple(float(v) for v in c), tuple(xyz))


def visibility_grid(site: SitePlan, calib: Calibration, cam: CameraInfo, cell_m: float = 0.5, height_m: float = 1.0) -> np.ndarray:
    """(rows, cols) bool, row 0 at y = 0: cell centres at `height_m` this camera can see."""
    cols, rows = int(round(site.width_m / cell_m)), int(round(site.depth_m / cell_m))
    boxes = _obstacle_boxes(site)
    out = np.zeros((rows, cols), bool)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = sees(site, calib, cam, ((c + 0.5) * cell_m, (r + 0.5) * cell_m, height_m), boxes=boxes)
    return out


FOOT_M = 0.1
HEAD_M = 1.7


def placeable_grid(site: SitePlan, calib: Calibration, cam: CameraInfo, cell_m: float = 0.5) -> np.ndarray:
    """(rows, cols) bool: cells where this camera sees a standing person's feet AND head.

    Feet are what put a person on the floor map; a camera that only sees heads over a
    rack (or has the feet cut off at the image edge) cannot place anyone there, so
    the cell must count as blind for honest coverage and "unseen" tracking.
    """
    cols, rows = int(round(site.width_m / cell_m)), int(round(site.depth_m / cell_m))
    boxes = _obstacle_boxes(site)
    out = np.zeros((rows, cols), bool)
    for r in range(rows):
        for c in range(cols):
            x, y = (c + 0.5) * cell_m, (r + 0.5) * cell_m
            out[r, c] = sees(site, calib, cam, (x, y, FOOT_M), boxes=boxes) and sees(site, calib, cam, (x, y, HEAD_M), boxes=boxes)
    return out
