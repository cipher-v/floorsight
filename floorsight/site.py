"""Site plan loading and floor geometry shared by every component."""

from __future__ import annotations

import json
import math

from floorsight.contracts import XY, SitePlan, Zone


def load_site(path: str) -> SitePlan:
    with open(path, encoding="utf-8") as f:
        return SitePlan.model_validate(json.load(f))


def point_in_polygon(x: float, y: float, poly: list[XY]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def polygon_centroid(poly: list[XY]) -> XY:
    a = cx = cy = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(a) < 1e-12:
        return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)
    return (cx / (3 * a), cy / (3 * a))


def zones_at(site: SitePlan, x: float, y: float) -> list[Zone]:
    return [z for z in site.zones if point_in_polygon(x, y, z.polygon)]


def zone_by_id(site: SitePlan, zone_id: str) -> Zone | None:
    return next((z for z in site.zones if z.zone_id == zone_id), None)


def box_distance(px: float, py: float, cx: float, cy: float, heading: float, length: float, width: float) -> float:
    """Distance from a point to an oriented rectangle (0 when inside)."""
    c, s = math.cos(-heading), math.sin(-heading)
    lx = (px - cx) * c - (py - cy) * s
    ly = (px - cx) * s + (py - cy) * c
    return math.hypot(max(abs(lx) - length / 2, 0.0), max(abs(ly) - width / 2, 0.0))


# Footprint assumed for a forklift when measuring near-miss distance (metres).
FORKLIFT_LENGTH_M = 2.0
FORKLIFT_WIDTH_M = 1.1


def sight_line_blocked(site: SitePlan, cam_xyz: tuple[float, float, float], target_xyz: tuple[float, float, float], step_m: float = 0.1) -> bool:
    """True if a straight line from camera to target passes through any obstacle volume."""
    cx, cy, cz = cam_xyz
    tx, ty, tz = target_xyz
    length = math.dist((cx, cy, cz), (tx, ty, tz))
    n = max(2, int(length / step_m))
    for k in range(1, n):
        a = k / n
        x, y, z = cx + a * (tx - cx), cy + a * (ty - cy), cz + a * (tz - cz)
        for ob in site.obstacles:
            if z < ob.height_m and point_in_polygon(x, y, ob.polygon):
                return True
    return False
