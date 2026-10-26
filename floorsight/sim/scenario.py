"""Demo warehouse and its scripted 3-minute shift.

Produces synchronised footage from four ceiling cameras plus exact ground
truth (every entity's floor position each frame, and every scripted incident),
so the vision and agent layers can be scored precisely.

    python -m floorsight.sim.scenario --out data/sim/demo_001
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import cv2
import numpy as np

from floorsight.contracts import SitePlan
from floorsight.sim.world import Renderer, SimCamera, make_forklift, make_pallet, make_person

FPS = 10.0
DURATION_S = 180.0


# --------------------------------------------------------------------------- #
# Site
# --------------------------------------------------------------------------- #


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def demo_cameras() -> list[SimCamera]:
    return [
        SimCamera("cam1", (0.3, 0.3, 4.3), (6.0, 4.5, 0.0)),
        SimCamera("cam2", (15.7, 0.3, 4.3), (12.5, 6.0, 0.0)),
        SimCamera("cam3", (9.5, 9.8, 4.3), (13.0, 5.5, 0.0)),
        # cam4 is knocked at t=100 s (scripted self-repair scenario)
        SimCamera("cam4", (0.2, 5.0, 4.5), (9.0, 5.0, 0.0), bump_t=100.0, bump_yaw_deg=3.5, bump_pitch_deg=1.2),
    ]


def demo_site(cameras: list[SimCamera]) -> dict:
    markers = [
        (0, (0.9, 6.6)), (1, (4.2, 2.6)), (2, (8.0, 2.6)), (3, (10.6, 2.0)),
        (4, (14.9, 2.6)), (5, (14.9, 5.2)), (6, (12.0, 8.9)), (7, (8.0, 8.9)),
        (8, (4.5, 8.9)), (9, (2.6, 5.0)), (10, (11.6, 3.0)), (11, (6.0, 0.5)),
    ]
    site = {
        "schema_version": "1.0",
        "site_id": "demo_warehouse",
        "name": "Demo warehouse (simulated)",
        "width_m": 16.0,
        "depth_m": 10.0,
        "zones": [
            {"zone_id": "exit_b", "name": "Exit B", "kind": "exit", "polygon": rect(14.4, 6.9, 15.9, 8.9)},
            {"zone_id": "lane_main", "name": "Forklift lane (main aisle)", "kind": "lane", "polygon": rect(0.9, 4.35, 13.7, 5.65)},
            {
                "zone_id": "lane_loop",
                "name": "Forklift lane (loop)",
                "kind": "lane",
                "polygon": [(0.9, 0.55), (13.7, 0.55), (13.7, 4.35), (12.3, 4.35), (12.3, 1.85), (2.1, 1.85), (2.1, 4.35), (0.9, 4.35)],
            },
            {"zone_id": "aisle4_crossing", "name": "Aisle 4 crossing", "kind": "area", "polygon": rect(11.2, 4.35, 12.3, 5.65)},
            {"zone_id": "picking_a", "name": "Picking face A", "kind": "area", "polygon": rect(3.0, 2.0, 11.0, 3.15)},
            {"zone_id": "cold_room", "name": "Cold room (restricted)", "kind": "restricted", "polygon": rect(0.0, 7.4, 3.4, 10.0)},
        ],
        "obstacles": [
            {"obstacle_id": "rack_a", "name": "Rack A", "polygon": rect(3.0, 3.2, 11.0, 4.0), "height_m": 3.0},
            {"obstacle_id": "rack_b", "name": "Rack B", "polygon": rect(3.0, 6.0, 11.0, 6.8), "height_m": 3.0},
        ],
        "markers": [
            {"marker_id": mid, "dictionary": "DICT_4X4_50", "center": c, "size_m": 0.6, "rotation_deg": 0.0}
            for mid, c in markers
        ],
        "cameras": [
            {
                "camera_id": c.camera_id,
                "width": c.width,
                "height": c.height,
                "fps": c.fps,
                "K": c.K.tolist(),
                "dist": [0.0] * 5,
                "nominal_position": list(c.position),
            }
            for c in cameras
        ],
    }
    SitePlan.model_validate(site)  # contract check
    return site


# --------------------------------------------------------------------------- #
# Motion
# --------------------------------------------------------------------------- #


class Path:
    """Piecewise-linear route with optional waits: [(x, y, wait_s), ...]."""

    def __init__(self, points, speed, t0=0.0, loop=False):
        self.points = points
        self.speed = speed
        self.t0 = t0
        self.loop = loop
        self.segments = []  # (t_start, t_end, p0, p1)
        t = t0
        for i in range(len(points) - (0 if loop else 1)):
            a = points[i]
            b = points[(i + 1) % len(points)]
            if a[2] > 0:
                self.segments.append((t, t + a[2], a[:2], a[:2]))
                t += a[2]
            d = math.dist(a[:2], b[:2])
            self.segments.append((t, t + d / speed, a[:2], b[:2]))
            t += d / speed
        if not loop and points[-1][2] > 0:
            self.segments.append((t, t + points[-1][2], points[-1][:2], points[-1][:2]))
            t += points[-1][2]
        self.period = t - t0

    def at(self, t):
        if t < self.t0:
            p = self.points[0]
            return p[0], p[1], 0.0, 0.0
        local = t - self.t0
        if self.loop:
            local %= self.period
        elif local >= self.period:
            p = self.segments[-1][3]
            return p[0], p[1], 0.0, 0.0
        tt = self.t0 + local
        for ts, te, p0, p1 in self.segments:
            if ts <= tt <= te:
                if te - ts < 1e-9:
                    return p0[0], p0[1], 0.0, 0.0
                a = (tt - ts) / (te - ts)
                x = p0[0] + a * (p1[0] - p0[0])
                y = p0[1] + a * (p1[1] - p0[1])
                vx = (p1[0] - p0[0]) / (te - ts)
                vy = (p1[1] - p0[1]) / (te - ts)
                return x, y, vx, vy
        p = self.segments[-1][3]
        return p[0], p[1], 0.0, 0.0


FORKLIFT = Path([(1.5, 5.0, 0), (13.0, 5.0, 0), (13.0, 1.2, 0), (1.5, 1.2, 0)], speed=1.5, loop=True)
# Forklift passes x=12 on the main aisle at 7.0 + 20.4k seconds.
NEAR_MISS_PASSES = [7.0 + 20.4 * k for k in (2, 5, 7)]  # ~47.8 s, ~109.0 s, ~149.8 s


def crossing_person():
    """Walks down through Aisle 4 crossing just after the forklift passes, three times."""
    pts = []
    t = 0.0
    start = (11.8, 7.9)
    bottom = (11.8, 2.8)
    shelf = (10.6, 2.8)
    pts.append((*start, 0.0))
    legs = []
    speed = 1.4
    for k, tp in enumerate(NEAR_MISS_PASSES):
        # be at y=5 0.45 s after the forklift centre passes x=12
        arrive_y5 = tp + 0.45
        leave_start = arrive_y5 - (start[1] - 5.0) / speed
        legs.append(leave_start)
    # Build an explicit timeline: wait at start, walk down, work at shelf, walk back up (when lane is clear)
    route = []
    cur_t = 0.0
    pos = start
    for k, leave in enumerate(legs):
        route.append(("wait", pos, leave - cur_t))
        route.append(("walk", start, bottom))
        route.append(("walk", bottom, shelf))
        t_at_shelf = leave + (math.dist(start, bottom) + math.dist(bottom, shelf)) / speed
        # Return when the forklift is on the far side of its loop (~10 s after a pass)
        ret_leave = t_at_shelf + 4.0
        route.append(("wait", shelf, 4.0))
        route.append(("walk", shelf, (11.8, 3.2)))
        route.append(("walk", (11.8, 3.2), start))
        cur_t = ret_leave + (math.dist(shelf, (11.8, 3.2)) + math.dist((11.8, 3.2), start)) / speed
        pos = start
    return route, speed


class RouteMotion:
    def __init__(self, route, speed):
        self.events = []
        t = 0.0
        for kind, a, b in route:
            if kind == "wait":
                self.events.append((t, t + max(0.0, b), a, a))
                t += max(0.0, b)
            else:
                d = math.dist(a, b)
                self.events.append((t, t + d / speed, a, b))
                t += d / speed

    def at(self, t):
        for ts, te, a, b in self.events:
            if ts <= t <= te:
                if te - ts < 1e-9:
                    return a[0], a[1], 0.0, 0.0
                k = (t - ts) / (te - ts)
                return (
                    a[0] + k * (b[0] - a[0]),
                    a[1] + k * (b[1] - a[1]),
                    (b[0] - a[0]) / (te - ts),
                    (b[1] - a[1]) / (te - ts),
                )
        last = self.events[-1][3]
        return last[0], last[1], 0.0, 0.0


PALLET_T = (75.0, 150.0)  # blocks Exit B
PALLET_XY = (15.05, 7.9)
GLARE_T = (28.0, 56.0)  # sun glare inside Exit B, seen only by cam3
GLARE_XY = (14.95, 7.6)


def people():
    p1 = Path(
        [(6.0, 8.6, 0), (13.5, 8.6, 0), (14.0, 7.9, 1.0), (13.5, 8.6, 0)],
        speed=1.1,
        loop=True,
    )
    # P1b: carries the pallet to Exit B at t=75 and fetches it at t=150
    p_carrier = RouteMotion(
        [
            ("wait", (13.0, 9.3), 68.0),
            ("walk", (13.0, 9.3), (14.3, 8.4)),
            ("wait", (14.3, 8.4), 5.0),
            ("walk", (14.3, 8.4), (13.0, 9.3)),
            ("wait", (13.0, 9.3), 70.0),
            ("walk", (13.0, 9.3), (14.3, 8.4)),
            ("wait", (14.3, 8.4), 4.0),
            ("walk", (14.3, 8.4), (13.0, 9.3)),
        ],
        speed=1.0,
    )
    # P2: goes into the restricted cold room (a blind spot) and stays
    p2 = RouteMotion(
        [
            ("wait", (6.5, 9.2), 15.0),
            ("walk", (6.5, 9.2), (1.6, 8.8)),
            ("wait", (1.6, 8.8), 40.0),
            ("walk", (1.6, 8.8), (6.5, 9.2)),
            ("wait", (6.5, 9.2), 200.0),
        ],
        speed=1.0,
    )
    # P3: works the picking face, drifting along it
    p3 = Path([(4.0, 2.75, 3.0), (9.0, 2.75, 3.0)], speed=0.5, loop=True)
    route, speed = crossing_person()
    p4 = RouteMotion(route, speed)
    return {
        "p1": (p1, (180, 80, 40)),
        "p2": (p2, (50, 50, 190)),
        "p3": (p3, (60, 150, 60)),
        "p4": (p4, (200, 60, 170)),
        "p5": (p_carrier, (40, 140, 220)),
    }


def heading_of(vx, vy, prev):
    if abs(vx) + abs(vy) < 1e-6:
        return prev
    return math.atan2(vy, vx)


# --------------------------------------------------------------------------- #
# Ground-truth incidents
# --------------------------------------------------------------------------- #


def forklift_box_distance(px, py, fx, fy, heading, length=2.0, width=1.1):
    c, s = math.cos(-heading), math.sin(-heading)
    lx = (px - fx) * c - (py - fy) * s
    ly = (px - fx) * s + (py - fy) * c
    dx = max(abs(lx) - length / 2, 0.0)
    dy = max(abs(ly) - width / 2, 0.0)
    return math.hypot(dx, dy)


def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def write_setup_plates(out_dir: str, cams=None, site=None) -> list[str]:
    """Empty-floor reference images, one per camera.

    On a real site the installer records a few seconds of the cleared floor while
    laying the markers; vision uses it as its starting picture of "nothing here".
    """
    cams = cams or demo_cameras()
    site = site or demo_site(cams)
    renderer = Renderer(site, cams, seed=99)
    paths = []
    for c in cams:
        path = os.path.join(out_dir, f"setup_empty_{c.camera_id}.png")
        cv2.imwrite(path, renderer.render(c, 0.0, []))
        paths.append(path)
    return paths


def run(out_dir: str, duration_s: float = DURATION_S, write_video: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    cams = demo_cameras()
    site = demo_site(cams)
    with open(os.path.join(out_dir, "site.json"), "w", encoding="utf-8") as f:
        json.dump(site, f, indent=2)
    if write_video:
        write_setup_plates(out_dir, cams, site)
    renderer = Renderer(site, cams)
    crowd = people()
    writers = {}
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for c in cams:
            writers[c.camera_id] = cv2.VideoWriter(
                os.path.join(out_dir, f"{c.camera_id}.mp4"), fourcc, FPS, (c.width, c.height)
            )
    truth_f = open(os.path.join(out_dir, "truth_frames.jsonl"), "w", encoding="utf-8")
    heading = 0.0
    n_frames = int(duration_s * FPS)
    near_miss_samples = []
    cold_room = next(z for z in site["zones"] if z["zone_id"] == "cold_room")["polygon"]
    cold_room_inside = []
    t_start = time.time()
    for i in range(n_frames):
        t = i / FPS
        entities = []
        bodies = []
        fx, fy, fvx, fvy = FORKLIFT.at(t)
        heading = heading_of(fvx, fvy, heading)
        bodies.append(make_forklift(fx, fy, heading))
        entities.append({"id": "forklift1", "cls": "forklift", "x": fx, "y": fy, "vx": fvx, "vy": fvy, "heading": heading})
        for pid, (motion, shirt) in crowd.items():
            x, y, vx, vy = motion.at(t)
            bodies.append(make_person(x, y, shirt))
            entities.append({"id": pid, "cls": "person", "x": x, "y": y, "vx": vx, "vy": vy})
            d = forklift_box_distance(x, y, fx, fy, heading)
            # closing speed: negative rate of change of distance
            d2 = forklift_box_distance(x + vx * 0.1, y + vy * 0.1, fx + fvx * 0.1, fy + fvy * 0.1, heading)
            closing = (d - d2) / 0.1
            near_miss_samples.append((t, pid, d, closing, x, y))
            if pid == "p2":
                cold_room_inside.append((t, point_in_poly(x, y, cold_room)))
        pallet_present = PALLET_T[0] <= t < PALLET_T[1]
        if pallet_present:
            bodies.append(make_pallet(*PALLET_XY))
            entities.append({"id": "pallet1", "cls": "object", "x": PALLET_XY[0], "y": PALLET_XY[1], "vx": 0.0, "vy": 0.0})
        glare_on = GLARE_T[0] <= t < GLARE_T[1]
        truth_f.write(json.dumps({"frame": i, "t": round(t, 3), "entities": entities}) + "\n")
        if write_video:
            for c in cams:
                g = None
                if glare_on and c.camera_id == "cam3":
                    pulse = 0.75 + 0.25 * math.sin(t * 1.3)
                    g = (GLARE_XY[0], GLARE_XY[1], 34, 0.95 * pulse)
                writers[c.camera_id].write(renderer.render(c, t, bodies, glare=g))
        if i % 300 == 0:
            print(f"  frame {i}/{n_frames}  ({time.time() - t_start:.0f}s)", flush=True)
    truth_f.close()
    for w in writers.values():
        w.release()

    # Scripted incidents, derived from the actual motion (same definitions the rules use)
    events = []
    by_person: dict = {}
    for t, pid, d, closing, x, y in near_miss_samples:
        if d < 0.8 and closing > 0.2:
            lst = by_person.setdefault(pid, [])
            if lst and t - lst[-1]["t_end"] <= 1.0:
                lst[-1]["t_end"] = t
                lst[-1]["min_distance_m"] = min(lst[-1]["min_distance_m"], d)
            else:
                lst.append({"t_start": t, "t_end": t, "min_distance_m": d, "location": [round(x, 2), round(y, 2)]})
    for pid, lst in by_person.items():
        for e in lst:
            events.append({"type": "near_miss", "track": pid, **e})
    events.append(
        {"type": "blocked_exit", "zone_id": "exit_b", "t_start": PALLET_T[0], "t_end": PALLET_T[1], "location": list(PALLET_XY)}
    )
    events.append(
        {
            "type": "false_alarm_source",
            "note": "Sun glare inside Exit B visible to cam3 only; no object present",
            "camera_id": "cam3",
            "t_start": GLARE_T[0],
            "t_end": GLARE_T[1],
            "location": list(GLARE_XY),
        }
    )
    cam4 = next(c for c in cams if c.camera_id == "cam4")
    events.append({"type": "camera_moved", "camera_id": "cam4", "t_start": cam4.bump_t, "yaw_deg": cam4.bump_yaw_deg, "pitch_deg": cam4.bump_pitch_deg})
    inside = [t for t, flag in cold_room_inside if flag]
    if inside:
        events.append({"type": "person_in_restricted", "zone_id": "cold_room", "track": "p2", "t_start": inside[0], "t_end": inside[-1]})
    events.sort(key=lambda e: e["t_start"])
    truth = {
        "fps": FPS,
        "duration_s": duration_s,
        "events": events,
        "cameras": [
            {
                "camera_id": c.camera_id,
                "K": c.K.tolist(),
                "R": c.pose(0)[0].tolist(),
                "t": c.pose(0)[1].tolist(),
                "bump_t": c.bump_t,
                "R_after": c.pose(1e9)[0].tolist(),
                "t_after": c.pose(1e9)[1].tolist(),
            }
            for c in cams
        ],
    }
    with open(os.path.join(out_dir, "truth_events.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)
    print(f"done: {n_frames} frames x {len(cams)} cameras in {time.time() - t_start:.0f}s -> {out_dir}")
    return truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sim/demo_001")
    ap.add_argument("--duration", type=float, default=DURATION_S)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--plates-only", action="store_true", help="only write the empty-floor setup images")
    a = ap.parse_args()
    if a.plates_only:
        os.makedirs(a.out, exist_ok=True)
        for path in write_setup_plates(a.out):
            print(path)
    else:
        run(a.out, a.duration, write_video=not a.no_video)
