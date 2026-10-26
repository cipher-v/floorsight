"""Re-test real-clip proximity alerts as floor distance, with uncertainty. GCP ONLY.

Inputs: the perception v5 per-frame log (person boxes and foot points, vehicle mask outlines),
its result (the pixel-ratio alerts), and a floor calibration from floor_calibration.py.

For every alert the clearance on the floor between the person and the vehicle footprint is
computed for each Monte Carlo draw of: calibration (segment jitter + standing-height prior),
vehicle size (the visible near-side ground line is extended away from the camera into a
rectangle of uncertain width and length, oriented by the vehicle's floor motion), the height
above the floor of the visible bottom edges, and the person's foot height (seated people sit on
chairs). Decision against the Floorsight near-miss distance (0.8 m):
  withdraw  - the 5th percentile of the minimum clearance is at least 0.8 m
  keep      - the median minimum clearance is below 0.8 m
  uncertain - otherwise (the agent would treat it as insufficient evidence, never as safe)
The floor map is drawn from the same numbers. Nothing here reads ground truth.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("cvsafe", __import__("pathlib").Path(__file__).with_name("cvsafe.py"))
cvsafe = _ilu.module_from_spec(_spec); _spec.loader.exec_module(cvsafe)

NEAR_MISS_M = 0.8
WHITE, GREY, BLACK = (255, 255, 255), (150, 150, 150), (20, 20, 20)
ORANGE, BLUE, RED, GREEN, YELLOW = (0, 140, 255), (255, 150, 40), (40, 40, 235), (60, 200, 60), (0, 215, 255)


def load_calib_module():
    spec = importlib.util.spec_from_file_location("fc", Path(__file__).with_name("floor_calibration.py"))
    fc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fc)
    return fc


def near_side_pixels_excluding(polygons, exclude, size, step=4, band=0.30, edge_px=3, min_pts=6):
    """near_side_pixels of the truck with the carried-load masks cut out first (v8, 23 Oct): crates on the forks sit
    above the floor, and an outline that includes a tall stack made the 'bottom 30 %' band reach up into the load, whose
    pixels project far behind the truck. Falls back to the plain union when the cut leaves too few points.
    Returns (points, union mask) - the mask of truck + loads is still the unit for edge-cut checks."""
    pts_all, union = near_side_pixels(list(polygons) + list(exclude), size, step, band, edge_px, max_band_of_width=0.25)
    if not exclude:
        return pts_all, union
    truck = np.zeros((size[1], size[0]), np.uint8)
    cvsafe.fill_poly(truck, [p for p in polygons if len(p) >= 3], 255)
    load = np.zeros_like(truck)
    cvsafe.fill_poly(load, [p for p in exclude if len(p) >= 3], 255)
    contact = cv2.bitwise_and(truck, cv2.bitwise_not(cv2.dilate(load, np.ones((5, 5), np.uint8))))
    cs = cv2.findContours(contact, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    cs = [c.reshape(-1, 2).astype(np.float32) for c in cs if cv2.contourArea(c) > 50]
    if not cs:
        return pts_all, union
    pts, _ = near_side_pixels(cs, size, step, band, edge_px, max_band_of_width=0.25)
    return (pts if len(pts) >= min_pts else pts_all), union


def near_side_pixels(polygons, size, step=4, band=0.30, edge_px=3, max_band_of_width=None):
    """Visible ground-contact line: lowest mask pixel per image column, kept only inside the bottom
    `band` of the mask's vertical extent, skipping image-edge cuts.

    v1 kept every column's lowest pixel. Side columns that contain only high parts (an overhanging
    load, the overhead guard) then contributed points metres above the floor, which project far away
    and stretched the 'ground line' (development job floorcal-03: forklift references 4x too long).
    """
    mask = np.zeros((size[1], size[0]), np.uint8)
    cvsafe.fill_poly(mask, [p for p in polygons if len(p) >= 3], 255)
    rows = np.nonzero(mask.any(axis=1))[0]
    if not len(rows):
        return np.zeros((0, 2)), mask
    cols = np.nonzero(mask.any(axis=0))[0]
    depth = band * (rows.max() - rows.min())
    if max_band_of_width is not None and len(cols):  # v8: a tall carried stack must not widen the contact band
        depth = min(depth, max_band_of_width * (cols.max() - cols.min()))
    floor_band = rows.max() - depth
    pts = []
    for x in cols[::step]:
        y = int(np.nonzero(mask[:, x])[0].max())
        if floor_band <= y < size[1] - edge_px:
            pts.append((x + 0.5, y + 1.0))
    return np.array(pts, float), mask


def rect_footprint(pts, heading, cam_xy, length, width):
    """Oriented rectangle through the visible near-side ground points, extended away from the camera."""
    u = np.array([np.cos(heading), np.sin(heading)])
    v = np.array([-u[1], u[0]])
    pu, pv = pts @ u, pts @ v
    cv_ = cam_xy @ v
    if cv_ <= pv.mean():
        v0, v1 = pv.min(), pv.min() + width
    else:
        v0, v1 = pv.max() - width, pv.max()
    u0, u1 = pu.min(), pu.max()
    if u1 - u0 < length:
        c = (u0 + u1) / 2
        u0, u1 = c - length / 2, c + length / 2
    corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    rect = np.array([a * u + b * v for a, b in corners])
    # The footprint must contain everything actually seen: hull of the model rectangle and the visible
    # ground points (v1 used the rectangle alone, which could leave seen points outside it).
    return cv2.convexHull(np.vstack([rect, pts]).astype(np.float32)).reshape(-1, 2).astype(float)


def point_rect_distance(p, poly):
    """Floor distance from a point to a footprint polygon (0 inside), via cv2.pointPolygonTest."""
    signed = cv2.pointPolygonTest(poly.astype(np.float32).reshape(-1, 1, 2), (float(p[0]), float(p[1])), True)
    return float(max(0.0, -signed))


def main() -> int:
    if int(cv2.__version__.split(".")[0]) != 5:
        raise SystemExit(f"OpenCV 5 required, found {cv2.__version__}")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--key-image", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--vehicle-width", type=float, nargs=2, default=[1.0, 1.3], help="m, uniform range (forklift or carried stillage)")
    ap.add_argument("--vehicle-length", type=float, nargs=2, default=[2.2, 3.8], help="m, uniform range (forklift with or without load)")
    ap.add_argument("--contact-height", type=float, nargs=2, default=[0.0, 0.25], help="m above floor of the lowest visible vehicle edge")
    a = ap.parse_args()
    fc = load_calib_module()
    calib = json.loads(a.calibration.read_text())
    result = json.loads(a.result.read_text())
    size = tuple(calib["imageSize"])
    fps = float(result["fps"])
    nominal = fc.Camera(calib["K"], calib["R_worldAxesInCamera"], calib["cameraHeight_m"])
    cams = [nominal] + [fc.Camera(s["K"], s["R"], s["h"]) for s in calib["monteCarloSamples"]]
    rng = np.random.default_rng(7)
    draws = [(cams[0], 0.0, 0.0, float(np.mean(a.vehicle_width)), float(np.mean(a.vehicle_length)))]
    for _ in range(a.draws):
        draws.append((cams[1 + rng.integers(len(cams) - 1)] if len(cams) > 1 else cams[0],
                      rng.uniform(*a.contact_height), rng.uniform(0.0, 0.15), rng.uniform(*a.vehicle_width), rng.uniform(*a.vehicle_length)))

    records = [json.loads(l) for l in a.detections.read_text().splitlines()]
    # Per frame: vehicle near-side pixels and masks; per person: foot pixel, seated flag, occlusion flag.
    frames = {}
    for r in records:
        vs, near = [], np.zeros((size[1], size[0]), np.uint8)
        for v in r.get("vehicles", []):
            if not v.get("polygons"):
                continue
            pts, mask = near_side_pixels(v["polygons"], size)
            near |= mask
            if len(pts) >= 4 and not v.get("estimated"):
                vs.append({"id": v["id"], "px": pts})
        near = cv2.dilate(near, np.ones((25, 25), np.uint8))  # a foot this close to a vehicle mask may be hidden
        ps = []
        for p in r.get("persons", []):
            x1, y1, x2, y2 = p["box"]
            foot = np.array(p["foot"], float)
            occluded = bool(near[int(np.clip(foot[1], 0, size[1] - 1)), int(np.clip(foot[0], 0, size[0] - 1))] > 0)
            ps.append({"id": p["id"], "foot": foot, "seated": (y2 - y1) < 2.0 * (x2 - x1), "occluded": occluded, "role": p.get("role")})
        frames[r["frame"]] = {"t": r["t"], "vehicles": vs, "persons": ps}

    def vehicle_floor(cam, vid, contact):
        """Per-frame floor points of the near-side line and a heading from floor motion (1 s window)."""
        cent = {f: (cam.to_plane(v["px"], contact), f) for f, fr in frames.items() for v in fr["vehicles"] if v["id"] == vid}
        cent = {f: p for f, (p, _) in cent.items() if np.isfinite(p).all()}
        keys = sorted(cent)
        mids = {f: np.nanmean(cent[f], 0) for f in keys}
        heading = {}
        for f in keys:
            near = [k for k in keys if abs(k - f) <= fps / 2]
            d = mids[near[-1]] - mids[near[0]]
            if np.linalg.norm(d) >= 0.3:
                heading[f] = float(np.arctan2(d[1], d[0]))
            else:  # nearly still: orient by the visible ground line itself (OpenCV minAreaRect)
                (_, _), (w, h), ang = cv2.minAreaRect(cent[f].astype(np.float32))
                heading[f] = float(np.radians(ang if w >= h else ang + 90))
        return cent, heading

    def person_floor(cam, pid, foot_h, frame_ids):
        pts = [cam.to_plane(fr_p["foot"], foot_h)[0] for f in frame_ids for fr_p in frames[f]["persons"] if fr_p["id"] == pid and not fr_p["occluded"]]
        pts = np.array([p for p in pts if np.isfinite(p).all()])
        return pts

    out_events = []
    a.out.mkdir(parents=True, exist_ok=False)
    for ev in result.get("events", []):
        pid, vid = ev["person"], ev["vehicle"]
        f0 = int(max(0, (ev["startT"] - 0.5) * fps)); f1 = int(ev["endT"] * fps)
        window = [f for f in sorted(frames) if f0 <= f <= f1]
        seated = np.mean([p["seated"] for f in window for p in frames[f]["persons"] if p["id"] == pid] or [0]) > 0.5
        all_frames = sorted(frames)
        per_draw, per_draw_upper, per_draw_frame, closing = [], [], [], []
        person_pts_nominal, rects_nominal, near_nominal = None, {}, {}
        for k, (cam, contact, foot_h, width, length) in enumerate(draws):
            fh = foot_h if seated else 0.0
            ppts = person_floor(cam, pid, fh, all_frames)  # stationary check uses the whole clip
            if not len(ppts):
                continue
            spread = float(np.linalg.norm(np.percentile(ppts, 90, 0) - np.percentile(ppts, 10, 0)))
            stationary = spread < 0.6
            cent, heading = vehicle_floor(cam, vid, contact)
            cam_xy = np.zeros(2)
            best, best_up, best_f, series = np.inf, np.inf, None, []
            for f in window:
                if f not in cent:
                    continue
                if stationary:
                    p = np.median(ppts, 0)
                else:
                    pf = person_floor(cam, pid, fh, [f])
                    if not len(pf):
                        continue
                    p = pf[0]
                rect = rect_footprint(cent[f], heading[f], cam_xy, length, width)
                d = point_rect_distance(p, rect)
                d_up = float(np.min(np.linalg.norm(cent[f] - p, axis=1)))
                series.append((f, d))
                if d < best:
                    best, best_f = d, f
                best_up = min(best_up, d_up)
                if k == 0:
                    rects_nominal[f] = rect; near_nominal[f] = cent[f]
            if not np.isfinite(best):
                continue
            if k == 0:
                person_pts_nominal = (np.median(ppts, 0) if stationary else None, ppts, stationary, spread)
            per_draw.append(best); per_draw_upper.append(best_up); per_draw_frame.append(best_f)
            sd = dict(series)
            if best_f - 3 in sd and best_f in sd:
                closing.append((sd[best_f - 3] - sd[best_f]) / (3 / fps))
        if not per_draw:
            out_events.append({"event": ev, "decision": "not computable", "reason": "no unoccluded person floor points or no vehicle mask in the window"})
            continue
        c = np.array(per_draw)
        p5, p50, p95 = (float(np.percentile(c, q)) for q in (5, 50, 95))
        decision = "withdraw" if p5 >= NEAR_MISS_M else "keep" if p50 < NEAR_MISS_M else "uncertain"
        nominal_f = per_draw_frame[0]
        visible_len = [float(np.linalg.norm(near_nominal[f].max(0) - near_nominal[f].min(0))) for f in near_nominal]
        out_events.append({
            "event": ev, "person": pid, "vehicle": vid, "personSeated": bool(seated),
            "personStationary": bool(person_pts_nominal[2]) if person_pts_nominal else None,
            "personFloorSpread90_m": round(person_pts_nominal[3], 3) if person_pts_nominal else None,
            "minClearance_m": {"median": round(p50, 2), "p5": round(p5, 2), "p95": round(p95, 2), "nominal": round(per_draw[0], 2)},
            "nearSideOnlyDistance_m": {"median": round(float(np.median(per_draw_upper)), 2), "note": "distance to the visible ground line only (upper bound when the person is beyond the vehicle)"},
            "closingSpeedAtMin_mps": round(float(np.median(closing)), 2) if closing else None,
            "minAt": {"frame": nominal_f, "t": round(nominal_f / fps, 2)},
            "visibleGroundLineLength_m": round(float(np.median(visible_len)), 2) if visible_len else None,
            "decision": decision, "rule": f"withdraw if p5 >= {NEAR_MISS_M} m; keep if median < {NEAR_MISS_M} m; else uncertain",
            "draws": len(per_draw),
        })
        # ---------------- floor map (nominal draw) ----------------
        allpts = [np.zeros((1, 2))] + list(rects_nominal.values()) + ([person_pts_nominal[1]] if person_pts_nominal else [])
        allpts = np.vstack(allpts)
        lo, hi = allpts.min(0) - 1.5, allpts.max(0) + 1.5
        W, H = 1000, 800
        s = min((W - 80) / (hi[0] - lo[0]), (H - 140) / (hi[1] - lo[1]))
        def m2p(xy):
            xy = np.atleast_2d(xy)
            return np.column_stack([40 + (xy[:, 0] - lo[0]) * s, H - 60 - (xy[:, 1] - lo[1]) * s]).astype(np.int32)
        fm = np.full((H, W, 3), 245, np.uint8)
        for gx in np.arange(np.floor(lo[0]), np.ceil(hi[0]) + 1):
            cv2.line(fm, tuple(m2p([gx, lo[1]])[0]), tuple(m2p([gx, hi[1]])[0]), (200, 200, 200) if gx % 5 else (160, 160, 160), 1)
        for gy in np.arange(np.floor(lo[1]), np.ceil(hi[1]) + 1):
            cv2.line(fm, tuple(m2p([lo[0], gy])[0]), tuple(m2p([hi[0], gy])[0]), (200, 200, 200) if gy % 5 else (160, 160, 160), 1)
        # camera and its field of view on the floor
        edge = nominal.to_plane(np.array([[0, size[1] - 1], [size[0] - 1, size[1] - 1]]))
        cv2.circle(fm, tuple(m2p([0, 0])[0]), 9, BLACK, -1, cv2.LINE_AA)
        for e in edge:
            if np.isfinite(e).all():
                far = e / np.linalg.norm(e) * np.linalg.norm(hi - lo)
                cv2.line(fm, tuple(m2p([0, 0])[0]), tuple(m2p(far)[0]), (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(fm, "camera", tuple(m2p([0, 0])[0] + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
        for f in sorted(rects_nominal)[::5]:
            cv2.polylines(fm, [m2p(rects_nominal[f])], True, (120, 190, 255), 1, cv2.LINE_AA)
        traj = np.array([near_nominal[f].mean(0) for f in sorted(near_nominal)])
        cv2.polylines(fm, [m2p(traj)], False, ORANGE, 2, cv2.LINE_AA)
        if nominal_f in rects_nominal:
            layer = fm.copy(); cvsafe.fill_poly(layer, [m2p(rects_nominal[nominal_f])], ORANGE); fm = cv2.addWeighted(layer, 0.45, fm, 0.55, 0)
            cv2.polylines(fm, [m2p(rects_nominal[nominal_f])], True, ORANGE, 2, cv2.LINE_AA)
            cv2.polylines(fm, [m2p(near_nominal[nominal_f])], False, (0, 80, 200), 3, cv2.LINE_AA)
        if person_pts_nominal is not None:
            pp = person_pts_nominal[1]
            for q in pp[:: max(1, len(pp) // 60)]:
                cv2.circle(fm, tuple(m2p(q)[0]), 2, (220, 180, 120), -1)
            pc = np.median(pp, 0)
            cv2.circle(fm, tuple(m2p(pc)[0]), 9, BLUE, -1, cv2.LINE_AA)
            cv2.putText(fm, f"person {pid}" + (" (seated)" if seated else ""), tuple(m2p(pc)[0] + [12, 4]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLUE, 2, cv2.LINE_AA)
            if nominal_f in rects_nominal:
                r = rects_nominal[nominal_f]
                best_pt, bd = None, np.inf
                for i in range(len(r)):
                    a_, b_ = r[i], r[(i + 1) % len(r)]
                    t_ = np.clip((pc - a_) @ (b_ - a_) / max((b_ - a_) @ (b_ - a_), 1e-12), 0, 1)
                    q_ = a_ + t_ * (b_ - a_)
                    if np.linalg.norm(pc - q_) < bd:
                        bd, best_pt = np.linalg.norm(pc - q_), q_
                col = RED if decision == "keep" else (0, 150, 230) if decision == "uncertain" else GREEN
                cv2.line(fm, tuple(m2p(pc)[0]), tuple(m2p(best_pt)[0]), col, 2, cv2.LINE_AA)
                anchor = m2p(best_pt + 0.35 * (pc - best_pt))[0] + np.array([-40, 34])  # below the line, clear of the person label
                txt = f"{p50:.1f} m"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(fm, tuple(anchor + [-4, -th - 6]), tuple(anchor + [tw + 4, 6]), WHITE, -1)
                cv2.putText(fm, txt, tuple(anchor), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        # 1 m scale bar and text
        cv2.line(fm, (W - 160, H - 30), (W - 160 + int(s), H - 30), BLACK, 3)
        cv2.putText(fm, "1 m", (W - 160, H - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
        verdict = {"withdraw": "ALERT WITHDRAWN", "keep": "ALERT KEPT", "uncertain": "UNCERTAIN - insufficient evidence"}[decision]
        cv2.rectangle(fm, (0, 0), (W, 70), (40, 40, 40), -1)
        cv2.putText(fm, f"Floor map  {a.name}  -  {verdict}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
        cv2.putText(fm, f"min clearance {p50:.1f} m (90%: {p5:.1f}-{p95:.1f}); near-miss rule 0.8 m; scale from assumed person height + forklift length",
                    (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(fm, "orange: forklift footprint (estimated) at closest moment, outlines every 5 frames; dark: visible ground line; blue: person",
                    (16, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.imwrite(str(a.out / f"{a.name}-{pid}-{vid}-floor-map.jpg"), fm, [cv2.IMWRITE_JPEG_QUALITY, 92])
        key = cv2.imread(str(a.key_image))
        if key is not None:
            key = cv2.resize(key, (int(key.shape[1] * H / key.shape[0]), H))
            cv2.putText(key, "BEFORE: image-plane ratio (uncalibrated)", (16, H - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(key, "BEFORE: image-plane ratio (uncalibrated)", (16, H - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
            cv2.imwrite(str(a.out / f"{a.name}-{pid}-{vid}-before-after.jpg"), np.hstack([key, fm]), [cv2.IMWRITE_JPEG_QUALITY, 90])

    report = {"schema": "floorsight.floor-retest.v1", "name": a.name, "opencv": cv2.__version__,
              "createdAt": datetime.now(timezone.utc).isoformat(), "calibration": str(a.calibration),
              "cameraHeight_m": calib["cameraHeight_m"], "cameraHeightRelSigma": calib["cameraHeightRelSigma"],
              "priors": {"vehicleWidth_m": a.vehicle_width, "vehicleLength_m": a.vehicle_length, "contactHeight_m": a.contact_height,
                         "seatedFootHeight_m": [0.0, 0.15], "standingHeight_m": [1.70, 0.08]},
              "events": out_events,
              "limitations": ["Single fixed camera; the vehicle's far side is modelled, not observed.",
                              "Scale from an assumed adult height; no surveyed floor distance exists for this site.",
                              "Development clips, not a held-out evaluation; no ground truth exists for these distances."]}
    (a.out / f"{a.name}-floor-retest.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
