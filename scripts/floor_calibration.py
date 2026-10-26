"""Floor-plane calibration for an uncalibrated fixed camera, from pixels only. GCP ONLY.

Single-view metrology (vanishing points + one reference height), implemented with OpenCV 5:
  1. `segments`: temporal-median background per clip (raw pixels stay in this process; the
     written image has faces and heads mosaicked first), a same-view check between clips (ORB +
     findHomography), line segments (LSD, else HoughLinesP), and an automatic RANSAC estimate of
     the three dominant vanishing points as a cross-check. Writes numbered segment overlays.
  2. `solve`: vanishing points from human-selected segment families (A: floor lines along the
     lane, B: floor lines across it, V: vertical edges) -> focal length (orthogonality, principal
     point at the image centre, checked against the vanishing-point orthocentre) -> camera
     rotation -> floor homography up to scale. Scale = camera height from standing people in the
     perception log: each person's height in camera-height units is measured, and an assumed adult
     standing height of 1.70 +/- 0.08 m converts it to metres. The spread of those heights across
     image positions is the check on focal length and tilt. Monte Carlo samples (segment jitter +
     height prior) are written for downstream uncertainty.

The dataset's published floor plan has lane layout and camera positions but no dimensions, so it
cannot supply scale; it is used only to name the lane direction. Everything here is an estimate
with stated uncertainty, not a surveyed calibration.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PERSON_HEIGHT_M, PERSON_HEIGHT_SIGMA_M = 1.70, 0.08
WHITE, GREY = (255, 255, 255), (170, 170, 170)
FAMILY_COLOURS = {"A": (0, 200, 255), "B": (255, 120, 0), "V": (60, 220, 60)}


def load_rcp(models: Path):
    """Reuse the perception script's exact privacy functions (same folder on GCP)."""
    spec = importlib.util.spec_from_file_location("rcp", Path(__file__).with_name("real_clip_perception.py"))
    rcp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rcp)
    return rcp


def median_background(clip: Path, samples: int = 41) -> np.ndarray:
    cap = cv2.VideoCapture(str(clip))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(0, total - 1), samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def privacy_filter(img: np.ndarray, rcp, models: Path) -> tuple[np.ndarray, dict]:
    """Mosaic faces (YuNet) and the head region of every person candidate (YOLOX via cv2.dnn, conf >= 0.10)."""
    spec = importlib.util.spec_from_file_location("open_detectors", Path(__file__).with_name("open_detectors.py"))
    od = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(od)
    onnx = models / "onnx" if (models / "onnx").is_dir() else models
    out = img.copy()
    people = od.YoloxDnn(onnx / "yolox_l.onnx", tiles=3, conf=0.10).detect(img)[od.COCO_PERSON]
    heads = sum(rcp.mosaic(out, rcp.head_region(b)) for b, _ in people)
    det = cv2.FaceDetectorYN.create(str(onnx / "face_detection_yunet_2023mar.onnx"), "", (img.shape[1], img.shape[0]), 0.6)
    _, faces = det.detect(img)
    nf = 0 if faces is None else sum(rcp.mosaic(out, (f[0] - 4, f[1] - 4, f[0] + f[2] + 4, f[1] + f[3] + 4)) for f in faces)
    return out, {"headsMosaicked": int(heads), "facesMosaicked": int(nf)}


def detect_segments(gray: np.ndarray, min_len: float = 40.0, keep: int = 260) -> tuple[np.ndarray, str]:
    try:
        lines = cv2.createLineSegmentDetector().detect(gray)[0]
        method = "LSD (cv2.createLineSegmentDetector)"
    except Exception:
        lines = None
    if lines is None or len(lines) == 0:
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 60, minLineLength=min_len, maxLineGap=4)
        method = "HoughLinesP on Canny"
    seg = np.asarray(lines, np.float64).reshape(-1, 4)
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    seg, length = seg[length >= min_len], length[length >= min_len]
    order = np.argsort(-length)[:keep]
    return seg[order], method


def seg_line(s: np.ndarray) -> np.ndarray:
    l = np.cross([s[0], s[1], 1.0], [s[2], s[3], 1.0])
    return l / np.linalg.norm(l[:2])


def vanishing_point(segs: np.ndarray) -> np.ndarray:
    """Least-squares intersection of segment lines (length-weighted); homogeneous, unit norm."""
    L = np.array([seg_line(s) for s in segs])
    w = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    _, _, vt = np.linalg.svd(L * w[:, None])
    v = vt[-1]
    return v / np.linalg.norm(v)


def angle_to_vp(s: np.ndarray, v: np.ndarray) -> float:
    """Angle (deg) between a segment and the line from its midpoint to the vanishing point."""
    m = np.array([(s[0] + s[2]) / 2, (s[1] + s[3]) / 2])
    d = np.array([s[2] - s[0], s[3] - s[1]])
    to = v[:2] - m * v[2]  # direction to VP (works for VPs at infinity too)
    c = abs(d @ to) / (np.linalg.norm(d) * np.linalg.norm(to) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, 0, 1))))


def ransac_vps(segs: np.ndarray, n: int = 3, tol_deg: float = 1.5, iters: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(segs))
    found = []
    for _ in range(n):
        best = (0, None, None)
        for _ in range(iters):
            if len(remaining) < 2:
                break
            i, j = rng.choice(remaining, 2, replace=False)
            v = np.cross(seg_line(segs[i]), seg_line(segs[j]))
            if np.linalg.norm(v) < 1e-12:
                continue
            v = v / np.linalg.norm(v)
            inl = remaining[[angle_to_vp(segs[k], v) < tol_deg for k in remaining]]
            score = np.hypot(segs[inl, 2] - segs[inl, 0], segs[inl, 3] - segs[inl, 1]).sum()
            if score > best[0]:
                best = (score, v, inl)
        if best[1] is None:
            break
        v = vanishing_point(segs[best[2]])
        found.append({"vp": v.tolist(), "inliers": best[2].tolist()})
        remaining = np.setdiff1d(remaining, best[2])
    return found


def draw_segments(img: np.ndarray, segs: np.ndarray, families: dict | None = None) -> np.ndarray:
    out = img.copy()
    fam_of = {i: f for f, ids in (families or {}).items() for i in ids}
    for i, s in enumerate(segs):
        if families is not None and i not in fam_of:
            continue
        ang = (np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0])) + 180) % 180
        col = FAMILY_COLOURS[fam_of[i]] if i in fam_of else tuple(int(c) for c in cv2.cvtColor(np.uint8([[[ang, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        p, q = (int(s[0]), int(s[1])), (int(s[2]), int(s[3]))
        cv2.line(out, p, q, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(out, p, q, col, 2, cv2.LINE_AA)
        m = ((p[0] + q[0]) // 2, (p[1] + q[1]) // 2)
        cv2.putText(out, str(i), (m[0] + 3, m[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, str(i), (m[0] + 3, m[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return out


def cmd_segments(a) -> int:
    rcp = load_rcp(a.models)
    a.out.mkdir(parents=True, exist_ok=False)
    meds, report = {}, {"schema": "floorsight.floor-segments.v1", "opencv": cv2.__version__, "clips": {}}
    for clip in a.clip:
        name = clip.stem
        med = median_background(clip)
        meds[name] = med
        filtered, priv = privacy_filter(med, rcp, a.models)
        cv2.imwrite(str(a.out / f"{name}-median-filtered.jpg"), filtered, [cv2.IMWRITE_JPEG_QUALITY, 92])
        segs, method = detect_segments(cv2.cvtColor(med, cv2.COLOR_BGR2GRAY))
        np.save(a.out / f"{name}-segments.npy", segs)
        vps = ransac_vps(segs)
        cv2.imwrite(str(a.out / f"{name}-segments.jpg"), draw_segments(filtered, segs), [cv2.IMWRITE_JPEG_QUALITY, 92])
        auto = {f"auto{k}": v["inliers"] for k, v in enumerate(vps)}
        cv2.imwrite(str(a.out / f"{name}-auto-vp-families.jpg"),
                    draw_segments(filtered, segs, {"A": auto.get("auto0", []), "B": auto.get("auto1", []), "V": auto.get("auto2", [])}),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        report["clips"][name] = {"privacy": priv, "segmentMethod": method, "segments": segs.round(2).tolist(),
                                 "autoVanishingPoints": vps, "size": [med.shape[1], med.shape[0]]}
    names = list(meds)
    if len(names) == 2:
        g1, g2 = (cv2.cvtColor(meds[n], cv2.COLOR_BGR2GRAY) for n in names)
        orb = cv2.ORB_create(4000)
        k1, d1 = orb.detectAndCompute(g1, None)
        k2, d2 = orb.detectAndCompute(g2, None)
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
        src = np.float32([k1[m.queryIdx].pt for m in matches]); dst = np.float32([k2[m.trainIdx].pt for m in matches])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        h, w = g1.shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        shift = np.linalg.norm(cv2.perspectiveTransform(corners, H) - corners, axis=2).max() if H is not None else None
        report["sameView"] = {"clips": names, "matches": len(matches), "inliers": int(inl.sum()) if inl is not None else 0,
                              "maxCornerShiftPx": None if shift is None else round(float(shift), 2),
                              "sameView": bool(shift is not None and shift < 4.0)}
    (a.out / "segments.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "clips"}, indent=1))
    for n, c in report["clips"].items():
        print(n, c["segmentMethod"], len(c["segments"]), "segments; auto VPs", [np.round(np.array(v["vp"][:2]) / (v["vp"][2] if abs(v["vp"][2]) > 1e-9 else 1e-9), 1).tolist() for v in c["autoVanishingPoints"]],
              "inliers", [len(v["inliers"]) for v in c["autoVanishingPoints"]], c["privacy"])
    return 0


# ----------------------------------------------------------------------------- solve

def focal_from(v1, v2, c) -> float | None:
    p = v1[:2] / v1[2] - c if abs(v1[2]) > 1e-12 else None
    q = v2[:2] / v2[2] - c if abs(v2[2]) > 1e-12 else None
    if p is None or q is None:
        return None
    f2 = -(p @ q)
    return float(np.sqrt(f2)) if f2 > 0 else None


def orthocentre(vs) -> list | None:
    try:
        P = [v[:2] / v[2] for v in vs]
        A = np.array([P[1] - P[2], P[0] - P[2]])
        b = np.array([P[0] @ (P[1] - P[2]), P[1] @ (P[0] - P[2])])
        return np.linalg.solve(A, b).tolist()
    except Exception:
        return None


class Camera:
    """K, world axes in camera coordinates (X along A, Y across, Z up), camera at (0, 0, h)."""

    def __init__(self, K, R, h):
        self.K, self.R, self.h = np.asarray(K, float), np.asarray(R, float), float(h)
        self.Kinv = np.linalg.inv(self.K)

    @classmethod
    def from_vps(cls, vA, vB, vV, size, f=None):
        """vB may be None: A and V alone fix focal length (orthogonality) and rotation (Y = up x A)."""
        c = np.array([size[0] / 2, size[1] / 2])
        fs = [x for x in (focal_from(vA, vB, c) if vB is not None else None, focal_from(vA, vV, c),
                          focal_from(vB, vV, c) if vB is not None else None) if x]
        f = f or (focal_from(vA, vV, c) if vB is None else (focal_from(vA, vB, c) or float(np.median(fs))))
        K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.0]])
        Ki = np.linalg.inv(K)
        def d(v):
            x = Ki @ v
            x = x / np.linalg.norm(x)
            return x if x[2] > 0 else -x
        down = d(vV)
        if down[1] < 0:  # the vanishing point of world-down lies below the centre for a down-looking camera
            down = -down
        up = -down
        r1 = d(vA); r1 = r1 - (r1 @ up) * up; r1 /= np.linalg.norm(r1)
        r2 = np.cross(up, r1)
        return cls(K, np.column_stack([r1, r2, up]), 1.0), {"focalPx": f, "focalEstimates": fs}

    def ray(self, px):
        px = np.atleast_2d(px).astype(float)
        dc = (self.Kinv @ np.column_stack([px, np.ones(len(px))]).T).T  # camera coordinates
        return dc @ self.R  # world coordinates (R columns = world axes in camera coords)

    def to_plane(self, px, z=0.0):
        """World (X, Y) where the ray through px meets the horizontal plane Z = z."""
        dw = self.ray(px)
        t = (np.atleast_1d(z) - self.h) / dw[:, 2]
        pts = np.array([0.0, 0.0, self.h]) + t[:, None] * dw
        pts[t <= 0] = np.nan  # above the horizon for this plane
        return pts[:, :2]

    def project(self, P):
        P = np.atleast_2d(P).astype(float)
        xc = (P - np.array([0.0, 0.0, self.h])) @ self.R.T  # camera coordinates
        uv = (self.K @ xc.T).T
        out = uv[:, :2] / uv[:, 2:3]
        out[xc[:, 2] <= 1e-6] = np.nan  # behind the camera
        return out

    def H_img_to_floor(self):
        H = self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], -self.h * self.R[:, 2]])
        return np.linalg.inv(H) / np.linalg.inv(H)[2, 2]

    def height_of(self, foot_px, head_px):
        """Height (m, in this camera's scale) of a vertical object standing at foot_px whose top is at head_px."""
        X = self.to_plane(foot_px)[0]
        if np.any(np.isnan(X)):
            return np.nan
        zs = np.linspace(0.0, 4.0 * self.h, 4001)
        uv = self.project(np.column_stack([np.full_like(zs, X[0]), np.full_like(zs, X[1]), zs]))
        return float(zs[np.argmin(np.linalg.norm(uv - np.asarray(head_px, float), axis=1))])


def standing_people(detections: Path, size, max_obs: int = 400):
    """(foot_px, head_px, track, row) of confidently detected, upright, unoccluded people."""
    obs = []
    for line in detections.read_text().splitlines():
        r = json.loads(line)
        vboxes = [v["box"] for v in r.get("vehicles", [])]
        for p in r.get("persons", []):
            x1, y1, x2, y2 = p["box"]
            w, h = x2 - x1, y2 - y1
            if p.get("conf", 0) < 0.5 or h < 2.2 * w or h < 60 or "role" in p:
                continue
            if x1 < 3 or y1 < 95 or x2 > size[0] - 3 or y2 > size[1] - 3:
                continue
            if any(not (x2 < b[0] or x1 > b[2] or y2 < b[1] or y1 > b[3]) for b in vboxes):
                continue  # touching a vehicle box: lower body may be hidden
            obs.append(((x1 + x2) / 2, y2, (x1 + x2) / 2, y1, p["id"], r["frame"]))
    if len(obs) > max_obs:
        obs = [obs[i] for i in np.linspace(0, len(obs) - 1, max_obs).astype(int)]
    return obs


def fit_yellow_lane(img: np.ndarray, region) -> dict | None:
    """Painted yellow lane line: HSV paint mask inside a floor region, robust line fit (cv2.fitLine)."""
    x0, y0, x1, y1 = region
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15, 70, 90), (40, 255, 255))
    roi = np.zeros_like(mask); roi[y0:y1, x0:x1] = 255
    mask = cv2.morphologyEx(cv2.bitwise_and(mask, roi), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(mask)
    if len(xs) < 200:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    d = (pts - [cx, cy]) @ np.array([vx, vy])
    dist = np.abs((pts - [cx, cy]) @ np.array([-vy, vx]))
    inl = dist < 6
    t0, t1 = np.percentile(d[inl], [1, 99])
    return {"segment": [float(cx + t0 * vx), float(cy + t0 * vy), float(cx + t1 * vx), float(cy + t1 * vy)],
            "pixels": int(len(xs)), "inlierShare": float(inl.mean()), "medianResidualPx": float(np.median(dist[inl]))}


def mosaic_detected_heads(img: np.ndarray, detections, rcp, every: int = 5) -> int:
    """Privacy for figures built on the median: mosaic the head region of every person box perception
    logged (every `every` frames), so a person who stayed still enough to survive the median is covered."""
    n = 0
    for det in detections:
        for k, line in enumerate(det.read_text().splitlines()):
            if k % every:
                continue
            for p in json.loads(line).get("persons", []):
                n += rcp.mosaic(img, rcp.head_region(p["box"]))
    return n


def cmd_solve(a) -> int:
    seg_report = json.loads(a.segments.read_text())
    clip = seg_report["clips"][a.clip_name]
    segs = np.array(clip["segments"])
    size = clip["size"]
    auto = {f"auto{k}": v["inliers"] for k, v in enumerate(clip["autoVanishingPoints"])}
    sel = {}
    for s in a.select:  # A=auto0 | A=1,2,3 ; B is optional
        k, v = s.split("=")
        sel[k] = list(auto[v]) if v in auto else [int(x) for x in v.split(",") if x]
    for k, drop in (s.split("=") for s in a.exclude):
        sel[k] = [i for i in sel[k] if i not in {int(x) for x in drop.split(",") if x}]
    vA, vV = vanishing_point(segs[sel["A"]]), vanishing_point(segs[sel["V"]])
    vB = vanishing_point(segs[sel["B"]]) if sel.get("B") else None
    cam, finfo = Camera.from_vps(vA, vB, vV, size, f=a.focal)
    if a.focal:
        finfo["focalSource"] = f"fixed by the focal sweep (+/- {a.focal_sigma} px); vanishing-point value kept for reference"
        finfo["focalFromVanishingPoints"] = focal_from(vA, vV, np.array([size[0] / 2, size[1] / 2]))
    residual = {k: float(np.median([angle_to_vp(segs[i], v) for i in sel[k]])) for k, v in (("A", vA), ("B", vB), ("V", vV)) if v is not None}
    # Near-field size reference: the longest visible ground line of the moving forklift-plus-load unit.
    ground = []
    if a.forklift_long:
        spec = importlib.util.spec_from_file_location("fr", Path(__file__).with_name("floor_retest.py"))
        fr = importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)
        for det in a.detections:
            for line in det.read_text().splitlines():
                for v in json.loads(line).get("vehicles", []):
                    if v.get("polygons") and v.get("moving") and not v.get("estimated"):
                        pts, _ = fr.near_side_pixels(v["polygons"], size)
                        if len(pts) >= 6:
                            ground.append(pts)

    def fork_long_units(c):
        ls = []
        for pts in ground:
            p = c.to_plane(pts)
            p = p[np.isfinite(p).all(1)]
            if len(p) >= 6:
                q = (p - p.mean(0)) @ np.linalg.svd(p - p.mean(0), full_matrices=False)[2][0]
                ls.append(q.max() - q.min())
        return float(np.percentile(ls, 95)) if ls else float("nan")
    obs = []
    for n, det in enumerate(a.detections):  # track ids restart per clip: key them by clip
        obs += [(*o[:4], f"{n}:{o[4]}", o[5]) for o in standing_people(det, size)]
    z = np.array([cam.height_of(o[:2], o[2:4]) for o in obs])  # in camera-height units (h = 1)
    ok = np.isfinite(z) & (z > 0.05)
    z, obs = z[ok], [o for o, k in zip(obs, ok) if k]
    tracks = sorted(set(o[4] for o in obs))
    zt = np.array([np.median([zz for zz, o in zip(z, obs) if o[4] == t]) for t in tracks]) if tracks else np.array([])
    z_med = float(np.median(zt)) if len(zt) else float("nan")
    h = h_people = PERSON_HEIGHT_M / z_med
    rel_spread = float(np.std(zt) / np.mean(zt)) if len(zt) > 1 else float("nan")
    rel_sigma = rel_people = float(np.hypot(PERSON_HEIGHT_SIGMA_M / PERSON_HEIGHT_M, (rel_spread if np.isfinite(rel_spread) else 0.05) / np.sqrt(max(1, len(zt)))))
    h_fork = rel_fork = None
    if ground:
        long_u = fork_long_units(cam)
        h_fork, rel_fork = a.forklift_long[0] / long_u, a.forklift_long[1] / a.forklift_long[0]
        w_p, w_f = 1 / rel_people ** 2, 1 / rel_fork ** 2  # inverse-variance mean in log space
        h = float(np.exp((w_p * np.log(h_people) + w_f * np.log(h_fork)) / (w_p + w_f)))
        rel_sigma = float(1 / np.sqrt(w_p + w_f))
    rows = np.array([o[1] for o in obs])
    slope = float(np.polyfit(rows, z * h, 1)[0] * 100) if len(rows) > 5 else None  # metres per 100 px of image row
    cam = Camera(cam.K, cam.R, h)

    # Monte Carlo: jitter segment endpoints (1 px) and, if fixed by the sweep, the focal length; re-solve;
    # re-scale with draws from both reference priors (people height, forklift-plus-load length).
    rng = np.random.default_rng(1)
    samples = []
    for _ in range(a.samples):
        js = segs + rng.normal(0, 1.0, segs.shape)
        f_s = float(rng.normal(a.focal, a.focal_sigma)) if a.focal else None
        try:
            c2, _ = Camera.from_vps(vanishing_point(js[sel["A"]]), vanishing_point(js[sel["B"]]) if sel.get("B") else None,
                                    vanishing_point(js[sel["V"]]), size, f=f_s)
        except Exception:
            continue
        zz = np.array([c2.height_of(o[:2], o[2:4]) for o in obs[:: max(1, len(obs) // 60)]])
        zz = zz[np.isfinite(zz) & (zz > 0.05)]
        if not len(zz):
            continue
        hp = rng.normal(PERSON_HEIGHT_M, PERSON_HEIGHT_SIGMA_M) / float(np.median(zz))
        h2 = hp
        if ground:
            hf = rng.normal(*a.forklift_long) / fork_long_units(c2)
            h2 = float(np.exp((np.log(hp) / rel_people ** 2 + np.log(hf) / rel_fork ** 2) / (1 / rel_people ** 2 + 1 / rel_fork ** 2)))
        samples.append({"K": c2.K.round(4).tolist(), "R": c2.R.round(6).tolist(), "h": round(h2, 4)})

    a.out.mkdir(parents=True, exist_ok=False)
    median = cv2.imread(str(a.median))
    # Independent check: the painted yellow lane, back-projected to the floor, should make a sensible
    # angle with family A (a wrong focal length or tilt bends floor angles). Not used in the solve.
    lane = fit_yellow_lane(median, tuple(int(x) for x in a.lane_region.split(","))) if a.lane_region else None
    if lane:
        def lane_angle(c):
            p = c.to_plane(np.array(lane["segment"]).reshape(2, 2))
            d = p[1] - p[0]
            return float(np.degrees(np.arctan2(d[1], d[0])) % 180), float(np.linalg.norm(d))
        lane["angleToA_deg"], lane["length_m"] = lane_angle(cam)
        mc = [lane_angle(Camera(s["K"], s["R"], s["h"]))[0] for s in samples]
        lane["angleToA_deg_p5_p95"] = [float(np.percentile(mc, 5)), float(np.percentile(mc, 95))] if mc else None
    grid = median.copy()
    heads = mosaic_detected_heads(grid, a.detections, load_rcp(Path(".")))  # privacy before any drawing
    # 1 m floor grid around the camera's view, clipped to the visible floor.
    corners = cam.to_plane(np.array([[0, size[1] * 0.35], [size[0], size[1] * 0.35], [size[0], size[1]], [0, size[1]]]))
    lo, hi = np.floor(np.nanmin(corners, 0)) - 1, np.ceil(np.nanmax(corners, 0)) + 1
    def visible(pts):
        keep = np.isfinite(pts).all(1)
        keep[keep] &= (np.abs(pts[keep]) < 8000).all(1)
        return pts[keep].astype(np.int32)
    for xv in np.arange(lo[0], hi[0] + 1):
        pts = cam.project(np.column_stack([np.full(200, xv), np.linspace(lo[1], hi[1], 200), np.zeros(200)]))
        cv2.polylines(grid, [visible(pts)], False, (0, 220, 255) if xv % 5 == 0 else (0, 160, 200), 2 if xv % 5 == 0 else 1, cv2.LINE_AA)
    for yv in np.arange(lo[1], hi[1] + 1):
        pts = cam.project(np.column_stack([np.linspace(lo[0], hi[0], 200), np.full(200, yv), np.zeros(200)]))
        cv2.polylines(grid, [visible(pts)], False, (255, 160, 0) if yv % 5 == 0 else (200, 120, 0), 2 if yv % 5 == 0 else 1, cv2.LINE_AA)
    grid = draw_segments(grid, segs, sel)
    if lane:
        s_ = [int(round(v)) for v in lane["segment"]]
        cv2.line(grid, (s_[0], s_[1]), (s_[2], s_[3]), (0, 0, 0), 9, cv2.LINE_AA)
        cv2.line(grid, (s_[0], s_[1]), (s_[2], s_[3]), (0, 255, 255), 4, cv2.LINE_AA)
        txt = f"yellow lane: {lane['angleToA_deg']:.0f} deg to conveyor direction on the floor"
        cv2.putText(grid, txt, (s_[0], s_[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(grid, txt, (s_[0], s_[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    for t in tracks:  # one label per person track: median height at the median head position
        mine = [(o, zz) for o, zz in zip(obs, z) if o[4] == t]
        hx, hy = np.median([o[2] for o, _ in mine]), np.median([o[3] for o, _ in mine])
        txt = f"{np.median([zz for _, zz in mine]) * h:.2f} m"
        cv2.putText(grid, txt, (int(hx) - 30, int(hy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(grid, txt, (int(hx) - 30, int(hy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 255, 60), 2, cv2.LINE_AA)
    cv2.putText(grid, f"1 m floor grid (estimated scale: camera height {h:.2f} m +/- {100 * rel_sigma:.0f}%)", (24, size[1] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(grid, f"1 m floor grid (estimated scale: camera height {h:.2f} m +/- {100 * rel_sigma:.0f}%)", (24, size[1] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    cv2.imwrite(str(a.out / f"{a.clip_name}-floor-grid.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])

    calib = {
        "schema": "floorsight.floor-calibration.v1", "clip": a.clip_name, "opencv": cv2.__version__,
        "createdAt": datetime.now(timezone.utc).isoformat(), "imageSize": size,
        "method": "single-view metrology: vanishing points of selected segment families + standing-person height",
        "selection": sel, "vanishingPoints": {"A": vA.tolist(), "B": None if vB is None else vB.tolist(), "V": vV.tolist()},
        "laneCheck": lane, "figureHeadsMosaicked": int(heads),
        "segmentResidualDeg": residual, **finfo, "principalPointAssumed": [size[0] / 2, size[1] / 2],
        "vpOrthocentre": orthocentre([vA, vB, vV]),
        "K": cam.K.tolist(), "R_worldAxesInCamera": cam.R.tolist(), "cameraHeight_m": round(h, 4),
        "cameraHeightRelSigma": round(rel_sigma, 4), "H_imageToFloor": cam.H_img_to_floor().tolist(),
        "tiltBelowHorizonDeg": round(float(np.degrees(np.arcsin(-cam.R[:, 2] @ np.array([0, 0, 1.0])))), 2),
        "scaleReferences": {"people": {"cameraHeight_m": round(h_people, 3), "relSigma": round(rel_people, 4)},
                            "forkliftPlusLoadLongest": None if h_fork is None else {"assumed_m": a.forklift_long, "cameraHeight_m": round(h_fork, 3), "relSigma": round(rel_fork, 4)},
                            "combined": "inverse-variance mean in log space" if h_fork is not None else "people only"},
        "referenceDimension": {"what": "standing adult height (head top to feet), assumed population value",
                               "value_m": PERSON_HEIGHT_M, "sigma_m": PERSON_HEIGHT_SIGMA_M,
                               "observations": len(obs), "tracks": len(tracks),
                               "perTrackHeight_m": [round(float(x) * h, 3) for x in zt],
                               "relSpreadAcrossTracks": rel_spread, "heightTrend_mPer100pxRow": slope},
        "floorPlanUse": "Published plan (Onal & Dandil 2024, Fig. 3) gives lane layout and camera positions but no dimensions; not used for scale.",
        "limitations": ["Principal point assumed at the image centre; lens distortion ignored.",
                        "Scale rests on assumed sizes (adult height 1.70 +/- 0.08 m; forklift-plus-load longest ground line 3.5 +/- 0.4 m when used); no surveyed site distance exists.",
                        "Points above the floor (chassis, chair) project too far away unless their height is modelled."],
        "monteCarloSamples": samples,
    }
    (a.out / f"{a.clip_name}-calibration.json").write_text(json.dumps(calib, indent=1))
    print(json.dumps({k: v for k, v in calib.items() if k not in ("monteCarloSamples", "H_imageToFloor", "R_worldAxesInCamera", "K")}, indent=1))
    return 0


def main() -> int:
    if int(cv2.__version__.split(".")[0]) != 5:
        raise SystemExit(f"OpenCV 5 required, found {cv2.__version__}")
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("segments"); s.add_argument("--clip", type=Path, action="append", required=True)
    s.add_argument("--models", type=Path, required=True); s.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("solve"); v.add_argument("--segments", type=Path, required=True); v.add_argument("--clip-name", required=True)
    v.add_argument("--select", action="append", required=True, help="A=1,2,3  B=...  V=...")
    v.add_argument("--detections", type=Path, action="append", default=[]); v.add_argument("--median", type=Path, required=True)
    v.add_argument("--samples", type=int, default=200); v.add_argument("--out", type=Path, required=True)
    v.add_argument("--exclude", action="append", default=[], help="A=4,5 removes segments from a family")
    v.add_argument("--lane-region", default="", help="x0,y0,x1,y1 floor region holding the painted yellow lane")
    v.add_argument("--focal", type=float, default=None, help="fix focal length (px), e.g. from floor_scale_sweep.py")
    v.add_argument("--focal-sigma", type=float, default=0.0)
    v.add_argument("--forklift-long", type=float, nargs=2, default=None, help="near-field reference: longest forklift+load ground line, mean sigma (m)")
    a = p.parse_args()
    return cmd_segments(a) if a.cmd == "segments" else cmd_solve(a)


if __name__ == "__main__":
    raise SystemExit(main())
