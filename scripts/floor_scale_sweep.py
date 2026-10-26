"""Focal-length sweep: which camera model makes the independent size references agree? GCP ONLY.

The first solve (development job floorcal-02) fixed focal length from two vanishing points and scale from
far-away standing people. It produced a 17 m camera height, a 0.77 m/100 px drift of people's heights
with image row, and a 7 m "visible forklift" - three signs that focal length and tilt are wrong (the
family-A lines sit in the lens-distorted bottom-left corner). Here focal length f is swept with the
same vanishing-point pixels (directions re-derived for each f). For every f the camera height is
estimated twice, independently:
  people   - far-field standing adults, assumed 1.70 +/- 0.08 m tall;
  forklift - near-field: the longest visible ground line of the moving forklift-plus-load unit
             (seen side-on) assumed 3.5 +/- 0.4 m (forklift ~2.3-2.6 m plus a carried stillage).
The right f should make the two heights agree, keep people's heights flat across image rows, and keep
the painted lane square to family A on the floor. All three are reported; nothing is chosen silently.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

FORK_LONG_M, FORK_LONG_SIGMA_M = 3.5, 0.4


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    if int(cv2.__version__.split(".")[0]) != 5:
        raise SystemExit(f"OpenCV 5 required, found {cv2.__version__}")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segments", type=Path, required=True)
    ap.add_argument("--clip-name", required=True)
    ap.add_argument("--median", type=Path, required=True)
    ap.add_argument("--detections", type=Path, action="append", required=True)
    ap.add_argument("--lane-region", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fc, fr = load("floor_calibration"), load("floor_retest")
    clip = json.loads(a.segments.read_text())["clips"][a.clip_name]
    segs, size = np.array(clip["segments"]), clip["size"]
    vA = fc.vanishing_point(segs[clip["autoVanishingPoints"][0]["inliers"]])
    vV = fc.vanishing_point(segs[clip["autoVanishingPoints"][1]["inliers"]])
    f_vp = fc.focal_from(vA, vV, np.array([size[0] / 2, size[1] / 2]))
    lane = fc.fit_yellow_lane(cv2.imread(str(a.median)), tuple(int(x) for x in a.lane_region.split(",")))
    obs = []
    for n, det in enumerate(a.detections):
        obs += [(*o[:4], f"{n}:{o[4]}", o[5]) for o in fc.standing_people(det, size)]
    # forklift near-side ground lines (moving, detected, with masks) from every clip
    ground = []
    for n, det in enumerate(a.detections):
        for line in det.read_text().splitlines():
            r = json.loads(line)
            for v in r.get("vehicles", []):
                if v.get("estimated") or not v.get("polygons") or not v.get("moving"):
                    continue
                pts, _ = fr.near_side_pixels(v["polygons"], size)
                if len(pts) >= 6:
                    ground.append((n, r["frame"], pts))
    rows = []
    for f in np.geomspace(800, 6000, 46):
        cam, _ = fc.Camera.from_vps(vA, None, vV, size, f=float(f))
        z = np.array([cam.height_of(o[:2], o[2:4]) for o in obs])
        ok = np.isfinite(z) & (z > 0.02)
        tracks = sorted(set(o[4] for o, k in zip(obs, ok) if k))
        zt = np.array([np.median([zz for zz, o, k in zip(z, obs, ok) if k and o[4] == t]) for t in tracks])
        h_people = 1.70 / float(np.median(zt)) if len(zt) else np.nan
        heights = z[ok] * h_people
        rowv = np.array([o[1] for o, k in zip(obs, ok) if k])
        trend = float(np.polyfit(rowv, heights, 1)[0] * 100) if len(rowv) > 5 else np.nan
        lengths = []
        for n, frame, pts in ground:
            p = cam.to_plane(pts)
            p = p[np.isfinite(p).all(1)]
            if len(p) >= 6:
                c = p - p.mean(0)
                ax = np.linalg.svd(c, full_matrices=False)[2][0]
                proj = c @ ax
                lengths.append(float(proj.max() - proj.min()))
        lengths = np.array(lengths)
        long_unit = float(np.percentile(lengths, 95)) if len(lengths) else np.nan
        h_fork = FORK_LONG_M / long_unit if long_unit > 0 else np.nan
        lane_angle = np.nan
        if lane:
            q = cam.to_plane(np.array(lane["segment"]).reshape(2, 2))
            d = q[1] - q[0]
            lane_angle = float(np.degrees(np.arctan2(d[1], d[0])) % 180)
        tilt = float(np.degrees(np.arcsin(-cam.R[2, 2])))
        rows.append({"f": round(float(f), 1), "tiltDeg": round(tilt, 2), "hPeople_m": round(h_people, 3), "hForklift_m": round(h_fork, 3),
                     "logRatio": round(float(np.log(h_fork / h_people)), 4) if np.isfinite(h_fork * h_people) else None,
                     "peopleHeightSpreadRel": round(float(np.std(zt) / np.mean(zt)), 4) if len(zt) > 1 else None,
                     "peopleHeightTrend_mPer100px": round(trend, 4), "laneAngleToA_deg": round(lane_angle, 2),
                     "forkliftShortestLine_m": round(float(np.percentile(lengths, 5)) * h_fork, 3) if len(lengths) else None})
    # Agreement: zero crossing of log(h_fork/h_people), interpolated in log f.
    lr = np.array([r["logRatio"] if r["logRatio"] is not None else np.nan for r in rows])
    fs = np.array([r["f"] for r in rows])
    cross = [float(np.exp(np.interp(0, [lr[i], lr[i + 1]], [np.log(fs[i]), np.log(fs[i + 1])]) if lr[i] < lr[i + 1] else
                       np.interp(0, [lr[i + 1], lr[i]], [np.log(fs[i + 1]), np.log(fs[i])])))
             for i in range(len(lr) - 1) if np.isfinite(lr[i]) and np.isfinite(lr[i + 1]) and np.sign(lr[i]) != np.sign(lr[i + 1])
             and all(rows[j]["peopleHeightSpreadRel"] is not None and rows[j]["peopleHeightSpreadRel"] < 0.3 for j in (i, i + 1))]
    # (a crossing where the people estimate has collapsed - spread >= 30% or undefined - is not agreement)
    # Uncertainty of the crossing: where |log ratio| <= combined relative sigma of the two references.
    sig = float(np.hypot(0.08 / 1.70, FORK_LONG_SIGMA_M / FORK_LONG_M))
    valid = np.array([r["peopleHeightSpreadRel"] is not None and r["peopleHeightSpreadRel"] < 0.3 for r in rows])
    band = fs[(np.abs(np.nan_to_num(lr, nan=9.0)) <= sig) & valid]
    out = {"schema": "floorsight.focal-sweep.v1", "opencv": cv2.__version__, "focalFromVanishingPoints": f_vp,
           "references": {"people_m": [1.70, 0.08], "forkliftPlusLoadLongest_m": [FORK_LONG_M, FORK_LONG_SIGMA_M]},
           "observations": {"people": len(obs), "forkliftGroundLines": len(ground)}, "lane": lane,
           "agreementFocalPx": cross, "agreementBandPx": [float(band.min()), float(band.max())] if len(band) else None,
           "combinedRelSigma": sig, "rows": rows}
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "focal-sweep.json").write_text(json.dumps(out, indent=1))
    # Chart (OpenCV drawing): camera height from people vs forklift, lane angle, people trend, against log f.
    W, H, L, B = 1400, 760, 90, 80
    img = np.full((H, W, 3), 250, np.uint8)
    lx = lambda f: int(L + (np.log(f) - np.log(fs[0])) / (np.log(fs[-1]) - np.log(fs[0])) * (W - L - 40))
    def plot(vals, lo, hi, col, top, height, label):
        cv2.rectangle(img, (L, top), (W - 40, top + height), (200, 200, 200), 1)
        cv2.putText(img, label, (L + 6, top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        pts = [(lx(f), int(top + height - (min(max(v, lo), hi) - lo) / (hi - lo) * height)) for f, v in zip(fs, vals) if np.isfinite(v)]
        cv2.polylines(img, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
        for val in (lo, hi):
            cv2.putText(img, f"{val:g}", (8, int(top + height - (val - lo) / (hi - lo) * height) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    hp = np.array([r["hPeople_m"] for r in rows], float); hf = np.array([r["hForklift_m"] for r in rows], float)
    plot(hp, 0, 25, (200, 90, 40), 40, 300, "camera height from far standing people (m, 1.70 m assumed) - blue")
    plot(hf, 0, 25, (0, 120, 230), 40, 300, "")
    cv2.putText(img, "camera height from near forklift+load (m, 3.5 m assumed) - orange", (L + 6, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 230), 1, cv2.LINE_AA)
    plot(np.array([r["laneAngleToA_deg"] for r in rows], float), 60, 120, (40, 160, 40), 370, 140, "painted lane angle to family A on the floor (deg; 90 expected)")
    plot(np.array([r["peopleHeightTrend_mPer100px"] for r in rows], float), -1, 1, (150, 40, 150), 540, 140, "people height drift with image row (m per 100 px; 0 expected)")
    for f in cross:
        cv2.line(img, (lx(f), 40), (lx(f), 680), (0, 0, 200), 1, cv2.LINE_AA)
        cv2.putText(img, f"agree f={f:.0f}px", (lx(f) + 4, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1, cv2.LINE_AA)
    cv2.line(img, (lx(f_vp), 40), (lx(f_vp), 680), (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(img, f"vanishing-point f={f_vp:.0f}", (lx(f_vp) + 4, 720), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    for f in (1000, 2000, 3000, 4000, 5000):
        cv2.putText(img, str(f), (lx(f) - 15, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(img, "focal length (px, log scale)", (W // 2 - 100, H - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.imwrite(str(a.out / "focal-sweep.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for r in rows[::3]:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
