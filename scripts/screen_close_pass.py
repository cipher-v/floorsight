"""Find real clips where a person truly comes close to a MOVING forklift (floor metres). GCP ONLY.

Bounded intake from the same CC BY 4.0 dataset (Onal & Dandil 2024, doi:10.17632/xjmtb22pff.1):
publisher file listings -> at most --max-clips files and --max-bytes total, honest User-Agent, https
only, publisher SHA-256 must match, raw video stays in private quarantine. Each file gets a neutral name
(screen-NNN); the publisher folder (a behaviour label) never reaches perception.

Screening: only clips from the already-calibrated camera pose are measured (ORB + findHomography against
the calibration's empty-floor image; max corner shift < 4 px). Two frames per second: people (YOLO11m),
forklifts with masks (YOLOE open-vocabulary), floor points from the calibration (nominal camera), forklift
ground line = lowest mask pixels in the bottom band (as in the floor re-test). A forklift counts as moving
when its floor ground-line centre moved >= 0.3 m/s between samples. Ranked by the smallest floor gap between
a person's foot point and a moving forklift's visible ground line. This is a screen, not the decision: the
top clips then go through the full perception + danger-zone run. No accuracy claim comes from a screen.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import cv2
import numpy as np

USER_AGENT = "floorsight-research-intake/1.0 (private CC BY 4.0 dataset evaluation)"


class HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise ValueError("non_https_redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def download(url, target: Path, expected_sha, limit):
    digest, n = hashlib.sha256(), 0
    req = Request(url, headers={"Accept-Encoding": "identity", "User-Agent": USER_AGENT})
    with build_opener(HTTPSRedirects()).open(req, timeout=60) as r, target.open("xb") as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            n += len(block)
            if n > limit:
                raise ValueError("size_limit")
            digest.update(block); f.write(block)
        final = r.geturl()
    return {"bytes": n, "sha256": digest.hexdigest(), "match": digest.hexdigest() == expected_sha, "finalHost": urlsplit(final).netloc}


def main() -> int:
    if int(cv2.__version__.split(".")[0]) != 5:
        raise SystemExit(f"OpenCV 5 required, found {cv2.__version__}")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listing", type=Path, action="append", required=True)
    ap.add_argument("--exclude", action="append", default=[], help="publisher file ids already downloaded")
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--reference-image", type=Path, required=True, help="calibrated empty-floor image (filtered)")
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--quarantine", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-clips", type=int, default=40)
    ap.add_argument("--max-bytes", type=int, default=1_100_000_000)
    ap.add_argument("--sample-hz", type=float, default=2.0)
    a = ap.parse_args()
    fc, fr, rcp = load("floor_calibration"), load("floor_retest"), load("real_clip_perception")
    a.quarantine.mkdir(parents=True, exist_ok=False); a.out.mkdir(parents=True, exist_ok=False)
    cal = json.loads(a.calibration.read_text())
    cam = fc.Camera(cal["K"], cal["R_worldAxesInCamera"], cal["cameraHeight_m"])
    size = tuple(cal["imageSize"])
    # candidates: interleave listings, deterministic order by trailing number
    lists = []
    for lp in a.listing:
        items = [x for x in json.loads(lp.read_text()) if x["id"] not in set(a.exclude)]
        items.sort(key=lambda x: int("".join(ch for ch in Path(x["filename"]).stem.split("_")[-1] if ch.isdigit()) or 0))
        lists.append(items)
    cand, i = [], 0
    while any(i < len(l) for l in lists):
        cand += [l[i] for l in lists if i < len(l)]
        i += 1
    ref = cv2.cvtColor(cv2.imread(str(a.reference_image)), cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(4000)
    kr, dr = orb.detectAndCompute(ref, None)
    from ultralytics import YOLO, YOLOE
    person = YOLO(str(a.models / "yolo11m.pt"))
    yoloe = YOLOE(str(a.models / "yoloe-11m-seg.pt"))
    names = ["person"] + rcp.FORKLIFT_PROMPTS
    yoloe.set_classes(names, yoloe.get_text_pe(names))
    faces = cv2.FaceDetectorYN.create(str(a.models / "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.6)
    mapping, results, total = [], [], 0
    for k, item in enumerate(cand[: a.max_clips]):
        cd = item["content_details"]
        if total + cd["size"] > a.max_bytes:
            mapping.append({"neutral": None, "fileId": item["id"], "skipped": "byte budget"}); continue
        neutral = f"screen-{k + 1:03d}"
        target = a.quarantine / f"{neutral}.mp4"
        rec = {"neutral": neutral, "fileId": item["id"], "publisherFilename": item["filename"], "folderId": item["folder_id"],
               "publisherSha256": cd["sha256_hash"], "publisherBytes": cd["size"]}
        try:
            rec.update(download(cd["download_url"], target, cd["sha256_hash"], cd["size"] + 1024))
        except Exception as e:
            rec["downloadError"] = f"{type(e).__name__}: {e}"; mapping.append(rec); continue
        total += rec["bytes"]
        mapping.append(rec)
        if not rec["match"]:
            continue
        cap = cv2.VideoCapture(str(target))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, first = cap.read()
        res = {"neutral": neutral, "frames": n, "fps": fps, "seconds": round(n / fps, 2)}
        if not ok:
            res["error"] = "decode"; results.append(res); continue
        g = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        kc, dc = orb.detectAndCompute(g, None)
        m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(dr, dc) if dc is not None else []
        shift = None
        if len(m) >= 20:
            H, inl = cv2.findHomography(np.float32([kr[x.queryIdx].pt for x in m]), np.float32([kc[x.trainIdx].pt for x in m]), cv2.RANSAC, 3.0)
            if H is not None:
                h_, w_ = ref.shape
                cs = np.float32([[0, 0], [w_, 0], [w_, h_], [0, h_]]).reshape(-1, 1, 2)
                shift = float(np.linalg.norm(cv2.perspectiveTransform(cs, H) - cs, axis=2).max())
        res["viewShiftPx"] = None if shift is None else round(shift, 2)
        res["calibratedView"] = bool(shift is not None and shift < 4.0)
        if not res["calibratedView"]:
            results.append(res); continue
        step = max(1, int(round(fps / a.sample_hz)))
        prev_v, best, samples = {}, {"gap": float("inf")}, 0
        for f in range(0, n, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                break
            samples += 1
            pr = person.predict(frame, imgsz=960, conf=0.30, classes=[0], verbose=False)[0]
            pboxes = pr.boxes.xyxy.cpu().numpy()
            o = yoloe.predict(frame, imgsz=960, conf=rcp.FORKLIFT_CONF, verbose=False)[0]
            polys = o.masks.xy if o.masks is not None else []
            vehicles = [(b, p) for b, c, k_, p in zip(o.boxes.xyxy.cpu().numpy(), o.boxes.conf.cpu().numpy(), o.boxes.cls.cpu().numpy().astype(int), polys)
                        if names[k_] in rcp.FORKLIFT_PROMPTS and len(p) >= 3]
            if not vehicles:
                prev_v = {}; continue
            # one unit: union of vehicle masks (cab + load)
            px, _ = fr.near_side_pixels([p for _, p in vehicles], size)
            if len(px) < 6:
                continue
            gl = cam.to_plane(px); gl = gl[np.isfinite(gl).all(1)]
            if len(gl) < 6:
                continue
            cen = gl.mean(0)
            speed = float(np.linalg.norm(cen - prev_v["cen"]) / ((f - prev_v["f"]) / fps)) if prev_v else None
            prev_v = {"cen": cen, "f": f}
            vb = np.array([b for b, _ in vehicles])
            for pb in pboxes:
                if any(rcp.contained(pb, b) > 0.6 for b in vb):
                    continue  # the driver
                foot = cam.to_plane(np.array([(pb[0] + pb[2]) / 2, pb[3]]))[0]
                if not np.isfinite(foot).all():
                    continue
                gap = float(np.min(np.linalg.norm(gl - foot, axis=1)))
                if speed is not None and speed >= 0.3 and gap < best["gap"]:
                    best = {"gap": gap, "frame": f, "t": round(f / fps, 2), "speed": round(speed, 2), "personBox": pb.round(1).tolist()}
        res.update(samples=samples, closestMovingGap_m=None if not np.isfinite(best["gap"]) else round(best["gap"], 2),
                   closest=None if not np.isfinite(best["gap"]) else best)
        if np.isfinite(best["gap"]):  # privacy-filtered still of the closest moment for human review
            cap.set(cv2.CAP_PROP_POS_FRAMES, best["frame"]); ok, frame = cap.read()
            if ok:
                img = frame.copy()
                for pb in person.predict(frame, imgsz=1280, conf=0.10, classes=[0], verbose=False)[0].boxes.xyxy.cpu().numpy():
                    rcp.mosaic(img, rcp.head_region(pb))
                faces.setInputSize((frame.shape[1], frame.shape[0])); _, fs = faces.detect(frame)
                for fb in ([] if fs is None else fs):
                    rcp.mosaic(img, (fb[0] - 4, fb[1] - 4, fb[0] + fb[2] + 4, fb[1] + fb[3] + 4))
                for b, _ in vehicles:
                    rcp.soften(img, rcp.cab_region(b))
                del frame
                x1, y1, x2, y2 = (int(v) for v in best["personBox"])
                cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 235), 3)
                cv2.putText(img, f"{neutral} t={best['t']}s screen gap {best['gap']:.2f} m (visible ground line), forklift {best['speed']} m/s",
                            (24, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.imwrite(str(a.out / f"{neutral}-closest-filtered.jpg"), cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
        cap.release()
        results.append(res)
        print(json.dumps(res), flush=True)
    ranked = sorted([r for r in results if r.get("closestMovingGap_m") is not None], key=lambda r: r["closestMovingGap_m"])
    (a.out / "screening.json").write_text(json.dumps({"schema": "floorsight.close-pass-screen.v1", "opencv": cv2.__version__,
        "createdAt": datetime.now(timezone.utc).isoformat(), "bytesDownloaded": total, "clipsTried": len(mapping),
        "results": results, "ranked": [{"neutral": r["neutral"], "gap_m": r["closestMovingGap_m"], "t": r["closest"]["t"], "speed": r["closest"]["speed"]} for r in ranked],
        "limitations": ["Screen: nominal calibration only, visible ground line (not the full footprint), 2 samples per second.",
                        "Only clips from the calibrated camera pose are measured."]}, indent=1))
    # private mapping (publisher ids and folders) kept apart from perception inputs
    (a.quarantine / "PRIVATE-mapping.json").write_text(json.dumps(mapping, indent=1))
    print("RANKED", json.dumps((json.loads((a.out / "screening.json").read_text()))["ranked"][:10]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
