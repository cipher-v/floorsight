"""Pack real-clip perception outputs into a Floorsight app replay folder. GCP ONLY.

Input: a perception v7d output folder (per clip: <clip>-detections.jsonl, <clip>-result.json, <clip>-filtered.mp4, and
calibration-used.json at the top). Output folder (read by floorsight.realclip):
    site.json        SitePlan of the real camera (floor_stage=True); floor coordinates shifted so the map starts at 0
    playlist.json    camera, calibration, offset, clips in play order
    <clip>-filtered.mp4 (copied)     privacy-filtered frames, no overlays
    <clip>.frames.jsonl              per frame: persons (floor xy, image box), vehicles (footprint, speed, heading, notes),
                                     zones (likely outlines), alerts (warning/danger with a 3-frame confirmation)
Alerts use the perception stage's own numbers: danger = moving forklift and (in stopping/tail-swing zone, in the blind
zone while load-first, or median clearance < 0.8 m); warning = moving forklift and P(within 0.8 m in 3 s) >= 0.5.
Each is "confirmed" after 3 frames in a row, as in the perception stage. Track ids get a clip code ("S002:P1").
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

MIN_RUN = 3
MAX_RANGE_M = 22.0  # ignore floor points farther than this from the camera when sizing the map (people far away still show)


def code_of(name: str) -> str:
    if name.startswith("screen-"):
        return "S" + name.split("-")[1]
    if name.startswith("clip-dev-"):
        return "D" + name.split("-")[2]
    return name.upper()[:6]


def vehicle_notes(v: dict) -> list[str]:
    out = []
    sp, ds = v.get("speed_mps") or [0, 0, 0], v.get("stoppingDistance_m") or [0, 0, 0]
    if sp[1] >= 0.3:
        acc = (v.get("accel_mps2") or [0, 0, 0])[1] if v.get("accelUsed", 0.0) >= 0.5 else 0.0
        out.append(f"{sp[1]:.1f} m/s" + (" accelerating" if acc >= 0.5 else " braking" if acc <= -0.5 else "") + f", stops in {ds[1]:.1f} m ({ds[0]:.1f}-{ds[2]:.1f})")
    else:
        out.append("not moving")
    if v.get("pLoadBlocksView", 0) >= 0.5:
        lf = v.get("pLoadFirst")
        out.append(f"load {v.get('loadHeight_m', 0):.1f} m: blocks view" + ("" if lf is None else (", load-first" if lf >= 0.5 else ", load trailing")))
    if v.get("cutByFrame"):
        out.append("cut by frame edge: outline uncertain")
    se, fit = v.get("turnRateSE_radps", 9.0), abs(v.get("turnRateFit_radps", 0.0))
    if v.get("pTurning", 0.0) >= 0.5 and sp[1] >= 0.3:
        out.append(f"turning {fit:.1f} rad/s (+/-{se:.1f})" if se <= 0.5 * fit else f"turn rate uncertain (+/-{se:.1f})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception", type=Path, required=True)
    ap.add_argument("--clips", nargs="+", required=True, help="play order")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gap-s", type=float, default=0.5)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cal = json.loads((a.perception / "calibration-used.json").read_text())
    K, R, h = cal["K"], cal["R_worldAxesInCamera"], float(cal["cameraHeight_m"])
    loaded = []
    pts = [np.zeros(2)]
    for name in a.clips:
        d = a.perception / name
        res = json.loads((d / f"{name}-result.json").read_text())
        rows = [json.loads(x) for x in (d / f"{name}-detections.jsonl").read_text().splitlines() if x.strip()]
        loaded.append((name, res, rows))
        for r in rows:
            fl = r.get("floor") or {}
            for p in fl.get("persons", {}).values():
                if p.get("xy_m"):
                    pts.append(np.asarray(p["xy_m"], float))
            for v in fl.get("vehicles", []):
                pts += [np.asarray(q, float) for q in v.get("footprint", [])]
    P = np.asarray([p for p in pts if np.isfinite(p).all() and np.hypot(*p) <= MAX_RANGE_M])
    lo, hi = np.floor(P.min(0)) - 1.0, np.ceil(P.max(0)) + 1.0
    off = lo  # site = floor - off
    width, depth = float(hi[0] - lo[0]), float(hi[1] - lo[1])

    def S(xy):
        return [round(float(xy[0] - off[0]), 3), round(float(xy[1] - off[1]), 3)]

    clips, offset_frames, offset_t = [], 0, 0.0
    totals = {"frames": 0, "alerts": 0, "confirmed": 0}
    for name, res, rows in loaded:
        code = code_of(name)
        fps = float(res["fps"])
        out_rows = []
        s_danger: dict[str, int] = {}
        s_warn: dict[str, int] = {}
        for r in rows:
            fl = r.get("floor") or {}
            boxes = {p["id"]: p for p in r.get("persons", [])}
            vboxes = {v["id"]: v for v in r.get("vehicles", [])}
            persons, alerts = [], []
            seen = set()
            for pid, p in fl.get("persons", {}).items():
                gid = f"{code}:{pid}"
                seen.add(pid)
                b = boxes.get(pid, {})
                if p.get("xy_m"):
                    persons.append({"id": gid, "xy": S(p["xy_m"]), "sigma": p.get("xySigma_m"), "seated": p.get("seated"), "box": b.get("box"), "foot": b.get("foot")})
                reasons = list(p.get("alert") or [])
                warn = bool(p.get("warn"))
                s_danger[pid] = s_danger.get(pid, 0) + 1 if reasons else 0
                s_warn[pid] = s_warn.get(pid, 0) + 1 if warn else 0
                if not (reasons or warn):
                    continue
                if reasons and s_danger[pid] >= MIN_RUN:
                    level, confirmed, streak = "danger", True, s_danger[pid]
                elif warn and s_warn[pid] >= MIN_RUN:
                    level, confirmed, streak = "warning", True, s_warn[pid]
                else:
                    level, confirmed, streak = ("danger" if reasons else "warning"), False, max(s_danger[pid], s_warn[pid])
                veh = p.get("vehicle")
                vrec = next((v for v in fl.get("vehicles", []) if v["id"] == veh), {})
                alerts.append({"person": gid, "vehicle": f"{code}:{veh}" if veh else None, "level": level, "confirmed": confirmed, "streak": streak,
                               "clearance": p.get("clearance_m"), "pass": p.get("pass_m"), "passT": p.get("passT_s"), "ttc": p.get("ttc_s"),
                               "pWarn": p.get("pWarn"), "pPath": max(p["pInZone"].get("stop", 0), p["pInZone"].get("tail", 0)) if p.get("pInZone") else 0.0,
                               "reasons": reasons, "moving": bool(p.get("vehicleMoving")), "cut": bool(vrec.get("cutByFrame"))})
                totals["alerts"] += 1
                totals["confirmed"] += int(confirmed)
            for pid in list(s_danger):
                if pid not in seen:
                    s_danger[pid] = s_warn[pid] = 0
            vehicles = []
            for v in fl.get("vehicles", []):
                vb = vboxes.get(v["id"], {})
                vehicles.append({"id": f"{code}:{v['id']}", "footprint": [S(q) for q in v.get("footprint", [])], "heading": v.get("heading_rad"),
                                 "speed": v.get("speed_mps"), "accel": (v.get("accel_mps2") or [0, 0, 0])[1], "turn": v.get("turnRateFit_radps"),
                                 "turnSE": v.get("turnRateSE_radps"), "cut": bool(v.get("cutByFrame")), "est": bool(vb.get("estimated")),
                                 "box": vb.get("box"), "notes": vehicle_notes(v), "restart": bool(v.get("directionRestart"))})
            zones = {k: [[S(q) for q in poly] for poly in polys] for k, polys in (fl.get("zones") or {}).items()}
            out_rows.append({"f": r["frame"], "t": r["t"], "privacy": r.get("privacy"), "persons": persons, "vehicles": vehicles, "zones": zones, "alerts": alerts})
        (a.out / f"{name}.frames.jsonl").write_text("\n".join(json.dumps(x, separators=(",", ":")) for x in out_rows) + "\n")
        shutil.copy2(a.perception / name / f"{name}-filtered.mp4", a.out / f"{name}-filtered.mp4")
        n = len(out_rows)
        clips.append({"name": name, "code": code, "fps": fps, "frames": n, "offset_frames": offset_frames, "offset_t": round(offset_t, 3),
                      "video": f"{name}-filtered.mp4", "records": f"{name}.frames.jsonl", "clipSha256": res.get("clipSha256")})
        offset_frames += n
        offset_t += n / fps + a.gap_s
        totals["frames"] += n
    cam_site = [round(float(-off[0]), 3), round(float(-off[1]), 3)]
    site = {"schema_version": "1.3", "site_id": "real-factory-cam-a", "name": "Factory floor · Önal & Dandıl (2024), CC BY 4.0",
            "width_m": width, "depth_m": depth, "zones": [], "obstacles": [], "markers": [],
            "cameras": [{"camera_id": "camA", "width": 1920, "height": 1080, "fps": clips[0]["fps"], "K": K, "dist": [0.0] * 5,
                         "nominal_position": [cam_site[0], cam_site[1], round(h, 3)]}],
            "floor_stage": True}
    (a.out / "site.json").write_text(json.dumps(site, indent=1))
    playlist = {"camera_id": "camA", "calibration": {"K": K, "R": R, "h": h}, "offset_m": [float(off[0]), float(off[1])], "frame_size": [1920, 1080],
                "clips": clips, "source": "Onal & Dandil 2024 factory CCTV dataset, CC BY 4.0; filtered by the Floorsight perception stage",
                "rules": {"minRunFrames": MIN_RUN}}
    (a.out / "playlist.json").write_text(json.dumps(playlist, indent=1))
    print("EXPORT", json.dumps({"out": str(a.out), "site_m": [width, depth], "offset_m": [float(off[0]), float(off[1])], "camera_site": cam_site, **totals,
                                "clips": [(c["name"], c["frames"], c["offset_t"]) for c in clips]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
