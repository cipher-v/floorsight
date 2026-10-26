"""Closed-loop test of the forklift interlock in the simulated warehouse: interlock OFF vs ON, same four situations.

Every frame: the simulator renders its four ceiling cameras -> Floorsight's vision pipeline (the same RealVision the app
uses on this site: ArUco-calibrated cameras, background subtraction, multi-camera floor tracking) -> the interlock ->
the simulated forklift's brakes (ON only) -> the next frame. Ground truth (true positions) scores each situation.

    python scripts/interlock_sim.py --out OUT [--duration 40] [--modes off,on] [--no-video]

Writes OUT/interlock-sim-results.json, OUT/interlock-<mode>.mp4 (cam1, cam2, cam3 and a floor map with the forklift's
state), OUT/interlock-<mode>-frames.jsonl (per frame: truth, what Floorsight tracked, interlock state, commands).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floorsight.contracts import TrackClass, TrackState  # noqa: E402
from floorsight.interlock import ForkliftInterlock  # noqa: E402
from floorsight.sim.live import FPS, ClosedLoopSim  # noqa: E402
from floorsight.vision.pipeline import RealVision  # noqa: E402

MAP_W, MAP_H = 960, 540
STATE_COL = {"normal": (60, 170, 240), "slow": (0, 190, 255), "stop": (40, 40, 230)}   # BGR: orange, amber, red
CMD_TXT = {"slow": "AUTO-SLOW: forklift limited to walking pace", "stop": "AUTO-STOP: forklift braking",
           "resume": "RESUME: way clear, forklift drives on", "horn": "HORN: person walking into the forklift"}


def floor_map(site, state, world, lock_snap, cmds_recent, t, mode):
    img = np.full((MAP_H, MAP_W, 3), 32, np.uint8)
    sx = (MAP_W - 40) / site.width_m
    sy = (MAP_H - 120) / site.depth_m
    s = min(sx, sy)
    ox, oy = 20, MAP_H - 20

    def P(x, y):
        return int(ox + x * s), int(oy - y * s)

    cv2.rectangle(img, P(0, 0), P(site.width_m, site.depth_m), (90, 90, 90), 2)
    for z in site.zones:
        if z.kind.value == "lane":
            cv2.polylines(img, [np.array([P(*q) for q in z.polygon], np.int32)], True, (0, 140, 160), 1)
    for ob in site.obstacles:
        cv2.fillPoly(img, [np.array([P(*q) for q in ob.polygon], np.int32)], (70, 60, 50))
    f = state["forklift"]
    vstate = "normal"
    for st in lock_snap.get("vehicles", {}).values():
        if st.get("state") in ("slow", "stop"):
            vstate = st["state"] if vstate != "stop" else vstate
    c, sn = math.cos(f["heading"]), math.sin(f["heading"])
    box = [(f["x"] + dx * c - dy * sn, f["y"] + dx * sn + dy * c) for dx, dy in ((-1, -.55), (1, -.55), (1, .55), (-1, .55))]
    col = STATE_COL[f["mode"]] if mode == "on" else STATE_COL["normal"]
    cv2.fillPoly(img, [np.array([P(*q) for q in box], np.int32)], col)
    cv2.arrowedLine(img, P(f["x"], f["y"]), P(f["x"] + 1.5 * c, f["y"] + 1.5 * sn), (255, 255, 255), 2, tipLength=0.3)
    for pid, p in state["people"].items():
        cv2.circle(img, P(p["x"], p["y"]), 9, (230, 230, 230), -1)
        cv2.putText(img, p["scene"].split()[0], (P(p["x"], p["y"])[0] + 12, P(p["x"], p["y"])[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    for tr in world.tracks:  # what Floorsight tracked (cross), with a 3 s course for moving forklifts
        if tr.state not in (TrackState.seen, TrackState.estimated):
            continue
        q = P(tr.x, tr.y)
        cv2.drawMarker(img, q, (120, 255, 120) if tr.cls == TrackClass.person else (255, 200, 120), cv2.MARKER_CROSS, 14, 2)
        if tr.cls == TrackClass.forklift and math.hypot(tr.vx, tr.vy) > 0.3:
            cv2.line(img, q, P(tr.x + 3 * tr.vx, tr.y + 3 * tr.vy), (255, 200, 120), 1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (MAP_W, 92), (20, 20, 20), -1)
    head = f"INTERLOCK {'ON' if mode == 'on' else 'OFF'}   t = {t:5.1f} s   forklift {f['v']:.1f} m/s"
    cv2.putText(img, head, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    waiting = [st.get("waiting") for st in lock_snap.get("vehicles", {}).values() if st.get("state") == "stop" and st.get("waiting")]
    if mode == "on" and waiting:
        cv2.putText(img, "STOPPED: " + waiting[0], (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 60, 255), 2, cv2.LINE_AA)
    elif mode == "on" and cmds_recent:
        last = cmds_recent[-1]
        if t - last["t"] <= 2.5:
            colr = {"slow": (0, 190, 255), "stop": (60, 60, 255), "resume": (120, 220, 120), "horn": (0, 220, 255)}[last["command"]]
            cv2.putText(img, CMD_TXT[last["command"]], (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colr, 2, cv2.LINE_AA)
    elif mode == "off":
        cv2.putText(img, "Floorsight sees and warns, but nothing stops the truck", (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(img, "grey dot: true person   cross: what Floorsight tracked   box: true forklift (red = stopping)",
                (16, MAP_H - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
    return img


def run(mode: str, duration: float, out: Path, video: bool) -> dict:
    sim = ClosedLoopSim(render=True)
    vision = RealVision(sim.site, None)
    vision.set_background_plates(sim.plates())
    lock = ForkliftInterlock(enabled=(mode == "on"))
    dt = 1.0 / FPS
    writer = None
    if video:
        writer = cv2.VideoWriter(str(out / f"interlock-{mode}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1920, 1080))
    frames_f = (out / f"interlock-{mode}-frames.jsonl").open("w", encoding="utf-8")
    proc_ms, seen_people, seen_fork, stopped_s, slowed_s = [], {}, 0, 0.0, 0.0
    t_wall = time.time()
    for i in range(int(duration * FPS)):
        t = i * dt
        state = sim.advance(t, dt)
        images = sim.render(state)
        t0 = time.perf_counter()
        world = vision.process(sim.packets(images, t))
        cmds = lock.update(world)
        proc_ms.append((time.perf_counter() - t0) * 1000.0)
        for c in cmds:
            c["applied"] = sim.fork.command(c, t) if c["command"] != "horn" else None
            c["latency_note"] = "vision frame 0.1 s + actuation 0.2 s"
            if c.get("person_xy"):  # which scripted person the tracked person was (nearest true position)
                px, py = c["person_xy"]
                pid, p = min(state["people"].items(), key=lambda kv: math.hypot(kv[1]["x"] - px, kv[1]["y"] - py))
                c["personTruth"] = p["scene"] if math.hypot(p["x"] - px, p["y"] - py) < 1.0 else None
        f = state["forklift"]
        stopped_s += dt if f["v"] < 0.05 and mode == "on" and sim.fork.mode == "stop" else 0.0
        slowed_s += dt if sim.fork.mode == "slow" else 0.0
        persons = [tr for tr in world.tracks if tr.cls == TrackClass.person and tr.state == TrackState.seen]
        forks = [tr for tr in world.tracks if tr.cls == TrackClass.forklift and tr.state in (TrackState.seen, TrackState.estimated)]
        seen_fork += int(any(math.hypot(tr.x - f["x"], tr.y - f["y"]) < 1.5 for tr in forks))
        for pid, p in state["people"].items():
            if any(math.hypot(tr.x - p["x"], tr.y - p["y"]) < 0.8 for tr in persons):
                seen_people[pid] = seen_people.get(pid, 0) + 1
        snap = lock.snapshot()
        frames_f.write(json.dumps({"t": round(t, 2), "truth": state, "mode": sim.fork.mode,
                                   "tracks": [{"id": tr.track_id, "cls": tr.cls.value, "x": round(tr.x, 2), "y": round(tr.y, 2),
                                               "vx": round(tr.vx, 2), "vy": round(tr.vy, 2), "state": tr.state.value} for tr in world.tracks],
                                   "interlock": {k: v.get("state") for k, v in snap["vehicles"].items()}, "commands": cmds}) + "\n")
        if writer is not None:
            def lab(img, txt):
                img = img.copy()
                cv2.rectangle(img, (0, 0), (img.shape[1], 34), (20, 20, 20), -1)
                cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                return img
            top = np.hstack([lab(images["cam1"], "cam1 (simulated)"), lab(images["cam2"], "cam2 (simulated)")])
            bot = np.hstack([lab(images["cam3"], "cam3 (simulated)"), floor_map(sim.site, state, world, snap, list(lock.recent), t, mode)])
            writer.write(np.vstack([top, bot]))
        if i % 100 == 0:
            print(f"  {mode} frame {i} t={t:.1f} fork {f['x']:.1f},{f['y']:.1f} v={f['v']:.1f} mode={sim.fork.mode} "
                  f"tracks={len(world.tracks)} ({time.time() - t_wall:.0f} s)", flush=True)
    frames_f.close()
    if writer is not None:
        writer.release()
    n = int(duration * FPS)
    return {"mode": mode, "duration_s": duration, "frames": n, "scenes": sim.truth.summary(sim.walkers),
            "commands": lock.log, "counts": lock.counts, "forkliftModes": sim.fork.log,
            "forkliftDistance_m": round(sim.fork.s, 1), "stoppedSeconds": round(stopped_s, 1), "slowSeconds": round(slowed_s, 1),
            "perception": {"forkliftTrackedFrames": seen_fork, "personTrackedFrames": seen_people,
                           "visionPlusInterlockMs": {"median": round(float(np.median(proc_ms)), 1), "p95": round(float(np.percentile(proc_ms, 95)), 1)}}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--modes", default="off,on")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = {"schema": "floorsight.interlock-sim.v1", "opencv": cv2.__version__, "fps": FPS,
           "note": "Simulated warehouse, scripted people who do not react; the forklift obeys the interlock only in mode 'on'. "
                   "Truth = true positions; contact = true gap under 0.05 m.", "runs": {}}
    for m in a.modes.split(","):
        res["runs"][m] = run(m, a.duration, a.out, not a.no_video)
        print(json.dumps({m: res["runs"][m]["scenes"]}), flush=True)
    (a.out / "interlock-sim-results.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
