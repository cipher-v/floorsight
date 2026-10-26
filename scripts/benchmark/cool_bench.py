"""Reproducible COOL benchmark: Floorsight's per-frame OpenCV workload on recorded footage.

Same script, same footage, same commit on every machine; the JSON proves which
OpenCV build ran (hash + key lines of cv2.getBuildInformation()).

    python scripts/benchmark/cool_bench.py --label laptop-x86 --out eval/results/bench_local_x86.json
    python scripts/benchmark/cool_bench.py --label c8g.xlarge-cool --price-usd-h 0.15952 --software-usd-h 0.01 \
        --out eval/results/bench_c8g_cool.json

Phases
  1. latency   one process, OpenCV's default thread pool (or --threads): every op timed
               separately over --frames frames after --warmup frames, then the whole
               chain timed again without per-op timers (end-to-end).
  2. throughput --workers processes (default: all logical CPUs), each with
               cv2.setNumThreads(1), each running the server chain (JPEG decode + the
               8 ops) on its own camera stream for --seconds. Aggregate frames/s / 10 =
               cameras one instance can serve at 10 fps.
  3. pipeline  if floorsight.vision.pipeline.RealVision imports, the full pipeline is
               timed on synchronised frame sets as well.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import platform
import re
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

PROC_W, PROC_H = 640, 360  # processing resolution for motion detection
FLOOR_W, FLOOR_H = 800, 500  # top-down floor raster (16 m x 10 m at 2 cm/px)
JPEG_QUALITY = 80  # same as floorsight.edge.uploader default
CORE_OPS = ["resize", "cvtColor", "GaussianBlur", "MOG2_apply", "morphologyEx", "findContours", "warpPerspective", "adaptiveThreshold"]
BUILD_KEYS = [
    "General configuration for OpenCV",
    "Version control",
    "Timestamp",
    "Host",
    "Target",
    "Baseline",
    "Dispatched code generation",
    "C++ Compiler",
    "C++ flags (Release)",
    "Parallel framework",
    "Custom HAL",
    "KleidiCV",
    "IPP",
    "Lapack",
    "Eigen",
    "ONNX",
    "Trace",
]

# --------------------------------------------------------------------------- #
# Workload
# --------------------------------------------------------------------------- #

_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def floor_homography(w: int, h: int) -> np.ndarray:
    """Fixed image->floor homography (trapezoid on the image to the floor raster). Cost-representative."""
    src = np.float32([[0.18 * w, 0.30 * h], [0.82 * w, 0.30 * h], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[0, 0], [FLOOR_W - 1, 0], [FLOOR_W - 1, FLOOR_H - 1], [0, FLOOR_H - 1]])
    return cv2.getPerspectiveTransform(src, dst)


def new_mog2():
    return cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)


def chain(frame, mog, H, timer=None):
    """Floorsight's per-frame image processing. `timer(name, t0)` records per-op time if given."""
    pc = time.perf_counter_ns
    t = pc()
    small = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
    if timer:
        t = timer("resize", t)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if timer:
        t = timer("cvtColor", t)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if timer:
        t = timer("GaussianBlur", t)
    fg = mog.apply(blur)
    if timer:
        t = timer("MOG2_apply", t)
    mask = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _K3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K5)
    if timer:
        t = timer("morphologyEx", t)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)  # drop MOG2 shadow label (127)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if timer:
        t = timer("findContours", t)
    top = cv2.warpPerspective(frame, H, (FLOOR_W, FLOOR_H), flags=cv2.INTER_LINEAR)
    if timer:
        t = timer("warpPerspective", t)
    at = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 7)
    if timer:
        timer("adaptiveThreshold", t)
    return len(contours), top, at


# --------------------------------------------------------------------------- #
# Footage
# --------------------------------------------------------------------------- #


def pick_footage(path: str | None) -> str:
    cands = [path] if path else [os.path.join(ROOT, "data", "sim", "demo_001"), os.path.join(ROOT, "data", "sim", "bench_dev")]
    for c in cands:
        if c and glob.glob(os.path.join(c, "cam*.mp4")) and (path or os.path.exists(os.path.join(c, "truth_events.json"))):
            return c
    raise SystemExit(
        "no complete footage found. Render some with:\n"
        "  python -m floorsight.sim.scenario --out data/sim/bench_dev --duration 20"
    )


def load_frames(footage: str, n: int) -> dict[str, list[np.ndarray]]:
    out = {}
    for p in sorted(glob.glob(os.path.join(footage, "cam*.mp4"))):
        cap = cv2.VideoCapture(p)
        frames = []
        while len(frames) < n:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if frames:
            out[os.path.splitext(os.path.basename(p))[0]] = frames
    if not out:
        raise SystemExit(f"could not decode any cam*.mp4 in {footage}")
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Environment / provenance
# --------------------------------------------------------------------------- #


def cpu_model() -> str:
    try:
        if sys.platform.startswith("linux"):
            txt = open("/proc/cpuinfo", encoding="utf-8", errors="ignore").read()
            m = re.search(r"^model name\s*:\s*(.+)$", txt, re.M)
            if m:
                return m.group(1).strip()
            try:
                ls = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5).stdout
                m = re.search(r"^Model name:\s*(.+)$", ls, re.M)
                v = re.search(r"^Vendor ID:\s*(.+)$", ls, re.M)
                if m:
                    return f"{v.group(1).strip() + ' ' if v else ''}{m.group(1).strip()}"
            except Exception:
                pass
            m = re.search(r"^CPU part\s*:\s*(\S+)", txt, re.M)
            if m:
                return f"arm CPU part {m.group(1)}"
        if sys.platform == "win32":
            import winreg

            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def build_info() -> dict:
    info = cv2.getBuildInformation()
    lines = []
    for line in info.splitlines():
        s = line.strip()
        if any(s.startswith(k) for k in BUILD_KEYS):
            lines.append(re.sub(r"\s{2,}", " ", s))
    return {"sha256": hashlib.sha256(info.encode()).hexdigest(), "key_lines": lines, "length": len(info)}


def opencv_dist() -> dict:
    from importlib import metadata

    out = {}
    for name in ("opencv-python-headless", "opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def hw_features() -> dict:
    names = ["CPU_NEON", "CPU_NEON_DOTPROD", "CPU_NEON_FP16", "CPU_NEON_BF16", "CPU_SVE", "CPU_SSE4_2", "CPU_AVX", "CPU_AVX2", "CPU_AVX_512F"]
    out = {}
    for n in names:
        if hasattr(cv2, n):
            try:
                out[n] = bool(cv2.checkHardwareSupport(getattr(cv2, n)))
            except Exception:
                pass
    return out


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"], capture_output=True, text=True, timeout=10).stdout.strip()
        return r.stdout.strip() + ("-dirty" if dirty else "") if r.returncode == 0 else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q)) if xs else None


def summarize(ns_list):
    ms = [x / 1e6 for x in ns_list]
    return {"p50_ms": round(pct(ms, 50), 4), "p95_ms": round(pct(ms, 95), 4), "mean_ms": round(float(np.mean(ms)), 4), "n": len(ms)}


def phase_latency(frames: dict[str, list[np.ndarray]], warmup: int, measured: int) -> dict:
    cams = list(frames)
    h, w = frames[cams[0]][0].shape[:2]
    H = floor_homography(w, h)
    jpegs = {c: [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1] for f in fs] for c, fs in frames.items()}
    per_op: dict[str, list[int]] = {k: [] for k in CORE_OPS}
    extra: dict[str, list[int]] = {"imdecode_jpeg": [], "aruco_detect": []}
    e2e_core: list[int] = []
    e2e_server: list[int] = []
    aruco = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

    def seq():
        # round-robin over cameras: frame i of every camera, like the live server
        n_total = warmup + measured
        i = 0
        while i < n_total:
            for c in cams:
                yield i, c, frames[c][i % len(frames[c])], jpegs[c][i % len(jpegs[c])]
            i += 1

    # Pass A: per-op timing (fresh background models)
    mogs = {c: new_mog2() for c in cams}
    for i, c, f, _ in seq():
        rec = i >= warmup

        def timer(name, t0, rec=rec):
            t1 = time.perf_counter_ns()
            if rec:
                per_op[name].append(t1 - t0)
            return time.perf_counter_ns()

        chain(f, mogs[c], H, timer)

    # Pass B: whole chain, no per-op timers
    mogs = {c: new_mog2() for c in cams}
    for i, c, f, _ in seq():
        t0 = time.perf_counter_ns()
        chain(f, mogs[c], H)
        if i >= warmup:
            e2e_core.append(time.perf_counter_ns() - t0)

    # Pass C: server chain = JPEG decode + whole chain; plus extra ops on their own
    mogs = {c: new_mog2() for c in cams}
    for i, c, _, j in seq():
        t0 = time.perf_counter_ns()
        f = cv2.imdecode(j, cv2.IMREAD_COLOR)
        t1 = time.perf_counter_ns()
        chain(f, mogs[c], H)
        t2 = time.perf_counter_ns()
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        t3 = time.perf_counter_ns()
        aruco.detectMarkers(g)
        t4 = time.perf_counter_ns()
        if i >= warmup:
            extra["imdecode_jpeg"].append(t1 - t0)
            e2e_server.append(t2 - t0)
            extra["aruco_detect"].append(t4 - t3)

    core = summarize(e2e_core)
    server = summarize(e2e_server)
    return {
        "frame_size": [w, h],
        "cameras": cams,
        "frames_per_camera_measured": measured,
        "per_op": {k: summarize(v) for k, v in per_op.items()},
        "extra_ops_not_in_e2e": {k: summarize(v) for k, v in extra.items()},
        "e2e_core": core,
        "e2e_server_with_jpeg_decode": server,
        "fps_single_stream_core": round(1000.0 / core["mean_ms"], 2),
        "fps_single_stream_server": round(1000.0 / server["mean_ms"], 2),
        "opencv_threads": cv2.getNumThreads(),
    }


def _worker(args):
    jpegs, warmup, start_at, seconds = args
    cv2.setNumThreads(1)
    frames_bgr0 = cv2.imdecode(jpegs[0], cv2.IMREAD_COLOR)
    h, w = frames_bgr0.shape[:2]
    H = floor_homography(w, h)
    mog = new_mog2()
    n = len(jpegs)
    for i in range(warmup):
        chain(cv2.imdecode(jpegs[i % n], cv2.IMREAD_COLOR), mog, H)
    late = time.time() > start_at
    while time.time() < start_at:
        time.sleep(0.001)
    end = start_at + seconds
    count, i = 0, warmup
    t0 = time.time()
    while time.time() < end:
        chain(cv2.imdecode(jpegs[i % n], cv2.IMREAD_COLOR), mog, H)
        count += 1
        i += 1
    return count, time.time() - t0, late


def phase_throughput(frames: dict[str, list[np.ndarray]], workers: int, seconds: float, warmup: int) -> dict:
    cams = list(frames)
    streams = [[cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1] for f in frames[c]] for c in cams]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        start_at = time.time() + 5.0 + 0.25 * workers  # time for spawn + warm-up
        jobs = [(streams[k % len(streams)], warmup, start_at, seconds) for k in range(workers)]
        try:
            res = pool.map_async(_worker, jobs, chunksize=1).get(timeout=seconds + 120 + 5 * workers)
        except mp.TimeoutError:
            pool.terminate()
            return {"workers": workers, "error": "throughput workers did not finish in time (machine overloaded?)"}
    counts = [r[0] for r in res]
    agg = sum(c / el for c, el, _ in res if el > 0)
    return {
        "workers": workers,
        "threads_per_worker": 1,
        "seconds": seconds,
        "frames_per_worker": counts,
        "workers_started_late": sum(1 for r in res if r[2]),
        "aggregate_fps_server_chain": round(agg, 2),
        "cameras_per_instance_at_10fps": int(agg // 10),
    }


def phase_pipeline(footage: str, frames: dict[str, list[np.ndarray]], warmup: int, measured: int) -> dict:
    try:
        mod = importlib.import_module("floorsight.vision.pipeline")
        RealVision = getattr(mod, "RealVision")
    except Exception as e:
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}
    from floorsight.contracts import FrameMeta
    from floorsight.interfaces import FramePacket
    from floorsight.site import load_site

    site = load_site(os.path.join(footage, "site.json"))
    vision = None
    errors = []
    for ctor in (lambda: RealVision(site), lambda: RealVision(site=site), lambda: RealVision(os.path.join(footage, "site.json"))):
        try:
            vision = ctor()
            break
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
    if vision is None:
        return {"available": False, "reason": "could not construct RealVision: " + " | ".join(errors)}
    cams = [c.camera_id for c in site.cameras if c.camera_id in frames]
    n = min(len(frames[c]) for c in cams)
    times = []
    try:
        for i in range(min(n, warmup + measured)):
            packets = {}
            for c in cams:
                f = frames[c][i]
                packets[c] = FramePacket(FrameMeta(camera_id=c, frame_index=i, t=i / 10.0, width=f.shape[1], height=f.shape[0], faces_blurred=0), f)
            t0 = time.perf_counter_ns()
            vision.process(packets)
            if i >= warmup:
                times.append(time.perf_counter_ns() - t0)
    except Exception as e:
        return {"available": True, "error": f"{type(e).__name__}: {e}", "sets_timed": len(times)}
    s = summarize(times)
    return {
        "available": True,
        "cameras_per_set": len(cams),
        "per_set": s,
        "sets_per_s": round(1000.0 / s["mean_ms"], 2) if times else None,
        "camera_frames_per_s": round(len(cams) * 1000.0 / s["mean_ms"], 2) if times else None,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--footage", help="directory with cam*.mp4 + site.json (default: data/sim/demo_001, else bench_dev)")
    ap.add_argument("--label", default=platform.node(), help="free text, e.g. c8g.xlarge-cool")
    ap.add_argument("--warmup", type=int, default=30, help="warm-up frames per camera (not measured)")
    ap.add_argument("--frames", type=int, default=150, help="measured frames per camera")
    ap.add_argument("--threads", type=int, help="cv2.setNumThreads for the latency phase (default: OpenCV default)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="processes in the throughput phase (0 = skip)")
    ap.add_argument("--seconds", type=float, default=15.0, help="throughput measurement window")
    ap.add_argument("--no-pipeline", action="store_true")
    ap.add_argument("--price-usd-h", type=float, help="EC2 on-demand $/h for the cost-per-camera figure")
    ap.add_argument("--software-usd-h", type=float, default=0.0, help="Marketplace software fee $/h (COOL)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.threads is not None:
        cv2.setNumThreads(a.threads)
    footage = pick_footage(a.footage)
    frames = load_frames(footage, a.warmup + a.frames)
    print(f"footage {footage}: {', '.join(f'{c}={len(v)}' for c, v in frames.items())} frames", flush=True)

    result = {
        "schema": "floorsight.cool_bench/1",
        "label": a.label,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "opencv": {
            "version": cv2.__version__,
            "module_path": os.path.dirname(cv2.__file__) if getattr(cv2, "__file__", None) else None,
            "pip_distributions": opencv_dist(),
            "build_information": build_info(),
            "hardware_support": hw_features(),
            "cpus_seen_by_opencv": cv2.getNumberOfCPUs(),
        },
        "machine": {
            "arch": platform.machine(),
            "cpu_model": cpu_model(),
            "logical_cores": os.cpu_count(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "run": {
            "git_commit": git_commit(),
            "script_sha256": sha256_file(os.path.abspath(__file__)),
            "footage_dir": os.path.relpath(footage, ROOT),
            "footage_sha256": {os.path.basename(p): sha256_file(p) for p in sorted(glob.glob(os.path.join(footage, "cam*.mp4")))},
            "warmup_frames_per_camera": a.warmup,
            "measured_frames_per_camera": a.frames,
            "argv": sys.argv[1:] if argv is None else argv,
        },
    }
    print("phase 1: latency", flush=True)
    result["latency"] = phase_latency(frames, a.warmup, a.frames)
    if a.workers:
        print(f"phase 2: throughput ({a.workers} workers x {a.seconds:g} s)", flush=True)
        result["throughput"] = phase_throughput(frames, a.workers, a.seconds, min(a.warmup, 20))
    if not a.no_pipeline:
        print("phase 3: full pipeline", flush=True)
        result["pipeline"] = phase_pipeline(footage, frames, min(a.warmup, 10), min(a.frames, 100))
    cams = result.get("throughput", {}).get("cameras_per_instance_at_10fps")
    if a.price_usd_h is not None and cams:
        hourly = a.price_usd_h + a.software_usd_h
        result["cost"] = {
            "ec2_usd_per_h": a.price_usd_h,
            "software_usd_per_h": a.software_usd_h,
            "usd_per_instance_month_730h": round(hourly * 730, 2),
            "usd_per_camera_month": round(hourly * 730 / cams, 4),
            "formula": "(ec2_usd_per_h + software_usd_per_h) * 730 / cameras_per_instance_at_10fps",
        }

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    lat = result["latency"]
    print(f"\nOpenCV {cv2.__version__}  build sha256 {result['opencv']['build_information']['sha256'][:16]}  {result['machine']['cpu_model']} ({result['machine']['arch']}, {os.cpu_count()} logical)")
    for k, v in lat["per_op"].items():
        print(f"  {k:<18} p50 {v['p50_ms']:8.3f} ms   p95 {v['p95_ms']:8.3f} ms")
    print(f"  {'e2e core':<18} p50 {lat['e2e_core']['p50_ms']:8.3f} ms   p95 {lat['e2e_core']['p95_ms']:8.3f} ms   -> {lat['fps_single_stream_core']} fps")
    print(f"  {'e2e +jpeg decode':<18} p50 {lat['e2e_server_with_jpeg_decode']['p50_ms']:8.3f} ms   -> {lat['fps_single_stream_server']} fps")
    if "throughput" in result:
        t = result["throughput"]
        print(f"  throughput {t['aggregate_fps_server_chain']} frames/s on {t['workers']} workers -> {t['cameras_per_instance_at_10fps']} cameras at 10 fps")
    if "pipeline" in result:
        print(f"  pipeline: {json.dumps(result['pipeline'])[:200]}")
    if "cost" in result:
        print(f"  cost: ${result['cost']['usd_per_camera_month']}/camera-month")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
