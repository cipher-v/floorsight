# COOL benchmark

For the complete real-clip pipeline (including all models), use
[`perception_compare.py`](perception_compare.py). For the agent baseline comparison,
use [`agent_eval.py`](agent_eval.py). Commands and interpretation are in
the [evaluation guide](../../docs/EVALUATION.md).

`cool_bench.py` times the OpenCV work Floorsight does on every camera frame, on the
same recorded footage, and writes one JSON file per machine. The JSON proves which
OpenCV build ran: `cv2.__version__`, the module path, and a SHA-256 of
`cv2.getBuildInformation()` plus its key lines (compiler, CPU baseline/dispatch,
parallel framework, HAL/KleidiCV). Two runs with the same hash ran the same build.

## What is measured

Per frame (960x540 BGR, 4 cameras, round-robin like the live server):

| Op | What Floorsight uses it for |
|---|---|
| `resize` (to 640x360, INTER_AREA) | processing resolution |
| `cvtColor` BGR->GRAY | |
| `GaussianBlur` 5x5 | denoise before background model |
| `MOG2_apply` (one model per camera) | moving-object mask |
| `morphologyEx` open 3x3 + close 5x5 | clean the mask |
| `findContours` (after a shadow threshold) | object blobs |
| `warpPerspective` to an 800x500 floor raster | top-down floor view |
| `adaptiveThreshold` 15 px | marker / edge preprocessing |

Reported:

- **per op** p50 / p95 / mean (ms), after warm-up;
- **end-to-end core** (the 8 ops, timed without per-op timers) and **end-to-end server**
  (JPEG decode + the 8 ops), with single-stream frames/s;
- extra ops not in the chain: JPEG decode, ArUco marker detection;
- **throughput:** one process per logical CPU, each `cv2.setNumThreads(1)`, each on its own
  camera stream for 15 s -> aggregate frames/s -> **cameras per instance at 10 fps**
  (= aggregate / 10);
- **full pipeline** (`floorsight.vision.pipeline.RealVision`) per synchronised 4-camera set, if importable;
- cost per camera-month if you pass `--price-usd-h` (and `--software-usd-h`).

Defaults: 30 warm-up + 150 measured frames per camera, footage `data/sim/demo_001`
(falls back to `data/sim/bench_dev`).

## Run locally

```bash
.venv/Scripts/python scripts/benchmark/cool_bench.py --label laptop-x86 --out eval/results/bench_local_x86.json
# quick check on 20 s of footage:
.venv/Scripts/python -m floorsight.sim.scenario --out data/sim/bench_dev --duration 20
.venv/Scripts/python scripts/benchmark/cool_bench.py --footage data/sim/bench_dev --frames 100 --out /tmp/b.json
```

## The 3-way protocol (for the "Best Use of COOL" claim)

Same **commit**, same **footage**, same **command**; only the machine / OpenCV build changes.

| Leg | Instance | OpenCV | How |
|---|---|---|---|
| A. x86 baseline | `c7i.xlarge` (4 vCPU) | stock `opencv-python-headless==5.0.0.93` wheel | `baseline_x86_count = 1` in Terraform |
| B. Arm, stock | `c8g.xlarge` (4 vCPU Graviton4) | same stock wheel | a second venv on the COOL instance (below), or `baseline_arm_stock_count = 1` |
| C. Arm, COOL | `c8g.xlarge` from the COOL AMI | COOL build (KleidiCV) | the app instance (`/opt/floorsight/py`) |

B vs C isolates the library (same hardware). A vs C is the whole "move to Graviton + COOL"
story. Optionally repeat B and C on `c8g.4xlarge` to show scaling.

Steps (via `aws ssm start-session --target <id>`, then `sudo -iu floorsight`):

```bash
cd /opt/floorsight/app
git rev-parse HEAD                         # must be identical on every leg
sha256sum data/sim/demo_001/cam*.mp4       # footage hashes are also stored in the JSON

# Footage: render once (on any machine) and copy it, or render on each machine.
# Rendering is deterministic code, but the mp4 encoder may differ per build: compare hashes,
# and if they differ, copy one set of files to all machines so every leg reads identical bytes.

# Leg C (COOL):
systemctl stop floorsight   # (as root) so the app does not compete for CPU
/opt/floorsight/py scripts/benchmark/cool_bench.py --label c8g.xlarge-cool \
    --price-usd-h 0.1595 --software-usd-h 0.01 --out eval/results/bench_c8g_xlarge_cool.json

# Leg B on the same instance (stock wheel, no COOL paths):
python3 -m venv /opt/floorsight/venv-stock
/opt/floorsight/venv-stock/bin/pip install -r requirements.txt
env -u PYTHONPATH -u LD_LIBRARY_PATH /opt/floorsight/venv-stock/bin/python scripts/benchmark/cool_bench.py \
    --label c8g.xlarge-stock --price-usd-h 0.1595 --out eval/results/bench_c8g_xlarge_stock.json

# Leg A on the c7i instance (installed by its user-data in /opt/floorsight/venv):
/opt/floorsight/py scripts/benchmark/cool_bench.py --label c7i.xlarge-stock \
    --price-usd-h 0.1785 --out eval/results/bench_c7i_xlarge_stock.json
```

Rules that keep it honest:

1. Run each leg 3 times; report the median run and keep all JSON files.
2. Nothing else running (stop the app service; no footage render in the background).
3. Check `opencv.build_information.sha256` differs between B and C, and `opencv.module_path`
   points into `/opt/cool/...` for C only. If C's path is the pip wheel, the COOL bindings were
   not used and the run does not count.
4. Prices in the commands are approximate (see `docs/aws/DEPLOY.md`); re-check before publishing.

## Cost per camera-month

```
cameras_per_instance = floor(aggregate_fps_server_chain / 10)
usd_per_camera_month = (ec2_usd_per_h + cool_software_usd_per_h) * 730 / cameras_per_instance
```

`cool_bench.py` writes this into the JSON under `cost` when `--price-usd-h` is given.
It counts the per-frame OpenCV work only; tracking, fusion, the agent and the web server
add load on top, so the full-pipeline numbers (`pipeline`) are the conservative ones to quote.

## Local sample (x86 laptop)

`eval/results/bench_local_x86.json`: Intel Core i5-13450HX (16 logical CPUs), Windows 11,
stock `opencv-python-headless 5.0.0.93`. It was recorded while other build jobs were running
on the same laptop, so p95 values are noisy. It is a sanity sample, not one of the three legs.
