# Measured agent behaviour and pipeline performance

The 27 October checks use the production agent. They distinguish real alert
records from recommendations and uncertain cases. All new agent scenes are
synthetic; this is not a field accuracy or workplace safety claim.

## Agent comparison

Run from the repository root:

```bash
python scripts/benchmark/agent_eval.py --out data/agent-evaluation.json --check
```

The baseline alerts on every candidate entering review. Ground-truth labels
come from scene geometry. The agent receives camera observations, not labels.
OpenCV measures a rendered second-look image, and the production replay tool
checks the recent track history. Each case saves the evidence and tool trace.

| Controlled test | Baseline | Current agent |
|---|---:|---:|
| Visible exit: false alerts | 8 | 0 |
| Visible exit: real hazards alerted | 8 / 8 | 8 / 8 |
| Exit cases without a second camera | Immediate alert | All 16 pending human review |
| Floor warnings: false candidates retained as actionable | 40 | 24 |
| Floor warnings: real hazards retained as actionable | 40 / 40 | 40 / 40 |

The exit test uses actual in-memory agent alert records. It never contacts an
external notifier. The floor-warning test measures **recommendations**: confirm
and uncertain both remain actionable; only reject is excluded. All 80 floor
warnings still require a reviewer. Thus the 40% reduction is in false flags
recommended for action, not a measured reduction in SNS notifications or
reviewer workload.

The old policy recommended rejecting 16 of the 40 hazardous floor cases when
the tracker jumped or the second measurement was closer than the tracker said.
The current policy rejects none. Conflicting or missing evidence stays uncertain.
A rejection recommendation now needs both a tracking artefact and an independent
measurement showing a substantially larger gap. Human approval still controls
the final decision; this change never relaxes the braking controller.

[Before receipt](receipts/agent-eval-before-20261027.json) ·
[After receipt](receipts/agent-eval-after-20261027.json).

These are ten authored floor-warning families with eight geometric variations,
plus four exit families with eight variations. Variations are correlated, so do
not present them as 112 independent field trials. CI uses different random scene
offsets and gaps. Unseen real-footage labels and a supervisor study remain needed.

## Bedrock check

```bash
python scripts/benchmark/bedrock_smoke.py --cases 4 --out data/bedrock-check.json
```

This opt-in command makes live model calls with synthetic evidence and sends no
notifications. The 27 October account denied Opus 5.5 and Sonnet 5 access. A
separate Haiku 4.5 check reported `INVALID_PAYMENT_INSTRUMENT`. The code cannot
repair account billing or grant model access. The app now stops repeating a
401/403 request and continues with its guarded local policy.

[Live check](receipts/bedrock-live-20261027.json) ·
[Alternative-model check](receipts/bedrock-sonnet-live-20261027.json).

## Complete perception benchmark

`scripts/benchmark/perception_compare.py` runs each build/configuration in a
separate process. It includes decoding, all detectors, floor processing, privacy
filtering and video encoding. It reverses run order on alternating repetitions,
records model revisions and input/source hashes, and compares detection records
and privacy regions. It does not replay cached detections.

Example on the existing Graviton environment:

```bash
/opt/floorsight/perc scripts/benchmark/perception_compare.py \
  --variant baseline-cool,/opt/floorsight/perc,1,4,contiguous,seek,numpy \
  --variant optimized-cool,/opt/floorsight/perc,1,4,contiguous,seek,compact \
  --variant optimized-stock,/opt/floorsight/perc-stock,1,4,contiguous,seek,compact \
  --clip /path/to/private/clip.mp4 --models /opt/floorsight/models \
  --hf-index /path/to/hf-models.json --calibration /path/to/calibration.json \
  --frames 30 --repeats 2 --out /path/to/new-benchmark-folder
```

The fourth variant field controls CPU threads per operation; the fifth controls
tensor memory layout; the sixth selects calibration-frame sampling; the seventh
selects the temporal median implementation. The default compact median partitions
one temporary stack in place and avoids a float64 output image. It retains the same sample
pixels and has an exact equality check against NumPy's original result.
Model weights, precision, detection thresholds and frame sampling stay the same.
Independent detectors can also run with two workers;
every privacy result is required before publication, and failures drain all
workers before the next frame. A concurrent stage-time sum can exceed elapsed
time. Use process wall time for the end-to-end comparison.

Lower thread counts were slower in the first sweep. Channels-last layout and
two-worker inference produced only small timing changes in the next pilot, with
one 0.1-pixel change in a recorded head-region coordinate. These options remain
off by default; they are tuning controls, not a claimed performance win.

The sequential-decoding experiment (`--background-sampling auto`) reduced a
20-frame COOL process median from 100.28 s to 98.31 s, but failed sample-pixel
equality on two of six clips. It is **off by default**. Matching outputs on the
20-frame prefix did not justify accepting different calibration pixels. Its
receipt remains published so the failed quality gate is visible.

The accepted compact median matched the complete background image byte for byte
on all six development clips, twice each. Its stage latency fell by about 1.6%
(roughly 0.02 s per clip). This is a small initialization improvement, not evidence
of real-time perception. The rejected sample-last layout is also retained in the
receipts: it was slower despite producing equal pixels.

The final complete-pipeline check kept the same COOL environment and restored
seeking on both sides. All 80 processed frames retained identical detection and
privacy-region records; no frames were withheld and all encoders succeeded.

| Full process, 20 frames | Run 1 | Run 2 | Median |
|---|---:|---:|---:|
| Original median calculation | 99.23 s | 99.36 s | 99.29 s |
| Compact median calculation | 99.41 s | 99.51 s | 99.46 s |

The compact run was about 0.17% slower overall despite the small isolated-stage
gain. This does **not** demonstrate a pipeline speedup. It avoids extra calibration
allocations while preserving outputs. Substantial end-to-end acceleration remains
unresolved. [Complete comparison](receipts/perception-median-comparison-20261027.json).

The stock/COOL comparison returned identical detection records on the tested
20-frame prefix, but different head-region coordinates. Rasterizing the recorded
regions found 378 baseline pixels uncovered and 472 additional pixels covered per
20-frame run with stock. That does not establish which build has better privacy
recall. The strict equivalence check correctly failed; do not treat these as
interchangeable outputs or claim a reliable COOL end-to-end advantage from the
small, variable timing difference.

The package audit also found SciPy 1.18.1 in the COOL environment versus 1.15.3
in stock. That comparison cannot isolate the effect of OpenCV alone. The benchmark
now rejects non-OpenCV dependency differences as well as unequal outputs. The
compact-median comparison uses the same COOL environment on both sides.

The first baseline process took 100.12 s: 65.49 s in the three PyTorch detector
stages and 22.01 s in the OpenCV privacy-person and face detectors. Faster
calibration alone cannot close that latency gap. The timing fix also stops
counting privacy-person inference twice in the stage totals.

| Evidence | Receipt |
|---|---|
| Complete pipeline, including rejected sequential decoding | [Six runs](receipts/perception-comparison-20261027.json) |
| Sampling quality gate | [Six-clip comparison](receipts/background-sampling-benchmark-20261027.json), [differing samples](receipts/background-sampling-differences-20261027.json) |
| Exact median optimization | [Six-clip comparison](receipts/median-benchmark-20261027.json), [slower layout pilot](receipts/median-layout-pilot-20261027.json) |
| Actual privacy pixel coverage | [Coverage comparison](receipts/perception-privacy-pixels-20261027.json) |
| Other configurations, including slower runs | [Thread sweep](receipts/perception-thread-sweep-20261027.json), [model layout pilot](receipts/perception-layout-pilot-20261027.json) |
| Runtime environment | [COOL packages](receipts/perception-cool-environment-20261027.txt), [stock packages](receipts/perception-stock-environment-20261027.txt), [model revisions](receipts/perception-models-20261023.json) |

The pipeline checks use one 20-frame prefix on a four-core Graviton4 host and
two repetitions. They are a bounded engineering check, not a throughput study
across camera workloads. Raw footage remains private. Public receipts contain
hashes, timings, counters and model revisions, not raw frames.
