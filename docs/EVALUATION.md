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
  --variant baseline-cool,/opt/floorsight/perc,1,4,contiguous,seek \
  --variant optimized-cool,/opt/floorsight/perc,1,4,contiguous,auto \
  --variant optimized-stock,/opt/floorsight/perc-stock,1,4,contiguous,auto \
  --clip /path/to/private/clip.mp4 --models /opt/floorsight/models \
  --hf-index /path/to/hf-models.json --calibration /path/to/calibration.json \
  --frames 30 --repeats 2 --out /path/to/new-benchmark-folder
```

The fourth variant field controls CPU threads per operation; the fifth controls
tensor memory layout; the sixth selects calibration-frame sampling. Short clips
now collect the same 25 background samples in one sequential decode pass instead
of repeatedly seeking and decoding video sections. Long clips still seek.
Model weights, precision, detection thresholds and frame
sampling stay the same. Independent detectors can also run with two workers;
every privacy result is required before publication, and failures drain all
workers before the next frame. A concurrent stage-time sum can exceed elapsed
time. Use process wall time for the end-to-end comparison.

Lower thread counts were slower in the first sweep. Channels-last layout and
two-worker inference produced only small timing changes in the next pilot, with
one 0.1-pixel change in a recorded head-region coordinate. These options remain
off by default; they are tuning controls, not a claimed performance win.
