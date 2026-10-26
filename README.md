# Floorsight

**Use existing CCTV to measure forklift risk in metres and review warnings before alerting a supervisor.**

Floorsight maps people and forklifts onto the floor, predicts close passes, and gives a reviewer the measurements behind each warning. In the simulator, it also slows and stops the forklift.

[Watch the 4:22 demo](https://tonystark101-debug.github.io/floorsight/#demo) · [Technical report](docs/TECHNICAL_REPORT.md) · [Judge guide](docs/JUDGING.md) · [All videos](https://tonystark101-debug.github.io/floorsight/)

> **Prototype:** six recorded factory clips and a simulated warehouse. Real clips are processed offline; braking commands are applied only in simulation. No field safety validation has been completed.

## The problem and solution

A person who looks close to a forklift in a camera image may be metres away on the floor. Pixel distance alone can create misleading alarms.

Floorsight uses OpenCV 5 to recover floor geometry, smooth movement, and estimate distance with uncertainty. It flags a predicted pass within 0.8 m over the next 3 seconds. A review agent checks the recent track, requests a close-up and a second detector measurement, then recommends a decision. The reviewer approves or rejects the incident; approved forklift warnings go to the floor lead through Amazon SNS.

![Floor measurements compared with the camera image](docs/report/figures/pixels-vs-floor-clip-dev-004.jpg)

## What the evidence shows

These measurements come from the saved 23–24 October 2026 runs. New code checks are recorded in the [repository audit](docs/AUDIT.md).

| Test | Result | What it establishes |
|---|---|---|
| Simulated forklift, same 40-second scenario | 2 in-path contacts without control; 0 with control | The control loop works in this scenario, at a throughput cost: 27 m travelled versus 80 m. |
| Six recorded clips, final AWS app run | 3 warnings, 3 approvals, 3 SNS alerts | The review and alert path runs end to end. Approvals in this recording used a scripted reviewer. |
| Tracking on saved detections | Sudden heading swings: 53 → 11 | Smoothing reduces unstable risk zones. |
| COOL versus stock OpenCV, same Graviton4 instance | 1.10× for the single-stream OpenCV chain; 0.97× for the full perception stage | Faster individual operations did not make the detector-heavy pipeline faster overall. |
| Privacy comparison against an earlier detector | 98.0% of 2,636 reference head regions covered | A development comparison, not a guarantee of anonymity or independently labelled recall. |

[Results, evidence files and failure cases](docs/TECHNICAL_REPORT.md#results-and-limits)

## Run a local demo

Requires Python 3.11 or 3.12. Run from the repository root. This simulator needs no AWS account, downloaded footage or large detector models.

```bash
git clone https://github.com/tonystark101-debug/floorsight.git
cd floorsight
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Then:

```bash
python -m pip install -r requirements.txt
python -m floorsight.server --source simlive --vision real --agent real --interlock on
```

Open **http://127.0.0.1:8000/**. The generated warehouse runs for 40 seconds of simulation time; restart the command to replay it. On slower CPUs it may take longer than 40 seconds. `--agent real` selects the review workflow; its default planner is a local policy, not a hosted language model.

```bash
# Tests
python -m pytest -q tests

# Compare control on/off; writes JSON and videos to data/interlock-check
python scripts/interlock_sim.py --out data/interlock-check
```

[Setup and troubleshooting](docs/DEVELOPMENT.md) · [AWS deployment](docs/aws/DEPLOY.md)

## How it is built

- **OpenCV 5:** floor mapping, motion, tracking, risk-zone geometry, face filtering and video I/O.
- **Real-clip detectors:** RT-DETRv2, Grounding DINO and SlimSAM. These use a separate perception environment.
- **Application:** FastAPI, a browser dashboard, review tools, approvals and an event trace.
- **AWS:** COOL on EC2 Graviton4, S3 evidence, DynamoDB history, SNS alerts and CloudWatch monitoring. Private access is the default.

![Architecture](docs/diagrams/floorsight-architecture-20261026.png)

## Limits that matter

The real footage comes from one camera and is a development set. Floor scale uses size references, not a surveyed floor. Different detectors disagree on the hardest clip. Perception takes about 4.1–4.6 seconds per frame on four Graviton cores. Real forklift integration, live-rate perception and field safety validation remain future work. The simulated controller cannot prevent every kind of contact.

## Data and licence

Code: [Apache-2.0](LICENSE). Footage: Önal & Dandıl (2024), [Video Dataset for Safe and Unsafe Behaviours](https://doi.org/10.17632/xjmtb22pff.1), CC BY 4.0; trimmed and pixelated, with no endorsement implied. Raw clips are not redistributed. Simulated people are rendered. [Model and library licences](docs/publish/THIRD_PARTY_LICENSES.md).
