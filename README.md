# Floorsight

**Floor-distance forklift safety from the cameras a site already has: it slows or stops the forklift before it
reaches a person, then an agent double-checks the warning and a person makes the final call.** Built for the OpenCV AI Competition 2026 on OpenCV 5 and AWS.

> Status (24 Oct 2026): runs end to end on six real factory clips, privately on AWS Graviton4 with the Cloud
> Optimized OpenCV (COOL) build. One camera, development clips, not field-validated. Every number below comes from a
> receipt in `docs/receipts/`.

Demo video: [`videos/floorsight-demo.mp4`](videos/floorsight-demo.mp4) (4 min 22 s, narrated, captions) · All our
videos: [`videos/`](videos/README.md) · Technical report: `docs/report/` · Architecture: `docs/diagrams/`

**Live demo for the judges:** the app runs on our private AWS deployment; we run it live for the judges by
screen-share on request (a temporary public HTTPS link is one Terraform variable away, `docs/aws/DEPLOY.md`).

## What it does

1. **Floor map in metres from one ordinary camera.** OpenCV's line-segment detector finds the floor's directions;
   vanishing points give the camera's pose; two independent size references (people, the forklift) fix the scale,
   with its error. Every person and forklift becomes a point on the floor.
2. **Forklift danger zones, each with its error.** Stopping distance from the measured speed, tail swing in turns,
   and the driver's blind zone behind a tall load, from 64 sampled variations per frame.
3. **A 3-second early warning.** If a moving forklift will pass within 0.8 m of a person in the next 3 s on its
   current course, the dashboard shows CHECKING, then WARNING once the condition holds for 3 frames.
4. **The machine acts first.** The moment a warning is confirmed, the forklift is slowed to walking pace, or stopped
   if the person is in its path, without waiting for anyone (`floorsight/interlock.py`); it resumes only when the
   person is seen clear. Proven in a closed loop in the simulated warehouse (`scripts/interlock_sim.py`); a real truck
   needs a receiver on its speed control; on recorded video the command is logged, not applied.
5. **An agent that double-checks every warning afterwards.** It replays the last 1.5 s of floor
   measurements, takes a close-up, and re-measures the gap with a second detector pass. It then recommends confirm
   or reject in one line with three numbers. **A person approves or rejects; nothing reaches the floor lead without
   that.** Every step is written to a trace as it happens.
6. **Privacy first, fail-closed.** Heads and faces are pixelated before any frame is stored or shown. If the filter
   fails, the frame is black (tested on every run). Raw clips never leave a private store.

## Honest results

- Simulated warehouse, closed loop (`docs/receipts/interlock-sim-20261024.json`): without the interlock the forklift
  hit both people who stepped into its path; with it, it stopped and passed them at 0.70 and 0.65 m. It cannot help a
  person who walks into the back of a forklift driving away, or anyone no camera sees (our first layout put a picker in
  an aisle only one camera covers, and the forklift itself blocked that view), and it cost two thirds of the
  forklift's distance in 40 s. The simulated people do not react.
- Pixel distance lies: two "near misses" in the footage were about 2 m apart on the floor. Both were withdrawn.
- The key clip, measured twice. With our earlier detectors (20 Oct) the forklift's outline swerved toward a seated
  operator in its last 40 ms: 0.67 m, and the warning came 0.04 s *after* the closest moment. With the publishable
  detectors on AWS and Kalman-filtered tracks (24 Oct) the same clip never comes closer than 1.55 m, and Floorsight
  does not warn. Our 23 Oct runs did warn here, about 2 s early, but those warnings came from a brief 0.2 rad/s "turn"
  toward the operator that the new turn-rate filter shows was noise: the forklift drove straight. With one camera we
  cannot say which outline is right: at the swerve instant the two disagree by about 1.2 m. Either way, no warning
  based on a forklift's course can foresee a real swerve. Chart: `docs/report/figures/screen-002-measured-twice.jpg`.
- In the final app run on AWS (24 Oct, with smoothing and the interlock): Floorsight slowed, stopped and then resumed
  the forklift on clip-dev-002 ("a stop now halts it 2.2 m short"; recorded video, so logged, not applied); 3 warnings
  on 2 clips; the agent recommended confirm for all 3, one of them a forklift pulling away from a stop in a new
  direction, which it now tells apart from a tracking glitch; the reviewer, a scripted stand-in that follows the agent,
  approved them; 3 alerts on SNS; 29 trace steps (`docs/receipts/app-run-09-summary-20261024.json`). The earlier run before smoothing had 4 warnings and the agent
  rejected one built on a tracking glitch (`docs/receipts/app-run-02-summary-20261024.json`). A confirmed warning is a
  prediction on the forklift's course: in these runs the driver turned or the clip ended before a close pass.
- Reviewing our own demo video frame by frame showed jumpy zones. Every floor track now runs through a Kalman filter,
  and a constant-turn-rate Kalman filter decides when a forklift is turning. On the same saved detections: outline
  jitter down 50-80 % per clip, sudden heading swings 53 → 11, "turning" flips 94 → 2, one-frame warning blips 8 → 0
  (`docs/receipts/motion-noise-before-after-20261024.json`; report section 4b).
- Replacing our earlier AGPL-3.0 detectors with Apache-2.0 ones: on 60 real frames RT-DETRv2 finds 97 % of the people
  and 98 % of the forklifts the old stack found (93 % / 89 % at the thresholds we run, before tiling and tracking), and
  the new privacy stage pixelates 98.0 % of the 2,636 heads the old stack saw; one real gap, the top of one head in two
  frames, no face visible (`docs/receipts/privacy-manual-review-20261024.md`). Floor results per clip, old vs new:
  `docs/receipts/floor-compare-aws-run3-vs-20oct.json`.
- COOL vs stock OpenCV on the same Graviton4 instance and frames: `adaptiveThreshold` 7.06x, `GaussianBlur` 1.13x,
  one stream of our OpenCV chain 1.10x; `resize` 0.75x, four processes 0.90x, the whole perception stage 0.97x
  (PyTorch detectors dominate it). Chart: `docs/report/figures/cool-vs-stock.jpg`.
- We found and reproduced a crash in COOL's `cv2.fillPoly` (vertices below the image's last row) and clip every
  polygon to the image first (`floorsight/cvsafe.py`). Details: `docs/findings/COOL-fillPoly-crash.md`.

## Videos

Everything is in [`videos/`](videos/README.md), with a line on what each one shows. The main ones:

- [`videos/floorsight-demo.mp4`](videos/floorsight-demo.mp4): the demo, 4 min 22 s.
- [`videos/simulator/interlock-off.mp4`](videos/simulator/interlock-off.mp4) and
  [`interlock-on.mp4`](videos/simulator/interlock-on.mp4): the simulated warehouse, 40 s each, with the forklift
  interlock off and on.
- [`videos/app/app-real-clips.webm`](videos/app/app-real-clips.webm): the app on AWS replaying the six real clips:
  automatic slow/stop, the agent's second look, the approval step.
- [`videos/real-clips/`](videos/real-clips/): each real clip with its floor measurements and danger zones drawn on.

## How it is built

| Part | What runs |
|---|---|
| Perception (`scripts/real_clip_perception.py`) | RT-DETRv2 people/vehicles, Grounding DINO forklift confirmation and carried loads, SlimSAM outlines, YuNet faces (`cv2.FaceDetectorYN`), OpenCV 5 for floor geometry, motion (MOG2), tracking, pixelation and video I/O |
| Danger zones (`scripts/danger_zone.py`) | 64-draw Monte Carlo on a 10 cm floor raster; `cv2.findContours`, `cv2.pointPolygonTest` |
| App (`floorsight/`, `web/`) | FastAPI server, replay source, rules, verify-then-act agent (local policy; Claude on Amazon Bedrock is built but was not open to our account, so the recorded runs used the local policy), dashboard |
| AWS (`infra/terraform/`) | EC2 Graviton4 `c8g.xlarge` from the COOL AMI; S3 evidence and artefacts; DynamoDB track history; SNS alerts; CloudWatch metrics and logs; IAM least privilege; private by default (SSM port-forward); `public_endpoint = true` adds a CloudFront HTTPS link |

## Run it

Private AWS deployment: `docs/aws/DEPLOY.md` (budget first, COOL subscription, `terraform apply`). Local
development: `python -m venv .venv && .venv/bin/pip install -r requirements.txt`, then
`python -m floorsight.server --source sim --sim-dir data/sim/demo_001 --vision real --agent real` after rendering the
simulated warehouse with `python -m floorsight.sim.scenario --out data/sim/demo_001`.

## Data, licences, credit

- Footage: six clips from Önal, O. & Dandıl, E. (2024), "Video Dataset for Safe and Unsafe Behaviours", Mendeley
  Data, V1, https://doi.org/10.17632/xjmtb22pff.1, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
  Changes: clips trimmed; heads and faces pixelated. No endorsement implied. The dataset paper states that every
  filmed employee signed an informed consent form. Raw clips are not redistributed; `videos/` holds only the
  trimmed, pixelated versions. Used for academic, non-commercial work.
- Floorsight code: Apache-2.0 (`LICENSE`).
- Models and libraries: `docs/publish/THIRD_PARTY_LICENSES.md` (all Apache-2.0, MIT or BSD; no AGPL dependency).

## Limits

One camera; scale from size references, not a surveyed floor plan; development clips, not a held-out test set; the
swerve case above; about 4.5 s of compute per frame on 4 Graviton cores, so the app replays processed clips; the
recorded decisions came from a scripted stand-in reviewer; the face filter can miss a face (the head pixelation and the fail-closed rule are the safety net,
not a guarantee).
