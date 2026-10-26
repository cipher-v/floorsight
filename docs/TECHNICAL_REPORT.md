# Floorsight: technical report

**OpenCV AI Competition 2026 · Updated 27 October 2026**

Floorsight converts existing camera footage into floor distances and reviewable forklift warnings. The prototype has been evaluated on six recorded factory clips and a simulated warehouse. Real footage is processed offline. Only the simulated forklift responds to braking commands.

The measurements below are from the saved 23–24 October runs. Repository fixes made after those runs do not change their recorded results. [Detailed methods and run history](report/Floorsight-Technical-Report-v0.5-20261024.md).

## Problem, users and intended benefit

Safety managers and shift supervisors need to know whether a person is actually in a forklift's path. Camera perspective makes image distance misleading: objects at different depths can appear to touch.

Floorsight measures positions on the floor, predicts a close pass, and shows the evidence behind a warning. The intended benefit is earlier, more useful intervention with fewer perspective-driven false alarms. The current evidence establishes a working prototype; it does not establish a reduction in workplace injuries.

## How the system works

1. **Filter:** detect faces and heads and pixelate them before writing processed images. A filter exception produces a black, withheld frame.
2. **Locate:** find people, forklifts and loads. Map their positions onto the floor in metres, with an uncertainty estimate.
3. **Stabilise:** filter frame-to-frame movement so outline jitter does not look like a sudden turn.
4. **Predict:** estimate the forklift's path, stopping area, tail swing and load blind zone. Flag a predicted pass within 0.8 m over the next 3 seconds; the floor-stage warning must persist for three frames.
5. **Control in simulation:** issue slow, stop, horn or resume commands. For recorded footage, log the command without applying it.
6. **Review:** replay recent measurements, inspect a close-up, and request another detector measurement. Recommend confirm, reject or uncertain.
7. **Decide:** a reviewer approves or rejects the forklift incident. Only an approved forklift warning becomes an SNS alert. Save the evidence, recommendation and decision in the trace.

![System architecture](diagrams/floorsight-architecture-20261026.png)

### Two perception paths

The real-clip pipeline uses RT-DETRv2, Grounding DINO and SlimSAM with camera calibration from floor directions and size references. The simulator uses rendered cameras, ArUco markers, background subtraction and multi-camera tracking. They share application contracts and the control/review components; the simulator is not a validation of the real-clip detector stack.

### Substantive OpenCV 5 work

| Job | Implementation |
|---|---|
| Camera and floor geometry | Line-segment detection, vanishing-point geometry, `findHomography`, `perspectiveTransform`; ArUco and `solvePnP` in the simulator |
| Motion and tracking | MOG2, contours and floor tracking; Kalman-based motion smoothing |
| Risk areas | Polygon rasterisation, contours, point-to-polygon distances and sampled uncertainty |
| Privacy | YuNet through `FaceDetectorYN`, pixelation with `resize`, smoothing with `GaussianBlur` |
| Video and evidence | Video decode/encode, image transformations and image encoding |

The principal real-clip object detectors run in PyTorch. This project does not claim they run in OpenCV DNN. [Code overview](../README.md#how-it-is-built).

## Agentic vision

![Review workflow](diagrams/floorsight-agent-workflow-20261026.png)

The agent operates on measured evidence, not just a description of a fixed image. Its next tool call and recommendation depend on the prior result.

In the earlier AWS run, a second check identified a tracking glitch and led to rejection, preventing an alert. In the final run, three warnings passed the checks and reached the approval stage. The local policy made the recorded decisions; Bedrock integration exists but was not exercised in these runs.

[Trace walkthrough and raw traces](receipts/traces/README.md) · [Earlier run](receipts/app-run-02-summary-20261024.json) · [Final run](receipts/app-run-09-summary-20261024.json).

The recorded reviewer was a scripted stand-in following the recommendation. A real person can use the same approval controls, but no supervisor usability study has been completed. These demonstrations establish orchestration and traceability, not general agent accuracy.

## AWS and COOL

The saved runs used a COOL Marketplace AMI on an EC2 `c8g.xlarge` with four Graviton4 cores. The imported OpenCV version was `5.1.0-dev`, under `/opt/cool/python_3.12/site-packages/cv2`. The baseline was stock OpenCV 5.0.0 on the same instance. [Build evidence](receipts/cool-opencv-build-20261023.txt).

S3 stores evidence; DynamoDB stores track history; SNS delivers alerts; CloudWatch receives metrics and logs. The [final app receipt](receipts/app-run-09-summary-20261024.json) records all four adapters, nine mirrored evidence images, three SNS alerts and no reported cloud errors.

Private access uses SSM port forwarding. Terraform has an optional CloudFront endpoint, but a running public endpoint has not been verified in this review. [Deployment and verification](aws/DEPLOY.md).

### Performance comparison

Same instance, code and input frames; speed-up means stock latency divided by COOL latency, or COOL throughput divided by stock throughput.

| Workload | COOL / stock |
|---|---:|
| Adaptive threshold | 7.06× |
| Gaussian blur | 1.13× |
| Resize | 0.75× |
| OpenCV chain, one stream | 1.10× |
| OpenCV chain, four processes | 0.90× |
| Full real-clip perception, including detectors | 0.97× |

COOL accelerated some operations, but the detector-heavy end-to-end workload was about 3% slower. This result does not support an overall speed or cost-saving claim. [Benchmark evidence](receipts/cool-bench-20261024.json) · [Benchmark procedure](../scripts/benchmark/README.md).

A reproduced `fillPoly` crash in the COOL build led to polygon clipping before rasterisation. [Reproduction and workaround](findings/COOL-fillPoly-crash.md).

## Results and limits

| Question | Recorded result | Evidence |
|---|---|---|
| Does control change the simulated outcome? | Two in-path contacts without control; neither with control. Closest passes: 0.70 m and 0.65 m. | [Simulation receipt](receipts/interlock-sim-20261024.json) |
| What does control cost? | 27 m travelled in 40 seconds versus 80 m without control. | Same simulation receipt |
| Does it prevent every contact? | No. It cannot prevent a person walking into the back of a forklift moving away. Hidden people are another failure case. | Same simulation receipt; [detailed report](report/Floorsight-Technical-Report-v0.5-20261024.md) |
| Does the AWS review path work? | Three warnings, three scripted approvals, three SNS alerts, 29 trace steps. | [Final app receipt](receipts/app-run-09-summary-20261024.json) |
| Does smoothing help? | Heading swings fell from 53 to 11; turning-state flips from 94 to 2; one-frame warning blips from 8 to 0. | [Before/after](receipts/motion-noise-before-after-20261024.json) |
| Is the hardest clip resolved? | Final closest gap 1.55 m with no warning. Earlier detectors disagreed substantially; no ground truth establishes which outline was right. | [Measured twice](receipts/screen-002-measured-twice-20261024.json) |
| Is processing live? | No: about 4.1–4.6 seconds per frame on four Graviton cores. The app replays prepared results. | [Benchmark evidence](receipts/cool-bench-20261024.json) |
| Is privacy filtering complete? | 98.0% coverage against 2,636 reference head regions, with a documented missed head region. | [Privacy review](receipts/privacy-manual-review-20261024.md) |

These are development clips from one camera, not an independent test set. The head-coverage score compares detector outputs; it is not human-labelled privacy recall. The simulator's people follow scripts and do not react. The controller assumes a braking model; it has not been integrated with a real truck.

## Responsible operation

- Display uncertainty and missing-camera conditions instead of interpreting missing evidence as safety.
- Preserve the distinction between camera measurements, predictions and simulation outcomes.
- Keep raw footage private. Only filtered derivatives are published.
- Review forklift incidents before notifying the floor lead; retain the decision trace.
- Evidence is currently held for review. The nominal 24-hour retention is inactive while preservation is enabled; automatic deletion is not claimed.
- Use the prototype for research and demonstration. Real deployment needs site-specific measurement, live-rate processing, hardware integration and independent safety validation.

## Reproduction and next work

The [development guide](DEVELOPMENT.md) provides a self-contained simulator, application tests and a control on/off run. [AWS instructions](aws/DEPLOY.md) describe a fresh private deployment and how to verify COOL and cloud adapters.

Exact reconstruction of the historical real-clip run still needs the owner's input manifest, detector/model revisions and full perception package lock. The public setup script does not pin every dependency. This is an outstanding reproducibility gap.

The next evidence should be a held-out, manually labelled evaluation of distances and warnings, a measured latency budget, and testing with safety supervisors. A production braking claim requires separate hardware and field work.

## Credits

Factory footage: Önal & Dandıl (2024), [Video Dataset for Safe and Unsafe Behaviours](https://doi.org/10.17632/xjmtb22pff.1), CC BY 4.0. Published derivatives are trimmed and pixelated. No endorsement implied. The simulator is original project code. [Code licence](../LICENSE) · [Third-party licences](publish/THIRD_PARTY_LICENSES.md).
