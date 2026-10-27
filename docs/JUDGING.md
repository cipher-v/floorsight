# Judge guide and submission checklist

**Project:** Floorsight — forklift risk measured from existing CCTV.

Start with the [3:41 submission video](https://tonystark101-debug.github.io/floorsight/#demo), then the [technical report](TECHNICAL_REPORT.md). The demo gallery is a static video page, not the running application. [Submission links and short description](SUBMISSION.md).

## Competition paths

The project targets the overall awards, Agentic Vision and Best Use of COOL. These are separate judging paths; using COOL or an agent alone does not establish eligibility or a winning result. [Official rules](https://opencv26.devpost.com/rules).

| Overall criterion | Weight | Evidence to inspect |
|---|---:|---|
| Technical execution | 30% | Floor geometry, smoothing, tests and failure cases |
| Innovation | 20% | Measured floor risk plus evidence-driven review |
| Real-world impact | 20% | Simulated control comparison; clearly bounded real-footage results |
| User experience | 10% | Dashboard, recommendation, approval and replay |
| Documentation/presentation | 10% | Report, diagrams, demo and runnable instructions |
| Cloud/reproducibility/responsible operation | 10% | AWS receipts, private deployment, privacy limits |

| Special award | Weighted criteria | Floorsight evidence |
|---|---|---|
| COOL | Integration 30%; architecture 25%; measured value 20%; innovation 15%; reproducibility/demo 10% | [Build](receipts/cool-opencv-build-20261023.txt), [full-pipeline experiments and quality checks](EVALUATION.md#complete-perception-benchmark), [historical comparison](receipts/cool-bench-20261024.json), [crash workaround](findings/COOL-fillPoly-crash.md) |
| Agentic Vision | Integration 30%; orchestration 25%; effectiveness 20%; safeguards/control 15%; UX/docs/demo 10% | [Controlled baseline comparison](EVALUATION.md), [workflow](diagrams/floorsight-agent-workflow-20261026.png), [recorded traces](receipts/traces/README.md), approval controls and tests |

Source: [published judging rubrics](https://opencv26.devpost.com/rules), checked 27 October 2026.

## Required material

The [competition page](https://opencv26.devpost.com/) calls for substantive OpenCV 5 work and a meaningful AWS component, a report, accessible code, pinned dependencies and instructions, an architecture diagram, a working endpoint or arranged live demo, a video of at most five minutes showing the team/application/architecture/results, and evaluation evidence.

| Item | Location or status |
|---|---|
| Technical report | [Current report](TECHNICAL_REPORT.md) |
| Judge-accessible code | This public repository |
| Dependencies/build/tests | [Application pins](../requirements.txt), [run guide](DEVELOPMENT.md), recovered [perception packages](receipts/perception-historical-environment-20261023.txt) and [model revisions](receipts/perception-models-20261023.json) |
| Architecture | [System diagram](diagrams/floorsight-architecture-20261026.png) |
| Working app access | **Owner to confirm later:** arrange a live screen-share or supply a tested endpoint and access instructions |
| Video | [3:41 submission video](https://tonystark101-debug.github.io/floorsight/#demo), updated 27 October; introduces Team Floorsight. Owner must confirm registered member details; individual footage was not supplied. |
| Evaluation and failures | [Results](TECHNICAL_REPORT.md#results-and-limits), [receipts](receipts/) |
| COOL evidence | Version/path, Graviton configuration and stock comparison linked above |
| Agent evidence | Workflow, traces and evaluation linked above |

**Deadline discrepancy:** Devpost's header says October 26, 2026 at 11:45 p.m. PDT (October 27, 12:15 p.m. IST); the page body says 11:59 p.m. Pacific (12:29 p.m. IST). Use the earlier time when planning. A repository push is not a Devpost submission. Check that the entry is submitted and the selected award paths are saved.

## Best use of the judge's time

1. Show the camera-versus-floor comparison: the product solves a concrete measurement problem.
2. Show the same simulated scene with control off and on. State the throughput penalty.
3. Show a trace where additional vision evidence changes the recommendation.
4. Show the final AWS receipt and the actual imported COOL path.
5. State the boundary: offline real footage, simulated braking, local policy and scripted recorded approvals.

## Priorities before submission

- **Access:** complete the owner details deferred above and test all judge links from a signed-out browser.
- **Reproducibility:** the historical perception package freeze and model revisions are now published. Arrange access to private inputs and calibration if judges need the real-clip run; the simulator runs without them.
- **Evaluation:** add a held-out labelled clip set only if time allows a real evaluation. Do not relabel the six development clips as independent validation.
- **COOL positioning:** explain the detector bottleneck and show complete-pipeline timing with output checks. The historical chain improved 1.10× but end-to-end was 0.97×. New experiments do not establish a reliable overall COOL advantage; the faster decoding option failed quality checks and remains off.
- **Agent positioning:** show the new baseline comparison and uncertainty cases alongside the recorded rejection trace. The new accuracy checks are synthetic. Live Bedrock access was denied by the account; the recorded demo uses the local policy.
- **Submission:** select both special awards if pursuing them and supply their extra evidence. Eligibility remains the judges' decision.

## Review status

See [repository audit](AUDIT.md) for fixes, checks and unresolved work. Saved receipts and videos retain the results of their original runs.
