# Floorsight submission links and description

**Title:** Floorsight — forklift risk from existing CCTV

**One-line description:** Measure floor distances, investigate a warning with
vision tools, and give a supervisor the evidence before sending an alert.

| Material | Public link |
|---|---|
| Code and setup | [GitHub repository](https://github.com/tonystark101-debug/floorsight) |
| Running judge simulator | [Protected HTTPS demo](https://dz0itcb127dm1.cloudfront.net/) — token supplied privately in testing instructions |
| Submission video | [Watch the 3:41 video](https://tonystark101-debug.github.io/floorsight/#demo) |
| Direct MP4 | [Download or play](https://tonystark101-debug.github.io/floorsight/videos/floorsight-demo.mp4) |
| Technical report | [Problem, implementation, results and limits](TECHNICAL_REPORT.md) |
| Architecture | [System](diagrams/floorsight-architecture-20261026.png), [agent workflow](diagrams/floorsight-agent-workflow-20261026.png) |
| AWS deployment | [Guide](aws/DEPLOY.md), [infrastructure code](../infra/README.md) |
| Evaluation | [Methods, comparisons and raw evidence](EVALUATION.md) |

## Description to adapt for Devpost

Forklifts and workers share crowded factory floors. In a camera image, a person
can look close to a truck while being metres away on the floor. Floorsight turns
existing CCTV into estimated floor distances and predicted close passes, then
helps a safety supervisor review the evidence behind each warning.

OpenCV 5 performs floor geometry, motion and tracking, risk-zone processing,
privacy filtering and video I/O. Separate PyTorch models detect people, vehicles
and loads. A review agent checks recent movement, requests a close-up and a second
measurement, then adapts its recommendation. Conflicting or missing evidence
stays uncertain. Human approval controls floor-warning alerts.

The system ran on AWS Graviton4 using COOL. S3 stores evidence, DynamoDB stores
track history, CloudWatch captures activity, and SNS sends approved alerts. The
repository includes Terraform, bootstrap code and private-access instructions.
The public simulator needs no private footage or AWS account to run locally.

We pursue **Agentic Vision** and **Best Use of COOL**. In controlled synthetic
exit tests, false alert records fell from eight to zero while all eight visible
hazards remained detected; blind cases requested review. Floor-warning tests
retained all forty hazards and reduced false actionable recommendations by forty
percent. These are synthetic results, not a field-accuracy claim. The historical
OpenCV chain improved 1.10× with COOL, while full perception was 0.97×; recent
experiments also do not establish an overall pipeline speedup.

The prototype processes real footage offline and applies braking only in the
simulator. Recorded approvals were scripted. The demonstrated planner is local;
Bedrock account access remains blocked. Independent labelled footage, real
supervisor evaluation, live-rate perception and hardware integration remain
future work. The code passes 306 tests on Linux and Windows, with two existing
skips. We publish the failures and tradeoffs alongside the successful results.

## Before pressing Submit

- Select both special awards if pursuing them, and attach their evidence above.
- Put the judge access token in Devpost's private testing instructions. The
  [running simulator](https://dz0itcb127dm1.cloudfront.net/) was verified on 27 October;
  the video gallery remains a separate recording page.
- Confirm the registered team details. The owner accepted **Team Floorsight**;
  individual names or team footage were not provided for this edit.
- Save the entry and complete Devpost's submission action. Publishing GitHub
  code and a video does not submit an entry automatically.

The [official requirements](https://opencv26.devpost.com/) specify a video of no
more than five minutes, showing the team, application, architecture and principal
results, plus a working endpoint or arranged live demonstration.
