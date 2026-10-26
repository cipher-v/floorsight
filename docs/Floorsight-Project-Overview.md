# Floorsight: project overview

Floorsight helps safety managers review possible forklift hazards using cameras a site already has.

It converts camera observations into floor distances, predicts a close pass, and shows the evidence behind the warning. A review agent can ask for more visual evidence. The supervisor decides whether a forklift warning should become an alert.

## What works today

- Six factory clips processed on AWS with OpenCV 5 and the COOL build.
- A dashboard with floor positions, risk zones, review cards and event history.
- A local review policy that requests measurements and records its decisions.
- S3 evidence storage, DynamoDB history, SNS alerts and CloudWatch adapters, exercised in the saved AWS run.
- A simulated forklift that responds to slow, stop and resume commands.

## What the prototype does not establish

Recorded footage cannot respond to a brake command. Real-clip perception is offline, not real time. The factory clips are development examples, not a held-out safety evaluation. The recorded reviewer was scripted. Bedrock support exists, but the submitted runs used the local policy.

A site-wide 3D model, live production control and field safety validation are future work.

## Start here

| Goal | Read |
|---|---|
| Understand the result in four minutes | [Demo](https://tonystark101-debug.github.io/floorsight/#demo) |
| Run the application | [Development guide](DEVELOPMENT.md) |
| Review methods and limitations | [Technical report](TECHNICAL_REPORT.md) |
| Check the competition submission | [Judge guide](JUDGING.md) |
| Deploy a private AWS demo | [AWS guide](aws/DEPLOY.md) |
| Inspect detailed historical measurements | [24 October report](report/Floorsight-Technical-Report-v0.5-20261024.md) |

The earlier planning overview is available in Git history. This page describes the implemented prototype.
