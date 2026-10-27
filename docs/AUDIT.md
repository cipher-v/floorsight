# Repository review — 27 October 2026

This review covers the public repository, demo gallery, current run instructions, saved evidence, and the competition's overview, rules, resources and updates. It is not a certification of the software or an independent validation of the recorded experiments.

## Fixed

| Issue | Change |
|---|---|
| Long, conflicting entry documents | Replaced the README and project overview; added a concise technical report, runnable setup guide and judge checklist. Historical reports remain clearly marked. |
| Claims exceeded the evidence | Distinguished offline real footage from simulated braking, local policy from Bedrock, scripted approvals from supervisor evaluation, and development comparisons from independent validation. |
| Broken gallery code link | Replaced the malformed link; checked local media, images, anchors and entry-document links. |
| Gallery loaded metadata for every video | Videos load on request. Earlier drafts are collapsed, with direct links preserved. Added demo captions, video labels, keyboard focus and phone-width layout. |
| Dashboard implied real braking and full processing delay | Show command and receiver status for the selected vehicle; label simulated application and the limited timing measurement explicitly. Simplified review instructions. |
| Fresh AWS deployments did not activate cloud adapters | Added `FLOORSIGHT_CLOUD=aws` to Terraform's application environment. |
| Bootstrap matched `simlive` as recorded `sim` | Match the complete source argument, avoiding unnecessary prerecorded-footage generation. |
| Deployment and retention instructions were inaccurate | Documented private access, adapter verification, actual preservation defaults, and the manual update required on existing instances. Corrected command-line help. |
| Shutdown could leave a producer blocked on a full queue | Use cancellable queue operations and wait for the producer before releasing its decoder. |
| Live simulation advanced before applying control commands | Process feedback-driven frames in sequence so the next scene reflects the current command. |
| A missing forklift track could release a speed limit | Retain the limit until the truck is seen again. For an applied simulated stop, retain its stopped pose and check the person's clearance and path before release. |
| Tests depended on private/generated local files | Create temporary fixtures; handle Windows symlink permissions explicitly. Added Linux and Windows CI. |

## Upstream review incorporated

The branch includes the concurrent `ca1861a` review from `main`. Its fixes cover floor-stage stop release, alert delivery retries, readiness checks, approval conflicts, token removal from URLs, retention controls, replay admission, lens distortion, marker dictionaries and camera shutdown. Their additional tests are retained. The overlapping approval implementation uses the upstream version, which returns the server's decision record to the dashboard.

The upstream [AWS test receipt](receipts/pytest-r56-20261027.txt) reports 275 passed and 2 skipped on COOL. Its [recorded-floor replay check](receipts/interlock-replay-check-20261027.txt) removes the old resume at 40.4 seconds in clip-dev-002 while preserving the slow at 37.6 seconds and stop at 38.0 seconds. The existing app video predates that fix.

## Validation

- **Local tests after incorporating upstream:** 283 passed, 5 skipped. Three skips require Windows symlink privileges; one is a pre-existing research regression and one needs an optional rendered camera fixture. One dependency deprecation warning remains.
- **GitHub CI:** [Linux/Python 3.12 and Windows/Python 3.11](https://github.com/tonystark101-debug/floorsight/actions/runs/40387775725) each passed 286 tests with 2 skips. Both also passed public-site and JavaScript checks.
- **Application:** 80 generated frames processed with real camera perception, local review policy and face filtering; zero drops and zero runtime errors. No AWS services were invoked.
- **Rendered control check:** all four scenarios started. With control on, both in-path contacts were avoided (0.70 m and 0.65 m closest gaps); the tail contact remained. The combined code travelled 45.0 m in 40 seconds locally. This is a separate run from the historical AWS result of 27 m, with different code and environment; it does not establish a throughput improvement.
- **Public site:** all 35 video references, local assets, anchors and current entry-document links pass the checker. Desktop and 360 px layouts were inspected; demo playback and archive deep links worked. The demo is 4:22.168, below five minutes.
- **Infrastructure:** Terraform formatting and validation pass using the existing provider lock. No plan or apply was run.
- **Code/dependencies:** Python compilation, JavaScript syntax and `pip check` pass.

[Local run receipt](receipts/repository-check-20261027.json). Historical AWS receipts and demo videos retain their original measurements; they have not been regenerated by this review. The single scripted simulation does not establish general safety or throughput improvement.

## Additional track work on 27 October

- Fixed unsafe rejection recommendations when a tracker jumps or a second
  measurement places a person closer to the truck. Conflicting evidence now
  stays uncertain, with human approval required. Failed or non-finite second
  detector results cannot be treated as evidence.
- Added reproducible production-agent comparisons: 8 false exit alert records
  fell to 0, retaining all 8 visible hazards; all 16 blind cases requested review.
  Hazardous floor rejection recommendations fell from 16/40 to 0/40. False floor
  candidates recommended for action fell from 40 to 24. These are controlled
  synthetic cases, not field accuracy or reviewer-workload measurements.
- Exercised live Bedrock access. The account rejected requests, including a
  payment-instrument error. Added a denial circuit and guarded local fallback;
  model requests to wait cannot bypass evidence gathering.
- Added full-process performance and output checks, bounded detector concurrency,
  exact median computation, privacy-pixel comparison and dependency checks.
  Slower thread/layout trials and a faster decoding trial that failed quality
  checks remain documented. No reliable overall COOL acceleration is established.
- Recovered and published historical package/model records. Current benchmarks
  record source/input hashes, all model stages, privacy counters and encoder exits.

[Evaluation, commands and receipts](EVALUATION.md). CI runs the complete agent
comparison and publishes its trace as an artifact for each operating system.
The [code verification run](https://github.com/tonystark101-debug/floorsight/actions/runs/40452100709)
passed 306 tests on Linux and Windows, with 2 existing skips and 1 dependency
warning on each. Agent evaluation, artifact upload, gallery and JavaScript checks
also passed.
The new cloud jobs used an isolated directory on the existing Graviton instance;
they did not deploy the application or send SNS alerts.

## What still needs work

| Gap | Next action |
|---|---|
| Judge access and team introduction | Owner deferred these details. Confirm a working endpoint or live screen-share and the team introduction in the video. |
| Exact real-footage reproduction | Historical packages and model revisions are now published. Private input and calibration access still needs arrangement; use the public simulator for a self-contained run. |
| Replay admission | The upstream checks are opt-in (`FLOORSIGHT_REPLAY_ADMISSION=enforce`). Existing recorded clips lack the required privacy-review manifests, and floor-record files are not hash-bound. Complete that evidence chain before treating replay admission as enforced. |
| Independent accuracy evidence | Evaluate held-out, manually labelled footage for distance error, missed incidents, false alarms and privacy misses. The six development clips cannot establish general accuracy. |
| Live operation | Reduce and measure the full 4.1–4.6 seconds/frame perception delay before claiming live safety intervention. |
| Real forklift control | Hardware feedback, braking validation and field testing are absent. Current control results apply to the simulator. Track identity and camera loss still require deployment-specific validation. |
| COOL value | Report the historical 1.10× chain and 0.97× end-to-end results alongside the new quality-gated experiments. No reliable overall acceleration or cost saving has been established. |
| Live model access | Account owner must resolve Bedrock access/payment. Current evidence supports the local agent; it does not demonstrate successful model-driven planning. |
| Submission status | A code push does not submit a Devpost entry. Confirm submission and award selections separately. |

The strongest presentation is the measured floor-distance problem, a visible control on/off comparison, and a review trace where extra vision evidence changes the decision. See the [judge guide](JUDGING.md) for the full rubric and evidence links.
