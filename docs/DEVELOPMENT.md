# Run and test Floorsight

## Local setup

Use Python 3.11 or 3.12. Create and activate `.venv`, then install the pinned application dependencies:

```bash
python -m pip install -r requirements.txt
python -c "import cv2; print(cv2.__version__, cv2.__file__)"
```

The version must start with `5.`. On the COOL AMI, use `/opt/floorsight/py` so the AMI's OpenCV build is loaded; installing the stock wheel over it invalidates a COOL comparison.

## Choose a demo

| Mode | Command | Input and behavior |
|---|---|---|
| Interactive simulation | `python -m floorsight.server --source simlive --vision real --agent real --interlock on` | Renders its own cameras. The simulated forklift obeys control commands. No preparatory render required. |
| Recorded simulator | Commands below | Camera files are generated once, then replayed. Commands are logged but cannot change those files. |
| Real footage replay | `python -m floorsight.server --source realclip --vision replay --agent real --sim-dir PATH_TO_EXPORTED_REPLAY --no-drop` | Requires a prepared replay folder, including floor records and privacy-filtered video. The gallery MP4s alone are not sufficient. |

Open http://127.0.0.1:8000/ after starting the server. Stop with Ctrl+C.

```bash
python -m floorsight.sim.scenario --out data/sim/demo_001 --duration 20
python -m floorsight.server --source sim --sim-dir data/sim/demo_001 --vision real --agent real --loop
```

`--source fake` and `--vision fake` use simulator truth for UI development. They do not demonstrate camera perception. `--agent demo` is a scripted stand-in. `--agent real` runs the review workflow with a local policy by default. Bedrock requires a separate, verified account setup; see the [AWS guide](aws/DEPLOY.md).

## Verify a clean checkout

```bash
python -m pytest -q tests
python scripts/check_public_site.py
python scripts/interlock_sim.py --out data/interlock-check --no-video
python scripts/benchmark/agent_eval.py --out data/agent-evaluation.json --check
```

The tests create synthetic fixtures in temporary folders. They do not require AWS credentials or `data/sim/demo_001`. Face-detector integration tests may fetch OpenCV sample images; if the network is unavailable, those tests skip. `FLOORSIGHT_TEST_ARTIFACT_ROOT` optionally keeps acquisition fixtures in a directory you choose. Explicitly skipped research regressions remain labelled as unresolved.

The simulation command evaluates the current code and writes `interlock-sim-results.json`. Historical numbers in `docs/receipts/` refer to their dated runs; a new run must be reported separately.

The agent evaluation renders its own controlled evidence and needs no cloud
account or large models. It includes blind-camera and conflicting-measurement
cases. The opt-in live Bedrock and full perception benchmark commands are in the
[evaluation guide](EVALUATION.md).

## Real-footage perception

The default requirements cover the app, simulator and tests. They do **not** include the large PyTorch/Transformers detector stack. The historical setup is in `infra/aws/jobs/setup_perception.sh`. The recovered [package freeze](receipts/perception-historical-environment-20261023.txt) and [model revisions](receipts/perception-models-20261023.json) identify the original environment. These are audit records, not a portable installer: COOL comes from its AMI, and private footage and calibration still require access. The public simulator remains the self-contained reproduction path.

The processing path is:

1. Acquire licensed footage into private storage and record its checksum.
2. Set up the detector environment and model weights; save package versions and model revisions.
3. Run `scripts/real_clip_perception.py` to produce filtered video and floor records.
4. Use `scripts/export_app_replay.py` to prepare the replay directory.
5. Run the real-footage replay command above.

Use each script's `--help` for its required paths. Historical cloud jobs use account-specific input manifests and directories; they are examples for the project owner, not commands that work unchanged on another account. The public simulator is the self-contained reproduction path.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `site.json` or simulator truth missing | Use the complete `simlive` command, or generate the recorded simulator first. |
| Simulator shows no tracks | Include `--vision real --agent real`; `fake` vision requires a truth dataset. |
| PowerShell blocks activation | Invoke `.venv\Scripts\python.exe` directly in place of `python`. No execution-policy change is needed. |
| Real clip shown as black | Inspect the privacy/filter error. Keep filtering enabled for real footage. |
| AWS stores and alerts are inactive | Set `FLOORSIGHT_CLOUD=aws` along with the resource variables. Check `/api/state` → `cloud`. |
| Recording ends before a review completes | Leave the server running; pending review work is drained after playback. |
| Approval says it is no longer pending | Refresh the dashboard. Another reviewer may already have decided, or the request expired. |

## Retention

Use `--preservation-hold off` only when automatic deletion is intended. `--retention-h` then controls the local retention window. Terraform also supplies `FLOORSIGHT_PRESERVATION_HOLD` from its `preservation_hold` setting.

Evidence preservation is enabled by default, including in the saved submission run. The configured 24-hour retention does not delete files while the hold is active. `/api/state` reports `privacy.preservation_hold` and `privacy.automatic_deletion`. Terraform also retains cloud evidence while `preservation_hold=true`. Treat a change to these settings as a deliberate retention decision.
