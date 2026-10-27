# Deploy a private Floorsight demo on AWS

The saved 23–24 October runs used COOL on EC2 Graviton4. [Build evidence](../receipts/cool-opencv-build-20261023.txt) · [App receipt](../receipts/app-run-09-summary-20261024.json).

This guide describes a fresh deployment from the public repository. The repository review validated Terraform locally; it did not inspect, start or change the existing AWS account. Existing instances may use older code bundles.

## What Terraform creates

| Component | Purpose |
|---|---|
| COOL AMI on `c8g.xlarge` | OpenCV, simulator, review workflow and dashboard |
| Private S3 buckets | Evidence and retained run artifacts |
| DynamoDB | Track history |
| SNS | Approved forklift alerts; an email subscription is optional |
| CloudWatch | Logs and metrics |
| Instance IAM role + SSM | Resource access and a private dashboard tunnel |
| Optional CloudFront + Elastic IP | Public HTTPS access when explicitly configured |

Defaults matter:

- `public_endpoint=false`: no inbound app port; the service listens on loopback.
- `agent_llm="local"`: the recorded review policy, without Bedrock calls.
- `preservation_hold=true`: cloud evidence does not expire, and the root volume is retained on termination. The application also preserves local evidence by default.
- `create_budget=false`: no budget is created unless enabled.
- The artifacts bucket has `prevent_destroy=true`. A blanket `terraform destroy` is not a supported shortcut for removing the stack.

## 1. Prepare

Use an AWS identity with permission to create the resources, Terraform 1.6 or later, AWS CLI and the Session Manager plugin. Choose a region offering the COOL AMI and the selected Graviton instance.

Subscribe to the [COOL Marketplace listing](https://aws.amazon.com/marketplace/pp/prodview-fdvbfiewzuehs) and copy the AMI ID for that region. Review the current Marketplace and AWS charges before launch. EC2, disk, public IPv4 and other services may cost money even when promotional credits are available.

Set a budget and alert recipient. A budget is a notification, not a spending stop. This guide does not estimate the current bill.

## 2. Configure a fresh instance

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
```

Fill in:

| Variable | Value |
|---|---|
| `cool_ami_id` | AMI ID from the subscribed listing in your region |
| `app_repo_url` | `https://github.com/tonystark101-debug/floorsight.git` |
| `app_git_ref` | Exact commit SHA to deploy |
| `app_start_command` | Supplied simulator command in the example |
| `alert_email` | Recipient, or empty to omit the subscription |
| `create_budget` | `true` if Terraform should create the budget |
| `budget_limit_usd` | Your chosen monthly limit |

The example starts a self-contained simulated warehouse. The simulator generates frames as it runs; no private clip package is needed. After 40 seconds of simulation time the recording ends, while the dashboard remains available. Restart the service for another run.

```bash
terraform init
terraform fmt -check
terraform validate
terraform plan -out floorsight.tfplan
```

Inspect the plan, including instance/disk sizes, region, retention and cost-bearing resources. Then apply the reviewed plan:

```bash
terraform apply floorsight.tfplan
terraform output ssm_session_command
terraform output port_forward_command
```

Bootstrap installs the app, binds it to `127.0.0.1:8000`, and enables the cloud adapters with `FLOORSIGHT_CLOUD=aws`. It attempts the COOL bindings first. If that import fails, it records `pip-fallback` and installs stock OpenCV so the demo can start; that run **does not count as COOL evidence**.

## 3. Verify the deployment

Use the SSM shell command from Terraform. On the instance:

```bash
sudo tail -n 60 /var/log/floorsight-bootstrap.log
sudo systemctl status floorsight --no-pager
cd /opt/floorsight/app
/opt/floorsight/py -c "import cv2; print(cv2.__version__, cv2.__file__)"
cat /opt/floorsight/OPENCV_SOURCE
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/api/state
```

Confirm:

- OpenCV is 5.x and its path is under `/opt/cool/`; `OPENCV_SOURCE` ends in `cool`.
- State reports `vision=real`, `agent=real`, no startup failure and moving simulation tracks.
- `cloud.adapters` contains evidence, metrics, notifier and tracks.
- Cloud errors remain empty. Verify S3, DynamoDB and CloudWatch outputs after activity.
- Test an approved alert only with a known demo recipient. Confirm the SNS subscription before expecting email.
- `privacy.preservation_hold` matches the intended retention policy.

A successful `/healthz` response alone is insufficient: inspect `mode`, `running`, component labels and errors. A completed recording can legitimately report `mode=ended`.

On your computer, run the port-forward command and open http://127.0.0.1:8000/. Keep the SSM session open for the demonstration.

## 4. Judge access

A screen-share of the privately forwarded application can satisfy the live demonstration route once it is actually arranged. The video gallery alone is not a working endpoint.

For a **new** public deployment, configure `public_endpoint=true` and a random `judge_token` of at least 20 characters before the first plan/apply. CloudFront adds a secret origin header; the app checks it, then checks the judge token. Share access details privately.

For an **existing** instance, changing these Terraform variables and restarting is **not enough**. The instance resource deliberately ignores `user_data` changes to avoid replacing the machine. Update the deployed code, `/etc/floorsight.env` (cloud switch, judge token, origin secret) and the service's bind address over SSM, then reload systemd and restart. Inspect the changes before applying them. Verify the CloudFront path works and direct app access is refused. Do not replace an evidence-bearing instance to refresh its configuration.

Tokens and the origin secret are held in Terraform state and instance configuration. Keep these files out of Git and public logs.

## 5. Optional Bedrock mode

The saved demonstrations used the local policy. To test Bedrock, obtain a model ID available to your account and region, configure its IAM access, and verify one bounded run before changing `agent_llm`. The Python dependency is already pinned in `requirements.txt`.

The model defaults in Terraform are project configuration, not evidence of account availability. Inspect invocation errors and usage; do not assume that a successful local demo proves Bedrock integration. Do not run repeated paid model calls without a measured budget.

## 6. COOL comparison

Follow the [benchmark guide](../../scripts/benchmark/README.md). Keep code, input hashes and hardware identical between stock and COOL runs. Record the imported OpenCV path and build hash. Stop other compute jobs during measurement.

The saved real-clip result is 1.10× for the single-stream OpenCV chain and 0.97× for full perception. The public simulator benchmark measures a different workload; label it separately.

## 7. Ending the demo

Stop the application instance when compute is no longer required. Retained disks, addresses, logs and buckets can still incur charges. Preserve submission evidence and review each resource before deletion. The artifacts bucket's deletion protection and preservation hold are intentional; do not disable them merely to make a destroy command succeed.
