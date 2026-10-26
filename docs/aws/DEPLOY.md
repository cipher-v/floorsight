# Deploying Floorsight on AWS: human checklist

**Status (27 Oct 2026): applied once, in private mode.** On 23 Oct 2026 the Terraform in `infra/terraform`
was applied in us-east-1 (Terraform 1.16.6) with `public_endpoint = false`: one apply of 23 resources, a targeted
apply of the artifacts bucket and, per the Terraform state, a later in-place change that added Amazon Polly to the
instance role. The apply logs and the state are kept privately (they contain the account id); they are not in the
public repository. What exists:

- one EC2 Graviton4 `c8g.xlarge` from the COOL Marketplace AMI, with the S3 buckets, DynamoDB table, SNS topic,
  IAM roles, security groups, CloudWatch log group and dashboard;
- **no** CloudFront distribution and **no** Elastic IP: the public switch has never been turned on.

The app was **not** started by Terraform. `app_start_command` was empty at that apply, so the systemd service was
installed but never ran. Every app run (`docs/receipts/app-run-*`) was started over SSM Run Command with the scripts
in `infra/aws/` (`ssm_run.py`, `jobs/deploy_bundle.sh`, `jobs/run_app.sh`), listening on 127.0.0.1 on the instance.
Since then the Terraform files changed only on 27 Oct: `app_start_command` is required, public mode needs a judge
token of at least 20 characters, the app environment carries `FLOORSIGHT_PRESERVATION_HOLD`, and `--source simlive`
no longer triggers the boot-time footage render. Those changes pass `terraform fmt -check` and `terraform validate`;
they have not been applied.

The checklist below is for a fresh deployment. It needs a person with an AWS account, and it costs money from
step 4 on.

Time needed: about 2 hours the first time (most of it waiting for Bedrock access,
the Marketplace subscription and the first boot).

---

## What gets created

| Piece | Why |
|---|---|
| EC2 Graviton4 instance (`c8g.xlarge`) from the **COOL** Marketplace AMI | Runs the vision pipeline, agent and dashboard on Arm with the Cloud-Optimized OpenCV build |
| Elastic IP + CloudFront distribution (only with `public_endpoint = true`) | HTTPS judge link (`https://xxxx.cloudfront.net/`), stable across instance stop/start |
| Security group | No inbound rule while private. With `public_endpoint = true`: port 8000 open **only** to CloudFront's origin-facing IP list. No SSH; you get a shell with SSM Session Manager |
| S3 bucket | Blurred evidence images. Private, SSE-S3 encrypted. Expires after `evidence_retention_days` (1 day) **only if `preservation_hold = false`** |
| DynamoDB table (on-demand) | Track history snapshots; each item deletes itself after `track_ttl_seconds` (24 h) **only if `preservation_hold = false`** |
| SNS topic (+ email subscription) | Alerts to a person |
| IAM role | Only: put/get on that bucket, put/query on that table, publish to that topic, invoke the configured Claude model (+ its fallback), write Floorsight metrics and logs, SSM |
| CloudWatch log group + dashboard | fps, latency, events, CPU, app log. Logs expire after `log_retention_days` (7) **only if `preservation_hold = false`** |
| AWS Budgets budget | Emails at 50 % and 80 % of the monthly limit (actual) and 100 % (forecast) |
| Optional: `c7i.xlarge` x86 and plain Ubuntu arm64 instances | Only for the 3-way benchmark; `count = 0` by default |

**Retention is conditional.** With the default `preservation_hold = true` (owner hold) nothing expires: no S3
expiry, no DynamoDB TTL, no log expiry, and the app itself keeps its history and evidence files (Terraform sets
`FLOORSIGHT_PRESERVATION_HOLD=on`; the server flag is `--preservation-hold on|off`, default on). Only with
`preservation_hold = false` do the retention variables above apply, and the app deletes its local history and
evidence after `--retention-h` hours (24).

---

## 1. Account safety first (15 min, free)

1. Sign in as an IAM user or SSO role with admin rights, not the root user. Turn on MFA for root.
2. **Billing -> Budgets**: Terraform creates a budget, but create one by hand now too, so
   you are covered before anything exists: monthly cost budget, $50, email alerts at 50 / 80 / 100 %.
   A budget only **emails**; it never stops resources. Tear-down (step 9) is what stops spend.
3. Billing -> Billing preferences: turn on "Receive Free Tier usage alerts".
4. Choose the region: **us-east-1** (all three needed services are expected there; confirm
   in steps 2 and 3).

## 2. Bedrock model access (10 min + approval wait, free until called)

1. Bedrock console -> **Model access** in us-east-1 -> request access to **Claude Opus 5.5**
   (and **Claude Opus 5**, the agent's refusal-fallback model). Anthropic models ask for a
   short use-case form.
2. Copy the exact **model id** the console shows. The code and Terraform assume
   `anthropic.claude-opus-5-5` (fallback `anthropic.claude-opus-5`). If the console only offers an
   inference profile (e.g. `us.anthropic.claude-opus-5-5`), put that in `bedrock_model_id`;
   the IAM policy handles both forms.
3. The agent uses the Bedrock **Messages-API ("Mantle") endpoint**
   (`AnthropicBedrockMantle`, `https://bedrock-mantle.<region>.api.aws`). **Not verified:** whether that
   endpoint authorises with `bedrock:InvokeModel*` (what the role grants) or needs another
   IAM action. Check the Bedrock IAM documentation and, if needed, add the action names to
   `bedrock_extra_actions` in `terraform.tfvars`.
4. Bedrock mode uses the `anthropic` Python package: `requirements.txt` pins `anthropic[bedrock]` and the
   bootstrap installs it. It is imported only when `FLOORSIGHT_LLM=bedrock` (`agent_llm = bedrock`).

## 3. Subscribe to COOL on AWS Marketplace (10 min)

Listing: **"Cloud Optimized OpenCV For AWS Graviton4"** by OpenCV, AMI, Ubuntu 24.04,
built on OpenCV 5.0, KleidiCV inside
(<https://aws.amazon.com/marketplace/pp/prodview-fdvbfiewzuehs>). There is also a Docker
listing ($0.03 per container-hour); Floorsight uses the **AMI** one.

- **Price:** software fee per instance-hour on top of EC2: `c8g.xlarge` $0.01/h, `c8g.4xlarge` $0.04/h
  (also c8g.large $0.01, c8g.2xlarge $0.02, c8g.8xlarge $0.08; the listing *recommends* m8g.4xlarge, $0.04/h).
- **7-day free trial, then it converts to paid automatically** unless you cancel. The trial
  waives the software fee only; EC2, storage and everything else are billed from minute one.
  After conversion you pay $0.01/h only while an instance from the AMI is running. No refunds.
- **When to subscribe:** the day you first launch (step 5). With the fee at $0.01/h, timing
  barely matters: the whole judging period costs $3.36 in software fees even with no trial.
  Starting the trial on the day of the 3-way benchmark means the benchmark runs free.
- After subscribing: **Continue to Configuration** -> region us-east-1 -> copy the **AMI id**
  (`ami-...`). That is `cool_ami_id`. Do not launch from the Marketplace page; Terraform launches it.
- Per the listing, OpenCV's Python bindings live in `/opt/cool/python_3.12/site-packages`
  and need `/opt/cool/cpp_sdk/lib` on `LD_LIBRARY_PATH` (prebuilt venvs exist in
  `/opt/cool/venvs/`). The user-data uses exactly that. If they cannot be imported, it falls back to the
  stock wheel and writes `pip-fallback` to `/opt/floorsight/OPENCV_SOURCE`, so the demo still runs and
  the benchmark JSON shows the build that actually ran.

## 4. Prepare Terraform (15 min, free)

Needs Terraform >= 1.6 (the 23 Oct apply used 1.16.6).

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars    # gitignored
# fill in: cool_ami_id, alert_email, app_repo_url, app_git_ref (pin a commit SHA), app_start_command
terraform init
terraform fmt -check && terraform validate
terraform plan -out floorsight.tfplan           # read it; expect ~30 resources, 1 instance
```

The instance clones the code from `app_repo_url` at `app_git_ref` (or unpacks `code_bundle_key` from the
artifacts bucket), so the repo must be reachable without credentials (public GitHub repo, or change the
user-data). systemd then runs `app_start_command` from `/opt/floorsight/app`, with `--host`/`--port` added
(127.0.0.1:8000 while private, 0.0.0.0:8000 when public). **`app_start_command` is required**: Terraform refuses
an empty one, which would install a service that never starts. The example file uses the command line that
app-run-10 ran on the instance (through `infra/aws/jobs/run_app.sh`, `SOURCE=simlive`):

```bash
/opt/floorsight/py -m floorsight.server --source simlive --vision real --agent real
```

That is the closed-loop simulated warehouse; it renders its own frames, so nothing is rendered at boot. It has
not yet run as the systemd service itself (see Status). A command with `--source sim` instead renders demo
footage on first boot (about 10 minutes).

Variables worth a look:

| Variable | Default | Note |
|---|---|---|
| `app_start_command` | none (required) | See above |
| `agent_llm` | `local` | `bedrock` makes every agent step a Bedrock call. Read the Bedrock cost row first |
| `judge_token` | empty | If set, the dashboard needs `?token=...` once; the server swaps it for an HttpOnly cookie and redirects to the same page without it. **Required (at least 20 characters) when `public_endpoint = true`** |
| `preservation_hold` | `true` | Nothing expires (see "Retention is conditional" above). `false` turns on `evidence_retention_days`, `track_ttl_seconds`, `log_retention_days` and the app's own purge |
| `budget_limit_usd` | 50 | Monthly. Judging spans two months (Oct 27-31, Nov 1-9), so each month stays under 50 |
| `baseline_x86_count`, `baseline_arm_stock_count` | 0 | Set to 1 only for the benchmark hour |

## 5. Apply (20 min)

```bash
terraform apply floorsight.tfplan
terraform output judge_url
```

1. Confirm the SNS subscription email ("AWS Notification - Subscription Confirmation").
2. Wait ~15 minutes for boot (plus ~10 for the footage render if the command uses `--source sim`), then follow the boot log:
   `aws ssm start-session --target <instance_id>` then `sudo tail -f /var/log/floorsight-bootstrap.log`.
   The last line says `bootstrap done: ... opencv=cool` (or `pip-fallback`: investigate).
3. CloudFront takes 5-15 minutes to deploy the distribution.

## 6. Smoke test (10 min)

- [ ] `curl -s -w ' %{http_code}\n' https://<cloudfront-domain>/readyz` prints `{"ready":true,...} 200`. A 503
      lists the reasons (for example `perception fell back to the null vision: nothing is being detected`).
      Do not use `/healthz` here: it only says the process is up, even when perception has failed.
      While private, run the same against `http://127.0.0.1:8000/readyz` through the port-forward (step 8).
- [ ] One application API answers: `curl -sf -H "Authorization: Bearer <token>" https://<cloudfront-domain>/api/state`
      returns JSON with `"mode": "live"` (or `"ended"` once a recording or the simulated scenario has finished)
      and `components.vision` other than `"none"`; with a judge token set, the same request without the header
      returns 401.
- [ ] `judge_url` opens the dashboard over HTTPS; the floor map moves; websocket connects.
- [ ] Direct access is refused: `http://<elastic-ip>:8000/` times out (security group allows CloudFront only).
- [ ] On the instance: `/opt/floorsight/py -c "import cv2; print(cv2.__version__, cv2.__file__)"` prints 5.x from `/opt/cool/...`.
- [ ] CloudWatch dashboard `floorsight` shows the CPU widget and the app log.
- [ ] After an event in the demo, an evidence object appears in the bucket (it disappears after a day only if
      `preservation_hold = false`).
- [ ] One test alert arrives by email (if the app publishes to SNS).
- [ ] Only if `agent_llm = bedrock`: the agent trace shows `mode: bedrock` steps; check the Bedrock
      invocation count in CloudWatch after one 3-minute loop and multiply out (cost table).

## 7. Run the COOL benchmark (1 h, ~$1)

Follow `scripts/benchmark/README.md`: same commit, same footage, three legs
(c7i.xlarge stock / c8g stock / c8g COOL). Set `baseline_x86_count = 1` (and optionally
`baseline_arm_stock_count = 1`), `terraform apply`, run, copy the JSON files out
(`cat` them in the SSM session), then set the counts back to 0 and apply again.

## 8. During judging (Oct 27 - Nov 9)

The deployment is private by default: the dashboard listens on 127.0.0.1 and the only way in is an SSM port-forward:

```bash
aws ssm start-session --region us-east-1 --target <instance-id>   --document-name AWS-StartPortForwardingSession --parameters portNumber=8000,localPortNumber=8000
# then open http://127.0.0.1:8000/
```

The public judge link is one step, and only after the owner's yes:

```bash
cd infra/terraform
terraform apply -var public_endpoint=true -var 'judge_token=<random string, at least 20 characters>'
terraform output judge_url          # https://xxxx.cloudfront.net/ ; share it with ?token=<token>
sudo systemctl restart floorsight   # on the instance, over SSM, so the app picks up the public settings
```

That apply creates an Elastic IP and a CloudFront HTTPS distribution, opens the app port to CloudFront's
origin-facing addresses only, and makes the app listen on all interfaces. Terraform refuses `public_endpoint = true`
without a judge token of at least 20 characters. The server refuses any request without CloudFront's secret
`X-Floorsight-Origin` header (`tests/server/test_origin_guard.py`), and the judge token gates the dashboard: the
`?token=` link sets an HttpOnly, SameSite=Lax, Secure cookie and redirects to the same page without the token.
Back to private: `terraform apply -var public_endpoint=false`.

**On the instance created on 23 Oct, the apply alone is not enough.** Its user data ran once, at first boot, with
`public_endpoint = false`, no judge token and an empty `app_start_command`, and Terraform ignores later user-data
changes (`ignore_changes = [user_data, ami]`). So `/etc/floorsight.env` there has no token and no origin secret, and
`floorsight.service` has no command. Before sharing a link, over SSM: set `FLOORSIGHT_TOKEN=<token>` and
`FLOORSIGHT_ORIGIN_SECRET=<terraform output -raw origin_secret>` in `/etc/floorsight.env`, give the service an
`ExecStart` ending in `--host 0.0.0.0 --port 8000` (`sudo systemctl edit --full floorsight`), restart it, then check
through the link that `/readyz` returns 200 and `/api/state` without the token returns 401. Every plan now also needs
a non-empty `app_start_command` in `terraform.tfvars`; on this instance it only feeds the ignored user data, so it
changes nothing there.

- Keep the instance running; check the budget emails.
- If you need to stop spending temporarily: `aws ec2 stop-instances --instance-ids <id>`.
  Stopped = no EC2 or COOL fee, but EBS and the Elastic IP still bill (about $0.20/day).

## 9. Tear-down (10 min) - do this right after judging

```bash
cd infra/terraform
terraform destroy
```

Then by hand:

1. AWS Marketplace -> **Manage subscriptions** -> cancel the COOL subscription.
2. Bedrock model access can stay (no cost while unused).
3. Delete the manual budget from step 1 when you are sure nothing is left.
4. Check **Billing -> Bills** the next day for anything still accruing (EBS snapshots, Elastic IPs).

---

## Costs

Prices are on-demand Linux, us-east-1. EC2 rates are **approximate**: taken from the Holori
price aggregator (updated 2026-10-16), not from AWS directly. **Check the AWS pricing
page before relying on them.** COOL software fees are from the Marketplace listing (2026-10-16).

| Item | $/hour | Note |
|---|---|---|
| `c8g.xlarge` EC2 (4 vCPU Graviton4, 8 GiB) | ~0.1595 | approximate |
| COOL software fee on `c8g.xlarge` | 0.01 | Marketplace listing; waived during the 7-day trial |
| **App instance total** | **~0.1695** | |
| `c8g.4xlarge` EC2 (16 vCPU) | ~0.6381 | approximate; COOL fee 0.04 -> ~0.678/h |
| `c7i.xlarge` EC2 (x86 baseline, 4 vCPU) | ~0.1785 | approximate; benchmark only |
| Elastic IP / public IPv4 | 0.005 | always billed while allocated |
| EBS gp3 30 GiB | ~0.0033 | $0.08 per GiB-month, approximate |
| CloudFront, S3, DynamoDB, SNS, Logs | ~0 | within free tiers / cents at demo volume |
| CloudWatch custom metrics | ~0.004 | ~10 metrics at $0.30/metric-month, the first 10 free; PutMetricData is batched to <= 1 call/min |
| CloudWatch dashboard | 0 | first 3 dashboards free |

**Judging period (Oct 27 - Nov 9 inclusive = 14 days = 336 h), app instance only, `agent_llm = local`:**

| | |
|---|---|
| c8g.xlarge EC2 336 h x 0.1595 | ~$53.59 |
| COOL fee 336 h x 0.01 (if the trial is already used up) | $3.36 |
| Elastic IP 336 h x 0.005 | $1.68 |
| EBS 30 GiB, 14 days | ~$1.12 |
| CloudWatch, S3, DynamoDB, SNS, CloudFront | < $2 |
| **Total** | **about $60** (about $21 in October, $39 in November) |

Add ~$1 for the benchmark hour (c7i.xlarge + c8g instances for ~1 h each) and ~$5-10 for
setup days before judging if the instance stays up.

**Bedrock (only if `agent_llm = bedrock`):** every agent step is one Claude call, and the demo
replays in a loop (480 loops a day). Cost = calls per loop x 480 x days x
(input tokens x input price + output tokens x output price). Claude Opus 5.5 is $4 / $20 per
million input/output tokens on Anthropic's own API; Bedrock prices it separately (check the
Bedrock pricing page). Example: 20 calls per loop x 3,000 input + 400 output tokens ->
~$0.02 per call -> ~$0.40 per loop -> **~$190 a day, ~$2,700 for the judging period.**
Do not run Bedrock mode in an endless loop. Either keep `agent_llm = local` for the looping
demo and switch to `bedrock` for a scheduled live demo window, or have the agent stream add
an hourly Bedrock call cap first, and confirm with a one-loop measurement (smoke test, last item).

## Known gaps (decide before applying; two were checked on the 23 Oct instance)

- IAM action names for the Bedrock Mantle endpoint (step 2.3).
- Python bindings path inside the COOL AMI: checked on the 23 Oct instance (`/opt/cool/python_3.12/site-packages`,
  OpenCV 5.1.0-dev; `docs/receipts/cool-opencv-build-20261023.txt`).
- FFmpeg video I/O in the COOL OpenCV build (the footage render and `--source sim` need it): present
  (`FFMPEG: YES` in the same receipt).
- CloudFront -> instance traffic is plain HTTP on port 8000 inside AWS (viewer side is HTTPS).
  CloudFront adds a secret `X-Floorsight-Origin` header (`terraform output -raw origin_secret`);
  the server refuses requests without it in public mode (`tests/server/test_origin_guard.py`).
- The origin secret and judge token are in the instance user-data (readable by anyone in the
  account with `ec2:DescribeInstanceAttribute`) and in the local Terraform state. Keep state off git.
- The SSM agent is assumed to be present in the COOL AMI (Ubuntu ships it; user-data also tries `snap install`).
