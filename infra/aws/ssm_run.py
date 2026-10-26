"""Run a shell script on the Floorsight instance through SSM Run Command and print its output.

    python infra/aws/ssm_run.py --instance i-... script.sh [--timeout 3600] [--env K=V ...]

No SSH, no inbound port: the script text travels through the SSM API. Long jobs should run detached
(nohup/systemd-run) and write their logs to S3; this helper is for setup steps and short checks.
Output is truncated by SSM at ~24 kB; the full output is in the command's S3 copy when --s3-out is set.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def aws(*args: str) -> dict:
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "AWS_PAGER": ""}  # aws.exe on Windows
    out = subprocess.run(["aws", *args, "--output", "json"], capture_output=True, text=True, encoding="utf-8", env=env)
    if out.returncode != 0:
        raise SystemExit(out.stderr.strip())
    return json.loads(out.stdout or "{}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("script", type=Path)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--env", action="append", default=[], help="K=V exported before the script")
    ap.add_argument("--s3-out", default="", help="s3://bucket/prefix for the full stdout/stderr")
    ap.add_argument("--comment", default="")
    a = ap.parse_args()
    body = a.script.read_text(encoding="utf-8").replace("\r\n", "\n")
    exports = "".join(f"export {kv.split('=', 1)[0]}='{kv.split('=', 1)[1]}'\n" for kv in a.env)
    # SSM runs commands with /bin/sh: save the script and run it with bash (it can then copy "$0" for detached jobs)
    tag = f"{a.script.stem}-{int(time.time())}"
    wrapped = "\n".join([
        "mkdir -p /opt/floorsight/ssm",
        f"cat > /opt/floorsight/ssm/{tag}.sh <<'FS_SSM_EOF'", body, "FS_SSM_EOF",
        f"{exports}bash /opt/floorsight/ssm/{tag}.sh", ""])
    params = {"commands": [wrapped], "executionTimeout": [str(a.timeout)]}
    args = ["ssm", "send-command", "--region", a.region, "--instance-ids", a.instance, "--document-name", "AWS-RunShellScript",
            "--parameters", json.dumps(params), "--timeout-seconds", "600", "--comment", (a.comment or a.script.name)[:100]]
    if a.s3_out:
        b, _, pfx = a.s3_out.removeprefix("s3://").partition("/")
        args += ["--output-s3-bucket-name", b, "--output-s3-key-prefix", pfx]
    cid = aws(*args)["Command"]["CommandId"]
    print(f"command {cid} sent", file=sys.stderr)
    t0 = time.time()
    while True:
        time.sleep(3)
        try:
            inv = aws("ssm", "get-command-invocation", "--region", a.region, "--command-id", cid, "--instance-id", a.instance)
        except SystemExit as e:
            if "InvocationDoesNotExist" in str(e) and time.time() - t0 < 60:
                continue
            raise
        if inv["Status"] not in ("Pending", "InProgress", "Delayed"):
            break
        if time.time() - t0 > a.timeout + 120:
            print("timed out waiting", file=sys.stderr)
            return 3
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.write(inv.get("StandardOutputContent", ""))
    err = inv.get("StandardErrorContent", "")
    if err:
        sys.stderr.write("--- stderr ---\n" + err)
    print(f"\n[status={inv['Status']} rc={inv.get('ResponseCode')} {time.time() - t0:.0f}s]", file=sys.stderr)
    return 0 if inv["Status"] == "Success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
