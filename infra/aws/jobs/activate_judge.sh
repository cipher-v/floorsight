#!/bin/bash
# Existing COOL instance only. CODE_DIR is an already extracted, versioned
# public source bundle. JUDGE_CONFIG is a private JSON file from publish_judge.py.
# Preserves prior deployments and data. No external SNS messages from judges.
set -euo pipefail
: "${CODE_DIR:?Set the absolute versioned code directory}"
: "${JUDGE_CONFIG:?Set the private endpoint JSON path}"
case "$CODE_DIR" in /opt/floorsight/judge-*) ;; *) echo "Unexpected code directory"; exit 2;; esac
test -f "$CODE_DIR/floorsight/server/app.py"
export CODE_DIR JUDGE_CONFIG
python3 - <<'PY'
import json, os
from pathlib import Path
cfg = json.loads(Path(os.environ['JUDGE_CONFIG']).read_text())
assert len(cfg['judge_token']) >= 20 and len(cfg['origin_secret']) >= 20
values = {}
for line in Path('/etc/floorsight.env').read_text().splitlines():
    if '=' in line and not line.startswith('#'):
        k, v = line.split('=', 1)
        values[k] = v.strip('"\'')
values.update(FLOORSIGHT_TOKEN=cfg['judge_token'], FLOORSIGHT_ORIGIN_SECRET=cfg['origin_secret'],
    FLOORSIGHT_LLM='local', FLOORSIGHT_CLOUD='aws', FLOORSIGHT_SNS_TOPIC_ARN='',
    FLOORSIGHT_SESSION_ID='judge-20261027', FLOORSIGHT_S3_PREFIX='evidence/judge-20261027/',
    FLOORSIGHT_PRESERVATION_HOLD='on')
target = Path('/etc/floorsight-judge.env')
target.write_text('\n'.join(k+'='+v for k,v in values.items())+'\n')
target.chmod(0o600)
PY
mkdir -p /opt/floorsight/judge-runtime
chown -R floorsight:floorsight "$CODE_DIR" /opt/floorsight/judge-runtime
cat > /etc/systemd/system/floorsight-judge.service <<EOF
[Unit]
Description=Floorsight protected judge simulator
After=network-online.target
Wants=network-online.target
[Service]
User=floorsight
WorkingDirectory=$CODE_DIR
EnvironmentFile=/etc/floorsight-judge.env
EnvironmentFile=-/etc/floorsight.opencv.env
ExecStart=/opt/floorsight/py -m floorsight.server --source simlive --vision real --agent real --loop --sim-duration 40 --speed 0.5 --no-drop --host 0.0.0.0 --port 8000 --runtime-dir /opt/floorsight/judge-runtime
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true
UMask=0077
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now floorsight-judge
systemctl restart floorsight-judge
echo "Judge service activated; check readiness and authenticated application state."
