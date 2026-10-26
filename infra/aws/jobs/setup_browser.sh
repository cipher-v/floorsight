#!/bin/bash
# Headless Chromium (Playwright, linux-arm64) for recording the dashboard on the instance. Root (apt deps).
# Needs: JOB, BUCKET.
set -uo pipefail
J=/opt/floorsight/jobs/$JOB
if [ -z "${DETACHED:-}" ]; then
  [ -e "$J" ] && { echo "job dir $J exists"; exit 2; }
  mkdir -p "$J"; cp "$0" "$J/run.sh"
  systemd-run --unit="fs-$JOB" --setenv=DETACHED=1 --setenv=JOB="$JOB" --setenv=BUCKET="$BUCKET" --property=WorkingDirectory=/opt/floorsight bash "$J/run.sh"
  echo "started fs-$JOB"; exit 0
fi
if [ -e "$J/STARTED" ]; then echo "restart refused $(date -u +%FT%TZ)" >> "$J/restarts.txt"; exit 0; fi
date -u +%FT%TZ > "$J/STARTED"
exec >> "$J/log.txt" 2>&1
set -x
export DEBIAN_FRONTEND=noninteractive PLAYWRIGHT_BROWSERS_PATH=/opt/floorsight/browsers
python3.12 -m venv /opt/floorsight/venv-browser
/opt/floorsight/venv-browser/bin/pip install -q "playwright==1.55.0" pillow || /opt/floorsight/venv-browser/bin/pip install -q playwright pillow
/opt/floorsight/venv-browser/bin/python -m playwright install --with-deps chromium
apt-get install -y -q fonts-dejavu-core fonts-noto-core >/dev/null || true
chown -R floorsight: /opt/floorsight/venv-browser /opt/floorsight/browsers
sudo -u floorsight env PLAYWRIGHT_BROWSERS_PATH=/opt/floorsight/browsers /opt/floorsight/venv-browser/bin/python - <<'PY'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=["--no-sandbox"])
    pg = b.new_page(); pg.set_content("<h1>ok</h1>"); print("chromium", b.version, pg.inner_text("h1")); b.close()
PY
echo "BROWSER_EXIT=$?"
/opt/floorsight/venv-browser/bin/pip freeze > "$J/freeze-browser.txt"
aws s3 cp --region us-east-1 --recursive "$J" "s3://$BUCKET/jobs/$JOB/" --only-show-errors
echo DONE > "$J/DONE"
