#!/bin/bash
# Render an SVG (uploaded to s3 SRC) to PNG with the instance's headless Chromium; PNG to s3 DST. Needs: BUCKET, SRC, DST (keys).
set -euo pipefail
D=$(mktemp -d); chown floorsight "$D"
aws s3 cp --region us-east-1 "s3://$BUCKET/$SRC" "$D/in.svg" --only-show-errors
chmod 644 "$D/in.svg"
sudo -u floorsight env PLAYWRIGHT_BROWSERS_PATH=/opt/floorsight/browsers /opt/floorsight/venv-browser/bin/python - "$D/in.svg" "$D/out.png" <<'PY'
import re, sys
from playwright.sync_api import sync_playwright
src, out = sys.argv[1], sys.argv[2]
head = open(src, encoding="utf-8").read(400)
w, h = (int(v) for v in re.search(r'width="(\d+)" height="(\d+)"', head).groups())
with sync_playwright() as p:
    b = p.chromium.launch(args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=1.5)
    pg.goto("file://" + src); pg.wait_for_timeout(500)
    pg.screenshot(path=out, clip={"x": 0, "y": 0, "width": w, "height": h})
    b.close()
print("rendered", w, h)
PY
aws s3 cp --region us-east-1 "$D/out.png" "s3://$BUCKET/$DST" --only-show-errors && echo "uploaded $DST"
