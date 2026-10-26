#!/bin/bash
# Crops of the PRIVACY-FILTERED video (no raw pixels) around reference persons whose head was not covered.
# Needs: JOB, CLIP, BUCKET, PICKS ("frame:id frame:id ..."). Output: s3 jobs/<JOB>/qa/<CLIP>-uncovered.jpg
set -uo pipefail
J=/opt/floorsight/jobs/$JOB; O=$J/qa; mkdir -p "$O"; chown floorsight "$O"
sudo -u floorsight /opt/floorsight/perc -P - "$J/$CLIP/$CLIP-filtered.mp4" "/opt/floorsight/jobs/${REF_JOB:-perc-full-02}/reference/$CLIP-detections.jsonl" "$PICKS" "$O/$CLIP-uncovered.jpg" <<'PY'
import json, sys, cv2, numpy as np
video, ref, picks, out = sys.argv[1:5]
want = {}
for p in picks.split():
    f, pid = p.split(":"); want.setdefault(int(f), []).append(pid)
boxes = {}
for r in map(json.loads, open(ref)):
    for p in r["persons"]:
        if p["id"] in want.get(r["frame"], []): boxes[(r["frame"], p["id"])] = p["box"]
cap, i, tiles = cv2.VideoCapture(video), 0, []
while want and i <= max(want):
    ok, img = cap.read()
    if not ok: break
    for pid in want.get(i, []):
        x1, y1, x2, y2 = map(int, boxes[(i, pid)]); w, h = x2 - x1, y2 - y1
        X1, Y1, X2, Y2 = max(0, x1 - w), max(0, y1 - h // 2), min(img.shape[1], x2 + w), min(img.shape[0], y2 + h // 2)
        crop = img[Y1:Y2, X1:X2].copy()
        cv2.rectangle(crop, (x1 - X1, y1 - Y1), (x2 - X1, y2 - Y1), (255, 0, 255), 1)
        crop = cv2.resize(crop, (300, int(300 * crop.shape[0] / max(1, crop.shape[1]))))[:400]
        canvas = np.zeros((420, 300, 3), np.uint8); canvas[:crop.shape[0]] = crop
        cv2.putText(canvas, f"f{i} {pid}", (4, 414), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(canvas)
    i += 1
rows = [np.hstack(tiles[k:k + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[k:k + 4]))) for k in range(0, len(tiles), 4)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); print("tiles", len(tiles), out)
PY
aws s3 cp --region us-east-1 "$O/$CLIP-uncovered.jpg" "s3://$BUCKET/jobs/$JOB/qa/$CLIP-uncovered.jpg" --only-show-errors && echo uploaded
