#!/bin/bash
# Review sheets from any privacy-filtered video on the instance: one frame per given time, COLS per row, ROWS rows per sheet.
# Needs: BUCKET, SRC (video path on the instance), NAME (output prefix), TIMES. Optional: COLS (4), ROWS (3), W (640).
# Output: s3 review/frames/<NAME>-<n>.jpg
set -uo pipefail
O=/opt/floorsight/jobs/review-frames/$NAME; mkdir -p "$O"; rm -f "$O"/*.jpg
for t in $TIMES; do ffmpeg -v error -y -ss "$t" -i "$SRC" -frames:v 1 -q:v 3 -vf "scale=${W:-640}:-2" "$O/f-$(printf %07.2f $t).jpg"; done
/opt/floorsight/perc - "$O" "${COLS:-4}" "${ROWS:-3}" "$NAME" <<'PY'
import sys, glob, cv2, numpy as np
d, cols, rows, name = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
fs = sorted(glob.glob(d + "/f-*.jpg")); ims = [cv2.imread(f) for f in fs]
n = cols * rows
for s in range(0, len(ims), n):
    grp = ims[s:s + n]
    for im, f in zip(grp, fs[s:s + n]):
        cv2.rectangle(im, (0, 0), (150, 34), (0, 0, 0), -1)
        cv2.putText(im, f.split("f-")[-1][:-4].lstrip("0") + " s", (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    while len(grp) % cols: grp.append(np.zeros_like(ims[0]))
    sheet = np.vstack([np.hstack(grp[i:i + cols]) for i in range(0, len(grp), cols)])
    cv2.imwrite(f"{d}/{name}-{s // n}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
print("frames", len(ims))
PY
aws s3 cp --region us-east-1 --recursive "$O" "s3://$BUCKET/review/frames/" --exclude "f-*" --only-show-errors && echo uploaded
