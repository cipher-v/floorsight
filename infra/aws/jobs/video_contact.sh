#!/bin/bash
# Contact sheet of a built demo video: one frame at each given time, 4 per row, for review. Needs: JOB, BUCKET, TIMES.
# Output: s3 jobs/<JOB>/review/contact-<n>.jpg (frames from the finished, privacy-filtered video only)
set -uo pipefail
J=/opt/floorsight/jobs/$JOB; O=$J/review; mkdir -p "$O"; rm -f "$O"/f-*.jpg
for t in $TIMES; do ffmpeg -v error -y -ss "$t" -i "$J/out/floorsight-demo.mp4" -frames:v 1 -q:v 3 -vf "scale=800:-2" "$O/f-$(printf %07.2f $t).jpg"; done
/opt/floorsight/perc - "$O" <<'PY'
import sys, glob, cv2, numpy as np
fs = sorted(glob.glob(sys.argv[1] + "/f-*.jpg")); ims = [cv2.imread(f) for f in fs]
for n in range(0, len(ims), 8):
    grp = ims[n:n + 8]
    for im, f in zip(grp, fs[n:n + 8]):
        cv2.putText(im, f.split("f-")[-1][:-4] + " s", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    while len(grp) % 4: grp.append(np.zeros_like(ims[0]))
    rows = [np.hstack(grp[i:i + 4]) for i in range(0, len(grp), 4)]
    cv2.imwrite(f"{sys.argv[1]}/contact-{n // 8}.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("frames", len(ims))
PY
aws s3 cp --region us-east-1 --recursive "$O" "s3://$BUCKET/jobs/$JOB/review/" --exclude "f-*" --only-show-errors && echo uploaded
