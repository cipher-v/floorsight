#!/bin/bash
# Review-page copy of a side-by-side video: VP9 WebM (plays in headless Chromium). Needs: BUCKET, SRC (mp4 on the instance).
# Output: s3 review/side-by-side/<basename SRC .mp4>.webm
set -uo pipefail
OUT="${SRC%.mp4}.webm"
nice -n 10 ffmpeg -v error -y -i "$SRC" -c:v libvpx-vp9 -b:v 0 -crf 34 -deadline good -cpu-used 5 -row-mt 1 -threads 4 -an "$OUT" \
  && aws s3 cp --region us-east-1 "$OUT" "s3://$BUCKET/review/side-by-side/$(basename "$OUT")" --only-show-errors && echo "webm ok $(stat -c %s "$OUT")"
