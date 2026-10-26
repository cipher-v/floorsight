#!/bin/bash
# Review-page clip: SRC (video on the instance) from START to END (s) as VP9 WebM, 1280 wide. Needs: BUCKET, SRC, START, END, NAME.
# Output: s3 review/clips/<NAME>.webm (simulated or privacy-filtered footage only)
set -uo pipefail
O=/opt/floorsight/jobs/review-frames/clips; mkdir -p "$O"
nice -n 10 ffmpeg -v error -y -ss "$START" -to "$END" -i "$SRC" -c:v libvpx-vp9 -b:v 0 -crf 34 -deadline good -cpu-used 5 -row-mt 1 -threads 4 \
  -vf scale=1280:-2 -an "$O/$NAME.webm" && aws s3 cp --region us-east-1 "$O/$NAME.webm" "s3://$BUCKET/review/clips/$NAME.webm" --only-show-errors && echo "webm ok $(stat -c %s "$O/$NAME.webm")"
