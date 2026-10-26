#!/bin/bash
# Review-page copies of a built demo video: VP9 WebM (plays in headless Chromium) and a poster frame. Needs: JOB, BUCKET.
# Output: s3 jobs/<JOB>/review/floorsight-demo.webm, poster.jpg
set -uo pipefail
J=/opt/floorsight/jobs/$JOB; O=$J/review; mkdir -p "$O"
nice -n 10 ffmpeg -v error -y -i "$J/out/floorsight-demo.mp4" -c:v libvpx-vp9 -b:v 0 -crf 36 -deadline good -cpu-used 5 -row-mt 1 -threads 4 \
  -vf scale=1280:-2 -c:a libopus -b:a 64k "$O/floorsight-demo.webm" && echo "webm ok $(stat -c %s "$O/floorsight-demo.webm")"
ffmpeg -v error -y -ss 90 -i "$J/out/floorsight-demo.mp4" -frames:v 1 -q:v 3 -vf scale=1280:-2 "$O/poster.jpg" && echo poster ok
aws s3 cp --region us-east-1 "$O/floorsight-demo.webm" "s3://$BUCKET/jobs/$JOB/review/floorsight-demo.webm" --only-show-errors
aws s3 cp --region us-east-1 "$O/poster.jpg" "s3://$BUCKET/jobs/$JOB/review/poster.jpg" --only-show-errors && echo uploaded
