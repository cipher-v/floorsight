#!/bin/bash
# Before/after video: two privacy-filtered overlay videos of the same clip side by side, each captioned. Numbers-free.
# Needs: BUCKET, LEFT, RIGHT (overlay mp4 paths on the instance), OUT (path), START, END. Optional: LTXT, RTXT.
# Output: OUT on the instance and s3 review/side-by-side/<basename OUT>
set -euo pipefail
mkdir -p "$(dirname "$OUT")"
T1="${LTXT:-before}"; T2="${RTXT:-after}"
ffmpeg -v error -y -ss "$START" -to "$END" -i "$LEFT" -ss "$START" -to "$END" -i "$RIGHT" -filter_complex \
  "[0:v]scale=960:540,setsar=1,drawbox=x=0:y=0:w=iw:h=46:color=0x101418@0.85:t=fill,drawtext=text='$T1':x=16:y=10:fontsize=28:fontcolor=white[a];\
   [1:v]scale=960:540,setsar=1,drawbox=x=0:y=0:w=iw:h=46:color=0x101418@0.85:t=fill,drawtext=text='$T2':x=16:y=10:fontsize=28:fontcolor=white[b];\
   [a][b]hstack=inputs=2[v]" -map "[v]" -an -c:v libx264 -pix_fmt yuv420p -crf 20 -r 25 "$OUT"
chown floorsight: "$OUT"
aws s3 cp --region us-east-1 "$OUT" "s3://$BUCKET/review/side-by-side/$(basename "$OUT")" --only-show-errors && echo "uploaded $(basename "$OUT")"
