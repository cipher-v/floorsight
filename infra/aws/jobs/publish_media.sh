#!/bin/bash
# Public-repo media (26 Oct): browser-playable H.264 copies of the simulator videos (OpenCV wrote them as MPEG-4 Part 2,
# which browsers cannot play) and an animated GIF preview of every published video, for the README.
# Needs: BUCKET. Sources: the S3 keys below (privacy-filtered or simulated footage only; raw clips are never read).
# Output: s3 publish/github-20261026/{h264,previews}/ and publish/github-20261026/DONE
set -uo pipefail
W=/opt/floorsight/jobs/publish-20261026; S=$W/src; H=$W/h264; P=$W/previews
mkdir -p "$S" "$H" "$P"
R=s3://$BUCKET; OUT=$R/publish/github-20261026; REV=review/opencv-owner-20261024-03/media
get() { [ -s "$S/$2" ] || { mkdir -p "$(dirname "$S/$2")"; aws s3 cp --region us-east-1 "$R/$1" "$S/$2" --only-show-errors; }; }

# repo path <- s3 key
get $REV/real/aws-20261024c/floorsight-demo.mp4 floorsight-demo.mp4
get jobs/interlock-sim-08/out/interlock-off.mp4 simulator/interlock-off.mp4
get jobs/interlock-sim-08/out/interlock-on.mp4 simulator/interlock-on.mp4
for c in 1 2 3 4; do get inputs/sim/demo_001/cam$c.mp4 simulator/cameras-demo_001/cam$c.mp4; done
get jobs/app-run-09/session/video/29df6c67b4f4ce5372b0b55ea32612db.webm app/app-real-clips.webm
get jobs/app-run-10/session/video/15b397654bf972f4e0b86ea0f49ed1de.webm app/app-simulator.webm
for c in clip-dev-002 clip-dev-003 clip-dev-004 screen-002 screen-032 screen-033; do
  get jobs/perc-full-13/$c/$c-overlay.mp4 real-clips/$c-overlay.mp4
  get jobs/perc-full-13/$c/$c-filtered.mp4 real-clips/privacy-filtered/$c-filtered.mp4
done
get jobs/perc-full-13/failclosed-test/failclosed-overlay.mp4 real-clips/fail-closed-test-overlay.mp4
get jobs/perc-full-13/failclosed-test/failclosed-filtered.mp4 real-clips/privacy-filtered/fail-closed-test-filtered.mp4
get review/side-by-side/screen-002-final-r46.mp4 smoothing/screen-002-raw-vs-smoothed.mp4
get $REV/region-history.webm earlier/2026-10-17-simulator-region-history.webm
get $REV/tracking.mp4 earlier/2026-10-17-simulator-tracking.mp4
for c in clip-dev-001 clip-dev-002 clip-dev-004; do get $REV/real/$c-overlay.mp4 earlier/2026-10-19-$c-overlay.mp4; done
get $REV/real/zone/clip-dev-004-zone.mp4 earlier/2026-10-20-clip-dev-004-zone.mp4
get $REV/real/zone/screen-002-zone.mp4 earlier/2026-10-20-screen-002-zone.mp4
get $REV/real/warn/screen-002-halfspeed.mp4 earlier/2026-10-20-screen-002-swerve-halfspeed.mp4
get $REV/real/app/app-run.mp4 earlier/2026-10-21-app-run.mp4
get $REV/real/aws-20261024/floorsight-demo.mp4 earlier/2026-10-24-demo-draft-1.mp4
get $REV/real/aws-20261024b/floorsight-demo.mp4 earlier/2026-10-24-demo-draft-2.mp4
get $REV/real/aws-20261024b/before-after-screen-002.mp4 earlier/2026-10-24-screen-002-before-after-draft.mp4
echo "sources: $(find "$S" -type f | wc -l) files"

# 1. H.264 copies of the MPEG-4 Part 2 simulator videos (same frames, same paths in the repo)
for f in simulator/interlock-off.mp4 simulator/interlock-on.mp4 simulator/cameras-demo_001/cam{1,2,3,4}.mp4; do
  mkdir -p "$(dirname "$H/$f")"
  nice -n 10 ffmpeg -v error -y -i "$S/$f" -c:v libx264 -preset slow -crf 22 -pix_fmt yuv420p -movflags +faststart -an "$H/$f" \
    && echo "h264 $f $(stat -c %s "$H/$f")"
done

# 2. GIF previews. gif NAME SRC START DUR SPEED WIDTH FPS
PAL="split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
gif() {
  nice -n 10 ffmpeg -v error -y -ss "$3" -t "$4" -i "$S/$2" -vf "setpts=PTS/$5,fps=$7,scale=$6:-1:flags=lanczos,$PAL" -loop 0 "$P/$1.gif" \
    && echo "gif $1 $(stat -c %s "$P/$1.gif")"
}
# montage NAME SRC WIDTH SEG_DUR T1 T2 ... : short cuts from a long video, joined
montage() {
  local name=$1 src=$2 w=$3 d=$4; shift 4; local ins=() fc="" n=0
  for t in "$@"; do ins+=(-ss "$t" -t "$d" -i "$S/$src"); fc+="[$n:v]fps=8,scale=$w:-1:flags=lanczos,setsar=1[v$n];"; n=$((n+1)); done
  for i in $(seq 0 $((n-1))); do fc+="[v$i]"; done
  nice -n 10 ffmpeg -v error -y "${ins[@]}" -filter_complex "${fc}concat=n=$n:v=1:a=0,$PAL" -loop 0 "$P/$name.gif" \
    && echo "gif $name $(stat -c %s "$P/$name.gif")"
}
montage floorsight-demo floorsight-demo.mp4 640 1.6 19 46 63 92 140 152 168
gif interlock-off simulator/interlock-off.mp4 0 40 4 640 6
gif interlock-on simulator/interlock-on.mp4 0 40 4 640 6
nice -n 10 ffmpeg -v error -y -ss 30 -t 60 -i "$S/simulator/cameras-demo_001/cam1.mp4" -ss 30 -t 60 -i "$S/simulator/cameras-demo_001/cam2.mp4" \
  -ss 30 -t 60 -i "$S/simulator/cameras-demo_001/cam3.mp4" -ss 30 -t 60 -i "$S/simulator/cameras-demo_001/cam4.mp4" \
  -filter_complex "[0:v]scale=320:-2[a];[1:v]scale=320:-2[b];[2:v]scale=320:-2[c];[3:v]scale=320:-2[d];[a][b][c][d]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0,setpts=PTS/6,fps=6,$PAL" \
  -loop 0 "$P/cameras-demo_001.gif" && echo "gif cameras-demo_001 $(stat -c %s "$P/cameras-demo_001.gif")"
gif app-real-clips app/app-real-clips.webm 82 18 2 720 6
gif app-simulator app/app-simulator.webm 4 30 3 720 6
for c in clip-dev-002 clip-dev-003 clip-dev-004 screen-002 screen-032 screen-033; do
  gif $c-overlay real-clips/$c-overlay.mp4 0 15 1.5 480 6
  gif $c-filtered real-clips/privacy-filtered/$c-filtered.mp4 0 15 1.5 400 5
done
gif fail-closed-test-overlay real-clips/fail-closed-test-overlay.mp4 0 1 0.125 480 3
gif fail-closed-test-filtered real-clips/privacy-filtered/fail-closed-test-filtered.mp4 0 1 0.125 400 3
gif screen-002-raw-vs-smoothed smoothing/screen-002-raw-vs-smoothed.mp4 0 7 1 720 8
gif 2026-10-17-simulator-region-history earlier/2026-10-17-simulator-region-history.webm 0 16 2 640 6
gif 2026-10-17-simulator-tracking earlier/2026-10-17-simulator-tracking.mp4 0 17 2 480 6
for c in clip-dev-001 clip-dev-002 clip-dev-004; do gif 2026-10-19-$c-overlay earlier/2026-10-19-$c-overlay.mp4 0 11 1.5 400 6; done
gif 2026-10-20-clip-dev-004-zone earlier/2026-10-20-clip-dev-004-zone.mp4 0 9 1.5 400 6
gif 2026-10-20-screen-002-zone earlier/2026-10-20-screen-002-zone.mp4 0 10 1.5 400 6
gif 2026-10-20-screen-002-swerve-halfspeed earlier/2026-10-20-screen-002-swerve-halfspeed.mp4 0 6 1 400 8
gif 2026-10-21-app-run earlier/2026-10-21-app-run.mp4 20 30 3 640 6
montage 2026-10-24-demo-draft-1 earlier/2026-10-24-demo-draft-1.mp4 480 1.6 20 50 90 130
montage 2026-10-24-demo-draft-2 earlier/2026-10-24-demo-draft-2.mp4 480 1.6 20 50 90 130
gif 2026-10-24-screen-002-before-after-draft earlier/2026-10-24-screen-002-before-after-draft.mp4 0 7 1 720 8

aws s3 cp --region us-east-1 "$H" "$OUT/h264" --recursive --only-show-errors
aws s3 cp --region us-east-1 "$P" "$OUT/previews" --recursive --only-show-errors
echo "previews: $(ls "$P" | wc -l) gifs, $(du -sh "$P" | cut -f1); h264: $(find "$H" -type f | wc -l) files"
date -u +%FT%TZ | aws s3 cp --region us-east-1 - "$OUT/DONE" --only-show-errors
echo done
