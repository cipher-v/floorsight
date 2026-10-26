# Floorsight videos

**▶ Watch them all in your browser: https://tonystark101-debug.github.io/floorsight/** (GitHub cannot play videos
stored in a repository; that page can). Moving previews of every video are on the [main page](../README.md) and in
[`previews/`](previews/).

Every video below was made by this project: on AWS (Graviton4, COOL OpenCV build) unless noted. Real footage is from
Önal & Dandıl (2024), CC BY 4.0, trimmed, with heads and faces pixelated before anything was stored or shown. The
simulated warehouse is our own renderer; its people are not real.

## Start here

| Video | Length | What it shows |
|---|---|---|
| [`floorsight-demo.mp4`](floorsight-demo.mp4) | 4 min 22 s | The demo: the problem, the floor map in metres, danger zones, smoothing, the app (automatic slow/stop, the agent's second look, the approval step), the simulated warehouse with the interlock off and on, failure cases, COOL, AWS, privacy. Narrated, captions burned in; subtitles in [`floorsight-demo.srt`](floorsight-demo.srt). |

## Simulated warehouse: the forklift interlock in a closed loop (24 Oct)

Four rendered ceiling cameras feed the same vision code the app uses; a simulated forklift obeys Floorsight's
slow/stop commands. Results: `docs/receipts/interlock-sim-20261024.json`; report section 4c.

| Video | Length | What it shows |
|---|---|---|
| [`simulator/interlock-off.mp4`](simulator/interlock-off.mp4) | 40 s | Interlock off: Floorsight sees the people, nothing brakes, and the forklift reaches the people who step into its path. |
| [`simulator/interlock-on.mp4`](simulator/interlock-on.mp4) | 40 s | Interlock on, same scene: the forklift slows, stops, waits until each person is seen clear, then drives on. |
| [`app/app-simulator.webm`](app/app-simulator.webm) | 1 min 58 s | The Floorsight app itself running on the simulated warehouse's four cameras, issuing the same commands. |
| [`simulator/cameras-demo_001/`](simulator/cameras-demo_001/) | 3 min each | The four rendered camera feeds of the original simulated scenario (17 Oct). |

## Real factory clips (final run, 24 Oct)

| Video | Length | What it shows |
|---|---|---|
| [`app/app-real-clips.webm`](app/app-real-clips.webm) | 3 min 7 s | The app on AWS replaying the six real clips: floor map, AUTO-SLOW / STOP / RESUME banners (recorded video, so logged, not applied), the agent's one-line recommendation, approve/reject, alerts. |
| [`real-clips/*-overlay.mp4`](real-clips/) | 8-13 s each | Each clip with Floorsight's floor measurements, Kalman-smoothed tracks and forklift danger zones drawn on. |
| [`real-clips/privacy-filtered/`](real-clips/privacy-filtered/) | 8-13 s each | The same clips after the privacy filter only: what the rest of the system is allowed to see. |
| [`real-clips/fail-closed-test-overlay.mp4`](real-clips/fail-closed-test-overlay.mp4) | 12 frames | The fail-closed test: on frames 4 and 8 the face filter is forced to fail, and those frames come out black and marked withheld (the version without overlays is in `privacy-filtered/`). |
| [`smoothing/screen-002-raw-vs-smoothed.mp4`](smoothing/screen-002-raw-vs-smoothed.mp4) | 6 s | Same clip, same detections: raw outlines (left) against Kalman-filtered tracks (right). |

## Earlier versions (kept for the record)

[`earlier/`](earlier/) holds videos from 17-24 Oct, named by date: the first simulator runs, overlays and danger
zones from our earlier detector stack (since replaced by Apache-2.0 detectors), the 20 Oct swerve case at half speed,
the 21 Oct app run, and two drafts of the demo. **They are superseded:** some numbers in them were corrected later
(for example, the key clip screen-002 no longer warns once the turn-rate noise is filtered). The final numbers are in
the demo above and in `docs/report/Floorsight-Technical-Report-v0.5-20261024.md`.
