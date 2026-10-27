# Floorsight — technical report

> Historical methods and results, retained for traceability. Start with the [current concise report](../TECHNICAL_REPORT.md). Braking was applied only in simulation; real clips were processed offline. The recorded reviewer was scripted and the planner used the local policy. Current setup instructions are in the [development guide](../DEVELOPMENT.md).

**Version 0.5, 24 October 2026 (numbers from the 23-24 Oct AWS runs; receipts in `docs/receipts/`). New in 0.5: motion smoothing (section 4b), the forklift interlock that slows or stops the truck itself, tested in a closed loop in the simulated warehouse (section 4c), and the runs that use them.** OpenCV AI Competition 2026 (Devpost).
Status markers used throughout: **[real]** built and run on real factory footage; **[sim]** built and run on the
simulated warehouse only; **[planned]** not built. Every number below links to an evidence file in the repository.

## 1. Problem and users

Forklifts and people share the same floor in warehouses and factories, and most sites already have ceiling
CCTV. The cameras record, but nobody watches them in real time, and a camera image cannot say how far a
person really is from a forklift: on screen, a seated operator *behind* a forklift looks as if they are
touching it.

Floorsight turns an existing camera into a floor map in metres. For every moving forklift it draws where the
forklift will travel before it can stop, where its rear swings out in a turn, and the floor a tall load hides from
the driver, each with its uncertainty, and it warns when a forklift will pass within 0.8 m of a person in the next
3 seconds if it keeps its course. The moment a warning is confirmed it slows that forklift to walking pace, or stops
it if the person is in its path, before anyone is asked; the agent and a person review afterwards.

Users: site safety (EHS) managers and shift supervisors. They need few, explainable alarms, an honest statement
of what the system could not see, and no surveillance of individuals.

## 2. What Floorsight does (one figure)

FIGURE: the screen-002 moment, measured twice (`docs/report/figures/screen-002-measured-twice.jpg`; numbers in
`docs/receipts/screen-002-measured-twice-20261024.json`).
A real forklift comes down a factory lane at about 2 m/s past a seated machine operator.
- **20 Oct, earlier detector stack** (Ultralytics, since replaced; development run 17 on GCP, OpenCV 5.0.0):
  predicted pass 1.1–2.0 m (median 1.4 m), so no warning; the forklift brakes; in the last two frames its outline
  turns at 1.0 rad/s toward the operator and the gap drops to 0.67 m. Floorsight warned 0.04 s *after* the closest
  moment. Numbers: `docs/receipts/screen-002-measured-twice-20261024.json` (the 20 Oct run's own receipts are kept
  in the team's private evidence store).
- **24 Oct, publishable stack on AWS with motion smoothing** (Apache-2.0 detectors, COOL OpenCV 5.1; the run 3
  detections replayed through the Kalman-filtered tracking of section 4b, final run `perc-full-12`): the gap tracks the
  earlier run within about 0.3 m until 5.8 s. The closest the operator gets is 1.55 m (5-95 %: 1.21-1.90 m) at 5.86 s;
  the gap never falls below 0.8 m, and Floorsight does not warn.
- **What changed on the way.** The unsmoothed 23 Oct run 3 warned twice here (1.57 s and 2.97 s, about 2.2 s ahead of
  the predicted pass) and the 24 Oct app run confirmed one of those warnings. Both came from a "turn" of about
  0.2 rad/s toward the operator that lasted about 0.3 s each time, while the forklift's heading stayed at 270-272°:
  a 1-second line fit read outline noise as a turn and bent the predicted course to within 0.2 m of the operator. The
  turn-rate filter of section 4b shows no measurable turn (its error, 0.2-0.5 rad/s, is larger than the "turn"), so the
  course is straight and passes at about 2.6-3.0 m. Those warnings were false alarms; we withdraw them.
- Reading: the "swerve" lived in two frames of one detector's forklift outline. With one camera and no ground truth we
  cannot say which outline is right, so we report the moment as it is: at 6.18 s the two outlines put the gap at
  0.67 m and 1.88 m, about 1.2 m apart. A course-based warning cannot foresee a real swerve; that limit stands either
  way.
The old on-screen pixel rule alarmed for this operator from 5.4 s onward regardless of the floor; on the six clips it
alarmed on five.

## 3. Architecture

DIAGRAM: `docs/diagrams/floorsight-architecture-20261024.svg` (PNG alongside) (solid = [real], dashed = [sim], dotted = [planned]).

Real-footage pipeline [real], per frame (AWS Graviton4, COOL OpenCV 5.1):
1. **Video in.** Clips stay in a private quarantine (a private S3 bucket and the instance disk); SHA-256 checked
   against the publisher.
2. **Privacy filter first, fail-closed.** YuNet faces (`cv2.FaceDetectorYN`) plus a head region estimated from every
   person box are pixelated. Person boxes for privacy come from a deliberately low threshold (0.2), three image tiles,
   and a second detector (YOLOX-S on `cv2.dnn`) whose boxes are used for blurring only; head regions are grown 10 %
   and held 12 frames after a person is lost. The forklift cab is blurred. If the filter cannot run, the frame is
   written black and listed as withheld (forced-failure test passes on every run).
3. **Detect and track.** People and vehicles: RT-DETRv2 (COCO). A vehicle box counts as a forklift only when Grounding
   DINO, prompted with "forklift", confirms it; carried loads come from Grounding DINO (every 5th frame) and must lie
   inside the forklift's extent. Outlines: SlimSAM, prompted with each box. Non-maximum suppression with
   `cv2.dnn.NMSBoxes`; a Hungarian tracker.
4. **Floor map in metres** (once per camera). Line segments (`createLineSegmentDetector`) → three vanishing points
   (RANSAC) → focal length, camera rotation and a floor homography. Scale from two independent size references
   (standing adults; forklift with load) that must agree; their agreement picks the focal length. 300 Monte Carlo
   draws carry the error. Result: camera 3.56 m above the floor (±4.7%), tilt 14.2°, f = 2150 ± 100 px.
5. **Danger zones** (64 draws per frame). Each draw samples the calibration and the physical priors (section 4) and
   rasterises, on a 10 cm floor grid: stopping zone, tail swing, driver blind zone, 3 s warning zone. Each draw's
   raster is mapped onto one common floor with `cv2.warpPerspective`; outlines at 50% ("likely") and 5%
   ("possible") come from `cv2.findContours`; person clearance from `cv2.pointPolygonTest`.
6. **Decide.** DANGER (person in the stopping or tail-swing zone, or in the blind zone while the forklift drives
   load-first, or within 0.8 m), WARNING (will come within 0.8 m in 3 s if it keeps its course), each only for a
   moving forklift and only after 3 frames in a row; CHECKING is shown while a condition is met but not yet
   confirmed. The old pixel rule is still computed and logged for comparison only.
   **Act** (section 4c): a confirmed warning slows the forklift, a person in its path stops it, at once and without
   approval; the command goes to the truck's receiver (simulated here; logged as "not applied" for recorded video).
7. **Outputs.** Filtered overlay video, filtered clean video, floor-map inset, per-frame JSON, events, the list of
   pixelated regions, and a receipt recording `cv2.__version__` and the build line (5.1.0-dev, KleidiCV 0.7.0 on AWS).
8. **App.** The filtered results replay through the Floorsight server as a live feed: floor map, warnings, the
   automatic slow/stop banner, the agent's second look, and the human approval step (section 6), with evidence in S3, track history in DynamoDB, alerts on
   SNS, and metrics in CloudWatch (section 7).

Prototype on the simulated warehouse [sim] (16–17 Oct): multi-camera floor tracking with ArUco markers and
`solvePnP`, a coverage map with seen / estimated / out-of-view states, camera-health checks, a rules engine, and the
first version of the agent.

Planned [planned]: the 3D twin in the browser, a second camera on the same floor, and the agent's planner on Amazon
Bedrock (built and unit-tested; on 24 Oct no Claude model was open to the AWS account, so the local planner runs,
section 6).

## 4. OpenCV 5 implementation, component by component

On AWS every stage runs on the COOL build (`cv2.__version__ == "5.1.0-dev"`, KleidiCV 0.7.0 + carotene HAL,
`-mcpu=neoverse-v2`; full build line in `docs/receipts/cool-opencv-build-20261023.txt`); development runs before 23 Oct used
`opencv-python-headless==5.0.0.93`, which is also the "stock" side of the benchmark (section 10). The version and build
are recorded in every run receipt; the runner refuses any other major version. The module-by-module table (module, what is new
in OpenCV 5, status) is section 0 of `docs/Floorsight-Project-Overview.md`; in addition: `dilate` + `getStructuringElement` (warning zone), `ORB_create` + `findHomography` (same-camera
check: the 65 screened clips within 0.66 px, clip-dev-002/-003 within 0.13 px of the calibrated view), `minAreaRect` / `convexHull` (forklift footprint).

Physical priors (uniform ranges, written into every result file): reaction 0.75–1.5 s; braking 1.5–3.0 m/s²; driver
eye 1.85–2.15 m; eye to load face 1.2–1.8 m; truck 1.0–1.3 × 2.2–3.8 m; contact height 0–0.25 m; seated foot height
0–0.15 m; load face to drive axle 1.3–1.8 m. Speed: straight-line fit over the last second plus its standard error,
and the along-track acceleration only when it is at least twice its standard error; zones use the larger of the mean
and the extrapolated speed, so an estimated slowdown never shrinks a zone. Turn rate: from a turn-rate Kalman filter on
the floor track (section 4b), with its standard error sampled per draw; "turning" is only claimed when the error is under
half the turn, for 3 frames in a row.

## 4b. Motion smoothing (24 Oct): one-frame noise out, real motion kept

**Why.** The detectors redraw every outline from scratch each frame. Reviewing the demo video frame by frame and the
23 Oct per-frame records (`scripts/audit_motion_noise.py`) showed what that did to the floor tracks: a forklift's
footprint jumped 0.2-1.0 m between frames for 9 frames in 10 (the truck itself moves about 0.1 m per frame at
2.5 m/s), its size swung between 3.4 and 11 m², its heading swung by more than 30° in up to 19 frames per clip frames, "turning"
flipped on and off up to 21 times per clip times, the detector lost the forklift for up to 42 % of a clip (and the zone vanished
with it), people's floor points jumped 0.3-0.8 m, and one person seen as two overlapping boxes could raise two
warnings at once. None of this changed a confirmed warning, because a warning needs 3 frames in a row, but it made the
zones jumpy on screen and it is noise in every distance.

**What runs now** (on by default; `--no-smoothing` restores the 23 Oct behaviour for comparison):
1. **Kalman filter on every floor track** (`scripts/motion_filter.py`). A constant-velocity filter (position and velocity
   on the floor) for each forklift and each person, causal as a live system needs. Every Monte Carlo draw (each draw has
   its own camera calibration, hence its own floor scale) carries its own state with one shared covariance. A
   measurement more than 3.7 standard deviations from the prediction (χ², 2 dof, 99.9 %) is treated as a one-frame
   outlier and skipped. A forklift's outline can switch for a few frames between two readings of where the truck meets
   the floor (screen-032: about 1.2 m apart, probably the load being cut out or not), so a forklift track re-anchors to
   the new reading only after 6 outliers in a row (0.24 s; people: 4). A re-anchor moves the position and keeps the
   speed, and the track's recent positions move with it, so the shift is not read as speed. Forklifts: process noise
   3 m/s², measurement noise 0.35 m; people: 2 m/s², 0.25 m. The speed error the zones sample comes from the filter's
   own covariance.
2. **Steady footprint.** Each frame's outline points are first clipped to the outline's usual extent over the last
   second, then the v8 footprint is built (model rectangle through the near side, hull with the points seen), and its
   reach in 16 directions around the truck is replaced by the median of the last 0.4 s. One frame's flare is cut back;
   what the outline shows consistently (a wide load, a body turning slowly) stays inside.
3. **Steady heading, honest turns.** Below 0.6 m/s a 1-second fit of the travel direction is noise, so the footprint
   keeps its last heading; above it, the truck's body axis turns no faster than 1 rad/s, while the travel direction may
   reverse once it has read reversed for 5 frames in a row (a reversing truck keeps its footprint and its zone points the
   new way; a two-frame misreading does not flip it). A parked outline's long axis changes by more than 30° only after
   5 frames in a row. The turn rate comes from a second filter (`TurnKF`): an extended Kalman filter
   with a constant-turn-rate-and-velocity model (position, signed speed, heading, turn rate) on the raw outline
   centre, whose turn-rate error is honest because its measurements are independent frame to frame. "Turning" counts
   only when the turn is measured (error under half the turn) and switches on or off only after 3 frames in a row. On
   synthetic noisy straight runs the 1-second fit this replaces said "turning" in 73 % of frames (30 of 30 runs); the
   filter in 14 frames of 4,500 (1 run of 30). The price: a gentle turn is confirmed later (0.5 rad/s after about 1.6 s,
   0.8 rad/s after about 1.1 s), because with outline noise of about 0.35 m a turn cannot honestly be measured sooner.
   A person's "seated" call changes only after 5 frames in a row.
4. **Short gaps bridged.** A forklift the detector misses keeps its zone for up to 0.5 s on the filter's prediction
   (marked "coasting"); after that it is dropped, as before.
5. **People.** Association uses each track's recent image motion and, for boxes IoU cannot match, centre distance in box
   heights. Two boxes on one person (heavy overlap, same feet) count once: one floor point, one warning. A person unseen
   for more than 0.3 s starts a fresh filter when seen again (a seated worker lost for 0.8 s reappeared standing 2.8 m
   away; the filter read that as walking at 3.5 m/s and carried a ghost of them to 0.65 m from a forklift). A forklift's
   driver stays its driver for 1 s after last seen inside the truck's box (one frame's shrunken forklift box made a driver
   a "pedestrian" 0.90 m from their own truck).
   A truck that has stopped (0.4 s below 0.6 m/s) takes its next travel direction at once; while it moves, a reversal
   needs 5 frames in a row and the turn-rate filter's agreement.
6. **Parked objects.** An object never seen moving is shown only if its typical confidence (median of the last 25
   detections) is at least 0.5: the press machine (median 0.33) and the stack of steel plates (0.35-0.39) no longer
   appear as "forklifts"; the real forklifts have medians 0.58-0.71.

**What it did** (same cached detections, 23 Oct raw vs 24 Oct smoothed; `docs/receipts/motion-noise-before-after-20261024.json`):

| Clip | Footprint jump, 9 in 10 frames (m) | Worst jump (m) | Heading swings > 30° | Turning on/off | Forklift on the floor (frames) | Worst one-frame speed change (m/s) | Person jump, 9 in 10 (m) | One-frame warning blips |
|---|---|---|---|---|---|---|---|---|
| screen-002 | 0.43 → 0.12 | 2.29 → 0.51 | 17 → 2 | 17 → 0 | 207 → 196 of 228 | 2.57 → 2.77 | 0.29 → 0.09 | 2 → 0 |
| clip-dev-004 | 0.38 → 0.10 | 0.88 → 0.49 | 6 → 1 | 19 → 0 | 188 → 199 of 201 | 1.30 → 1.48 | 0.84 → 0.42 | 2 → 0 |
| screen-032 | 0.96 → 0.23 | 2.10 → 2.37 | 3 → 2 | 18 → 0 | 147 → 195 of 302 | 1.44 → 1.31 | 0.28 → 0.05 | 1 → 0 |
| clip-dev-002 | 0.17 → 0.09 | 0.74 → 0.22 | 4 → 2 | 13 → 2 | 213 → 214 of 251 | 0.92 → 0.92 | 0.31 → 0.08 | 1 → 0 |
| screen-033 | 0.04 → 0.02 | 0.14 → 0.05 | 4 → 0 | 6 → 0 | 212 → 202 of 225 | 1.13 → 0.67 | 0.44 → 0.68 | 0 → 0 |
| clip-dev-003 | 0.36 → 0.07 | 5.58 → 0.34 | 19 → 4 | 21 → 0 | 260 → 314 of 325 | 1.98 → 1.46 | 0.26 → 0.05 | 2 → 0 |

Totals over the six clips: heading swings over 30° 53 → 11, "turning" switched on or off 94 → 2 times, one-frame warning blips 8 → 0, a person's warning switching on or off 33 → 6 times; objects shown as forklifts that are not (a press machine, a stack of plates) 4 → 0. "Turning on/off" counts the turning decision (24 Oct: the recorded flag; 23 Oct: most draws turning). Person jumps count only frames near a moving forklift; on screen-033 that is about 20 frames, so its 9-in-10 figure is the second-largest of 20 (the same two jumps, 0.68 and 0.76 m, in both runs).

Floor results on the same detections (`docs/receipts/floor-compare-raw-vs-smoothed-20261024.json`):

| Clip | Closest moving pass, 20 Oct / 23 Oct raw / 24 Oct smoothed (m) | Confirmed warnings, 20 Oct / raw / smoothed |
|---|---|---|
| screen-002 | 0.67 / 1.43 / 1.55 | 1 / 2 / 0 |
| clip-dev-004 | 1.80 / 2.38 / 2.83 | 0 / 0 / 0 |
| screen-032 | 1.37 / 1.55 / 1.79 | 0 / 1 / 0 |
| clip-dev-002 | 2.03 / 2.02 / 2.27 | 1 / 2 / 2 |
| screen-033 | 0.53 / 1.39 / 1.71 | 1 / 0 / 0 |
| clip-dev-003 | 1.45 / 1.77 / 2.44 | 0 / 0 / 1 |

What moved, and why: screen-002 loses both warnings (section 2: they came from a turn-rate artefact). screen-032 loses its warning, which came from the same kind of brief "turn" (the 24 Oct app run's agent had already rejected it). clip-dev-003 gains one: in its last half second the forklift pulls away from a stop toward a person 4.6 m ahead; the unsmoothed run measured that direction too, the intermediate smoothed runs kept the stopped truck's old heading (a bug we fixed: a stopped truck takes its new direction at once). The clip ends 0.12 s later. clip-dev-002 keeps two warnings, both on the forklift's course after it reverses out and drives off (the driver then turns; passes 2.6 and 4.8 m). Closest passes grow by 0.1-0.7 m: the steadier footprint no longer flares toward people for single frames.

**Cost and honesty.** The filter trades a little lag for the noise: a sudden change of the outline is followed over about
0.1-0.2 s instead of at once. A two-frame outline swerve like the 20 Oct one would now be smoothed out; we think that is
right for an outline artefact, and it is also why no warning should rely on two frames. The detector run is identical
(the detections were saved in a live run and replayed, `--save-detections` / `--load-detections`; that live run's
screen-002 output is byte-identical to run 3, and a 12-frame pilot replay is byte-identical to its live pilot), so every
difference in the tables comes from the tracking and floor logic alone.

**What still moves.** screen-032's forklift outline switches for good, at 4.2 s, to a reading about 2.1 m further along
the lane; the track re-anchors once, after 6 frames (a 2.4 m jump), instead of 5.8 m/s of phantom speed. After a stop a
forklift's new direction is taken when the 1-second fit reads 0.6 m/s, typically 0.2-0.5 s after it moves off; the zone
of a slow truck is small in that time. The turn-rate filter confirms a gentle turn later than the old fit "detected" one.

**How we found these.** Each pass replayed the saved detections (25 minutes on the instance), looked at the per-frame
records of every clip for the biggest jumps and swings, and traced each one back to a cause. The fixes are the ones
described above; three of the issues were introduced by our own earlier fixes (a heading seeded from a parked outline's
axis, a reversal accepted from two misread frames, and a wait for both direction estimates that kept a zone pointing
backwards for 0.7 s) and were caught the same way. Every fix has a test (`tests/realclip/test_motion_smoothing.py`, 222 tests pass on the instance).

## 4c. Forklift interlock: the machine acts first (24 Oct)

**Why.** A warning comes 1-3 s before the close pass; the agent's second look takes seconds and a person's approval
5-30 s, so an approval cannot stop that forklift. The order is now: the machine acts the moment a warning is confirmed,
and the agent and a person review afterwards (`floorsight/interlock.py`, run inside the app before the agent).

**What it does.**
- Warning (it will come within 0.8 m of a person in the next 3 s on its course): **slow** to walking pace (0.7 m/s).
  Slow comes first because a hard stop with a raised load can tip a truck or drop the load.
- The person is in the truck's path (closer than 0.25 m within 1.5 s), or already within 0.8 m: **stop**.
- A person walking into a truck whose own motion is not closing the gap (into its back as it drives away): **horn**;
  braking would not help.
- **Resume** after 2 s with no condition; but a truck stopped for a person resumes only if the last time it saw that
  person they were standing clear of its path, and a truck we stopped is taken to stand where it stopped.
- States belong to the physical truck (a track that reappears within 2 m of a lost one is the same truck), not to the
  tracker's label.
- Every command is logged with its reason and what a stop does: the stopping distance at the measured speed (0.3 s
  from frame to brakes, 1.5 m/s² worst-case braking) against the floor gap ("a stop now halts it 2.2 m short").
The command goes to the truck's receiver. On a real truck that is a unit wired to its speed-limit input, hardware this
project does not include; recorded video cannot be braked, so on the real clips the command is logged and shown as
"not applied".

**Real clips** (final app run on AWS, app-run-09, code r55; receipt `docs/receipts/app-run-09-summary-20261024.json`): on clip-dev-002 the forklift's course pointed at a person 4.2 m away: Floorsight sent SLOW, then STOP 0.4 s later ("a stop now halts it 2.2 m short"), and RESUME when the course cleared 2.4 s later; on clip-dev-003 it sent SLOW as the forklift pulled away from a stop toward a person 4.6 m ahead ("a stop now halts it 4.1 m short"). Then the agent reviewed all three warnings and recommended confirm, the scripted stand-in approved, and three alerts went out on SNS (29 trace steps).

**Closed loop in the simulated warehouse.** The simulator from 16-17 Oct now has a forklift that obeys the interlock:
four rendered ceiling cameras -> the same vision pipeline the app uses on that site (ArUco-calibrated cameras,
background subtraction, multi-camera tracking) -> the interlock -> the simulated brakes (0.2 s actuation, 2.5 m/s²) ->
the next frame. Four scripted situations, with people who do not react, run with the interlock off and on
(`scripts/interlock_sim.py`; receipt `docs/receipts/interlock-sim-20261024.json`; run interlock-sim-08, code r55):

| Situation | Interlock OFF | Interlock ON |
|---|---|---|
| S1: a picker steps into the lower lane ahead of the forklift and stays 6 s | contact | stopped, waited until the picker was seen clear, drove on and slowed to pass; closest 0.70 m |
| S2: a person crosses the main aisle ahead of the forklift | contact | stopped, then passed; closest 0.65 m |
| S3: a person walks alongside, 1.15 m from the forklift's side | no conflict | no command for this person (one needless slow-down earlier, at 2.2 s, came from the forklift's first, jittery velocity estimate) |
| S4: a person walks into the back of the forklift as it drives away | contact | contact: braking cannot help; the horn should sound, but the vision picked this person up too late |

Cost: in 40 s the forklift drove 26.8 m instead of 79.8 m (stopped 18.1 s, slowed 9.9 s).
Vision plus interlock: 13.5 ms per frame (median, 4 cameras). The same scenario also runs inside the app
(`--source simlive`, recorded as app-run-10, which issued the same commands at the same moments).

**What the closed loop taught us** (each fixed, each now a test in `tests/test_interlock.py`): a stopped truck "saw"
nothing in front of it, resumed and crept closer; a person waiting where the truck's straight-line course pointed made
a deadlock; the stopped truck faded from the background model and came back under a new track id whose "resume"
released the brakes; a picker standing still faded too, and the truck drove on into the picker it could no longer see;
a person who had crossed and then stood out of view kept the truck waiting 35 s; drifting tracks of standing objects
read as people walking away. One result is a limit, not a bug: in the first layout the picker stepped into the middle
of the rack aisle, where only one camera looks down the aisle and the approaching forklift blocks its view. Floorsight
never saw the picker, so nothing could brake. A camera system can only stop for what a camera sees; the coverage map
shows where that is.

**Limits.** The simulated people do not react to the horn; the truck's receiver is simulated; the real clips cannot
show a stop; live use on real cameras needs the detectors in real time (about 4.5 s per frame on this CPU instance
today), which needs a GPU or lighter detectors.

## 5. AI models and data

| Item | Use | Licence (read from the source, 23 Oct) |
|---|---|---|
| Factory CCTV dataset, Önal & Dandıl 2024, Mendeley Data V1, doi:10.17632/xjmtb22pff.1 | 6 clips analysed in full, 65 screened | CC BY 4.0. The dataset paper (Data in Brief 56:110791) states every filmed employee signed an informed consent form; ethics approval Bilecik Şeyh Edebali University, 11/06/2021. Factory permission wording: "in academic studies" (owner question open). |
| RT-DETRv2 R50 (COCO), `PekingU/rtdetr_v2_r50vd` | people and vehicle boxes, every frame | Apache-2.0 |
| Grounding DINO tiny, `IDEA-Research/grounding-dino-tiny` | confirms a vehicle box is a forklift; carried loads (every 5th frame) | Apache-2.0 |
| SlimSAM-77, `Zigeng/SlimSAM-uniform-77` | box-prompted forklift and load outlines | Apache-2.0 |
| YuNet `face_detection_yunet_2023mar` (OpenCV Zoo) | faces for the privacy filter (`cv2.FaceDetectorYN`) | MIT |
| YOLOX-S/L/M (Megvii) via `cv2.dnn` | YOLOX-S: extra person boxes, used only for pixelation; YOLOX-L: second look on the crop | Apache-2.0 |
| Simulated warehouse (own) | prototype scoring | ours |

**Detector swap (23 Oct).** Until 22 Oct the perception stage used Ultralytics YOLO11m and YOLOE-11m-seg (AGPL-3.0).
Before publishing we replaced them, choosing by measurement on AWS: 60 real frames from 5 clips, compared with the
accepted detections (share of accepted people / forklifts found at IoU 0.5, time per frame on c8g.xlarge):
RT-DETRv2 97 % / 98 % at the probe's lowest threshold (0.1; at the thresholds the pipeline runs, 0.2 for people and
0.3 for vehicles, a single untiled pass finds 93 % / 89 %), 0.95 s; YOLOX-M on 3 tiles through `cv2.dnn` 87 % / 98 %, 2.8 s; YOLOX-L tiled 90 % / 86 %,
6.4 s; Grounding DINO 96 % / 100 %, 6.0 s; OWLv2 85 % / 91 %, 8.0 s; OmDet-Turbo would not load (transformers 4.57.3).
Forklift outlines from the accepted boxes: SlimSAM IoU 0.925 with the old masks (1.4 s), SAM ViT-B 0.924 (6.4 s).
Receipts: `docs/receipts/detector-probe-03-20261023.json`, `docs/receipts/detector-probe-04-threshold-sweep-20261023.json` (threshold sweep). Privacy cross-check (run 3, five clips with a
reference run): the new stack covers 98.0 % of the 2,636 heads the old stack saw (100 % on screen-002, 96.9 % on the
crowded screen-033). Every flagged frame type was then looked at in the filtered video: almost all were pixelated heads
that the old box placed lower; one real gap remains, the top of one head (about 20 px, no face visible) in two frames of
clip-dev-002. Receipts: `docs/receipts/privacy-crosscheck-aws-run3-20261023.json`,
`docs/receipts/privacy-manual-review-20261024.md`. The target was 99 %; we report the 98 %. Ultralytics is not installed on AWS;
the old models remain only as an explicitly named legacy option for reproducing runs before 23 Oct.
No model was trained or fine-tuned. No ground-truth labels are used at run time.

## 6. The agent

[real] Every floor WARNING or DANGER event goes to the agent before anyone is asked; by then the interlock (section 4c)
has already slowed or stopped the forklift, so the agent and the person decide what the warning was, not whether to
brake. Its tools are OpenCV steps on
the private side of the privacy filter:
1. **Short replay**: the last 1.5 s of floor measurements for that person and forklift (gap, predicted pass, turn).
2. **Sharper crop**: a close-up of the pair from the unblurred frame (kept private; only the pixelated copy is stored).
3. **Re-detect**: a second detector pass on the crop, checking that the person and the forklift outline are real and
   that the gap holds.
It then recommends confirm, reject or uncertain in one line with three numbers (floor gap at the warning, share of the
replay within 0.8 m, closest predicted pass with its 5–95 % range). The recommendation goes to a human; **no alert reaches
the floor lead without a human's approval**, and the alert then goes out on SNS. Every step is written to a trace file
as it happens, and the planner's choice of next tool is checked against a fixed local policy.
Results: in-app run on the real clips, 20 Oct (earlier detectors, GCP): 3 warnings; the agent recommended 1 confirm and
2 rejects (a 3.5 m pass and a tracking glitch); the stand-in human agreed on all 3; 23 steps. On AWS with the
publishable stack before smoothing (24 Oct morning, replaying run 3; `docs/receipts/app-run-02-summary-20261024.json`):
4 warnings; the agent recommended confirm for 3 and reject for 1 (the forklift's tracked speed jumped from 1.0 to
2.1 m/s in one step, a tracking glitch, on screen-032). One of the three confirmations was on screen-002, where the
warning later turned out to come from a turn-rate artefact (section 2): the agent checks for tracking glitches, not for
a turn that was never measurable; the turn-rate filter now removes that case upstream (the screen-032 warning also goes).
Final app run with smoothing (24 Oct, replaying `perc-full-13`, code r46; `docs/receipts/app-run-05-summary-20261024.json`):
3 warnings on 2 clips; the agent recommended confirm for all 3, the stand-in followed, 3 alerts went out on SNS, 29 trace
steps, 9 blurred evidence crops mirrored to S3. Two are on clip-dev-002, on the forklift's course as it drives off after
reversing (the driver then turns; real passes 2.6 and 4.8 m); the third is on clip-dev-003, as the forklift pulls away
from a stop toward a person 4.6 m ahead, 0.12 s before the clip ends. That third one exposed one more bug: the agent read
the forklift's 164° change of direction after the stop as a tracking glitch and recommended reject (app run 4). The
floor stage now marks a stopped forklift taking its new direction, and the agent reads that as real motion ("pulled away
in a new direction after a stop"); a test covers it.
A confirmed warning is a prediction on the forklift's course, not a measured near miss: in all three the forklift turned
or the clip ended before a close pass. Decisions in every run came from a scripted stand-in that follows the agent; the
owner's own verdicts are still open.
Planner: Claude on Amazon Bedrock is built (strict tool definitions, every choice vetted by the local policy), but on
24 Oct the account could not call any Claude model (Opus 5.5, Opus 5 and Sonnet 5 refused; Sonnet 5.5 and Haiku 5.5
not found), so the local rule-based planner ran; every step records which planner chose it.
[sim] On the simulated warehouse (16–17 Oct): 3 of 3 near-misses confirmed, 0 false confirmed incidents, a glare false
alarm rejected (7 of 7 scenario checks).

## 7. Cloud deployment

Everything below runs in one private AWS deployment in us-east-1, created with Terraform (`infra/terraform`,
`docs/aws/DEPLOY.md`) on 23 Oct; every resource is logged with its time and purpose.

| Piece | What it does | Status |
|---|---|---|
| EC2 `c8g.xlarge` (Graviton4) from the COOL Marketplace AMI | perception batch, app server, agent, dashboard; OpenCV 5.1.0-dev + KleidiCV 0.7.0 | [real] |
| S3 artifacts bucket (versioned, never expires) | code bundles, the 6 clips (private), job receipts | [real] |
| S3 evidence bucket | blurred evidence crops, mirrored from the app under the same ids | [real]: 12 crops in the 24 Oct app run, 0 errors |
| DynamoDB `anisha-robotics-tracks` | one floor snapshot per second | [real]: 63 snapshots in the 24 Oct app run |
| SNS `anisha-robotics-alerts` | the floor-lead alert, published only after a human approves | [real]: 3 alerts in the 24 Oct app run |
| CloudWatch | metrics (fps, latency, events), logs, dashboard | [real]: CameraFps, PipelineLatency, Events, Tracks |
| Amazon Bedrock, Claude | the agent's planner (every choice vetted by the local policy) | built; no Claude model open to the account on 24 Oct, so the local planner ran |
| Amazon Polly | narration of the demo video | [real] (neural voice, built on the instance) |
| AWS Budgets | $50/month on the Project tag, alerts at 50 / 80 / 100 % | [real] |
| CloudFront + Elastic IP | HTTPS judge link | off: `terraform apply -var public_endpoint=true` after the owner's yes |

Security: no inbound port while private; access by SSM Session Manager only (no SSH); IMDSv2; encrypted disk and
buckets; TLS-only bucket policies; least-privilege instance role; in public mode the server refuses any request without
CloudFront's secret header. Owner hold: every automatic deletion (S3 expiry, DynamoDB TTL, log expiry) is off.
Cost of the 23-24 Oct runs, smoothing and interlock passes included: about $6.50 (EC2 about 34 h, $5.45; disk and
public IPv4 about $0.50; S3, Polly, CloudWatch, DynamoDB and SNS under $0.55), against a $50 budget (c8g.xlarge $0.1595/h; COOL $0.01/h after its
7-day trial; Bedrock $0, never reached). The instance is stopped between work sessions.

## 8. Privacy and responsible use

- The privacy filter runs before anything else and fails closed (black, withheld frames), tested on every run.
- Raw video never leaves the private quarantine; only filtered frames reach any output, figure or review page.
- Floorsight measures distances between people and vehicles; it does not identify people. Track numbers are
  anonymous and per clip.
- Alarms are advisory. The system says what it could not see ("cut by frame edge: floor outline uncertain", "turn rate
  uncertain", "person not seen this frame") instead of guessing.
- Not a certified safety system; no claim of accident prevention is made from development clips.

## 9. Evaluation

1. **Floor calibration** (self-consistency, no surveyed distances): two size references agree at one focal length;
   seven people's heights come out 1.55–1.82 m with no drift across the image (20 Oct floor-map retest; private working note, numbers quoted here).
2. **Real clips, alarms** (6 clips, 4 new; development clips without distance ground truth): the pixel rule alarmed on
   5; the floor rule confirmed none. Closest real pass 0.67 m for two frames (screen-002, section 2).
3. **Alarm control on real motion** (virtual people on three recorded forklift tracks): path person alarmed before
   contact on all three, 3 m person never alarmed. The 3 s warning came earlier than the alarm on all three: 2.5 s vs 1.8 s (clip 002), 1.9 s vs 1.1 s (clip 004), 0.7 s vs 0.5 s (clip 032, curved path; the 3 m person got a warning there, closest 1.4 m).
4. **Simulated warehouse** (scored against simulator truth): recall 86.9% (target ≥ 90%, not met), median position
   error 0.084 m, 0.114 false dots per update, 7 of 7 scenario checks.
5. **Detector swap, floor results** (6 clips, 20 Oct earlier stack vs 23 Oct run 3; `docs/receipts/floor-compare-aws-run3-vs-20oct.json`):
   closest moving pass old / new: clip-dev-002 2.03 / 2.02 m, clip-dev-003 1.45 / 1.77 m, clip-dev-004 1.80 / 2.39 m,
   screen-032 1.37 / 1.55 m, screen-002 0.67 / 1.46 m (section 2), screen-033 0.53 / 1.39 m. Warnings: 3 / 5. On
   screen-033 the old stack warned (and the agent rejected that warning on 20 Oct); the new stack sees the same person at
   about 1.4 m while it judges the forklift to be standing, so it does not warn. Like screen-002, the result hinges on
   the forklift's outline and motion from one camera.
6. **Forklift interlock, closed loop** (24 Oct, simulated warehouse, section 4c): both people who stepped into the
   forklift's path were hit without the interlock and passed without contact with it (closest 0.70 and 0.65 m); a person walking into the back of a forklift that drives away could not be helped; the forklift lost 53 m
   of its 80 m in 40 s to stopping and slowing. `docs/receipts/interlock-sim-20261024.json`.
7. **Motion smoothing** (24 Oct, same saved detections, section 4b): outline jitter down 50-80 % per clip, sudden
   heading swings 53 → 11, "turning" flips 94 → 2, one-frame warning blips 8 → 0; confirmed warnings 5 → 3 (two
   turn-rate artefacts withdrawn, one new warning at the end of clip-dev-003). Receipts:
   `docs/receipts/motion-noise-before-after-20261024.json`, `docs/receipts/floor-compare-raw-vs-smoothed-20261024.json`.
8. **Failure cases shown, not hidden**: swerve 0.04 s before the closest moment (no course-based warning can see it);
   forklift running off the bottom of the picture; start-of-track speed with a wide error; single-frame warning spikes
   far from the person (filtered by the 3-frame rule); a bin attached to a forklift as its load; a 0.3 s warning in clip-dev-002 when the forklift's course pointed at a seated operator 3.5 m away and the driver then turned (closest 2.0 m); an intermediate version that read a braking artefact as 'not moving' (fixed: a slowdown estimate never shrinks a zone).

## 10. COOL (Graviton) benchmark

COOL = the Cloud Optimized OpenCV AMI for Graviton4 (OpenCV 5.1.0-dev, KleidiCV 0.7.0 + carotene HAL,
`-mcpu=neoverse-v2`, FFmpeg on, DNN MLAS). Stock = `opencv-python-headless` 5.0.0.93 on the same instance, same
code, same real frames (4 real clips for the operations; 30 frames of screen-002 for the perception stage, run
interleaved COOL / stock / stock / COOL). Receipts: `docs/receipts/cool-bench-20261024.json`, chart
`docs/report/figures/cool-vs-stock.jpg`.

| Measure (median latency, 1080p frames) | COOL / stock speed-up |
|---|---|
| `adaptiveThreshold` | 7.06x |
| ArUco detection | 1.22x |
| `GaussianBlur` | 1.13x |
| `findContours` | 1.05x |
| `cvtColor`, `morphologyEx` | 1.00x, 0.99x |
| `warpPerspective`, JPEG decode, MOG2 | 0.96x, 0.93x, 0.91x |
| `resize` | 0.75x |
| OpenCV core chain, 1 stream | 1.10x (316 vs 287 fps) |
| OpenCV chain, 4 processes | 0.90x (236 vs 262 fps) |
| Whole perception stage (detectors included) | 0.97x (0.234 vs 0.240 fps) |

Reading: COOL helps some OpenCV kernels a lot (adaptive threshold) and a single stream of our OpenCV chain by 10 %, but
it is slower on `resize` and with 4 parallel processes, and the whole perception stage, which spends almost all its
time in PyTorch detectors, is 3 % slower. For Floorsight the COOL build's value on this instance is the OpenCV 5.1
feature set and the ARM-tuned build, not end-to-end speed; moving the detectors to `cv2.dnn` is the next step to test.

**Defect found and reproduced.** COOL's `cv2.fillPoly` crashes the process (SIGSEGV, inside
`libopencv_imgproc.so.501`) when polygon vertices lie a few rows below the image: a 5-point polygon on a 251 x 321
uint8 mask with its lowest vertices 4 rows below the last row; stock 5.0.0 fills it (254 px). In Floorsight it was
triggered by forklift poses that ran off the danger-zone raster. Floorsight now clips every polygon to the image first
(`floorsight/cvsafe.py`, Sutherland-Hodgman; unit tests), which gives the same pixels on both builds. Separately, a
polygon point at INT_MIN (NaN cast to int32) raises "cannot create std::vector larger than max_size()" on COOL and
fills on stock. Write-up and minimal reproduction: `docs/findings/COOL-fillPoly-crash.md`.

## 11. Limitations and future work

One camera; scale from assumed sizes (next: check against the dataset's published floor plan, and printed floor markers
in our own footage); development clips without ground truth; the floor gap at a given instant depends on the detector's
forklift outline (screen-002: about 1.2 m between two stacks at one instant); main detectors still in PyTorch (only YOLOX and YuNet
run on `cv2.dnn`); perception runs at about 4.1-4.6 s per frame on a 4-vCPU Graviton4, so the app replays
pre-processed clips rather than a live camera (next: a GPU or a lighter `cv2.dnn` detector for live use); the warning
cannot foresee a swerve (next: a wider caution margin and the driver's braking as a cue); the agent's planner ran
locally, not on Bedrock; privacy coverage 98 %, not the 99 % we aimed for; decisions in the recorded runs came from a
scripted stand-in, not a site's safety lead. Motion smoothing (section 4b) trades a little lag for steadiness: a gentle
turn is confirmed after about 1.1-1.6 s, a stopped truck's new direction 0.2-0.5 s after it moves off, and a lasting
change in the detector's outline is followed by one re-anchoring jump (2.4 m once on screen-032). The interlock
(section 4c) is proven only in the simulator, whose people do not react; a real truck needs a receiver wired to its
speed control, and real cameras need the detectors in real time; it can only stop for what a camera sees, and it costs
working time (in the simulator, two thirds of the forklift's distance).

## 12. Reproducibility

Every result folder holds its inputs' SHA-256 (clips checked against the publisher), the calibration used, the
per-frame JSON, the privacy cross-check and a receipt with `cv2.__version__`, the OpenCV build line (HAL), the library
versions and per-stage timings. AWS jobs are shell scripts in `infra/aws/jobs/` run over SSM; each job folder (log,
receipts, outputs) is copied to the artifacts bucket. Code bundles are immutable and numbered (r1-r24 on 23 Oct, r25-r55 on 24 Oct; r46 is the code of the perception results (perc-full-13), r55 of the app runs and the
interlock results; r38-r45 and r47-r54 are the intermediate passes, kept). Smoothing results replay saved detector output (`--save-detections` / `--load-detections`), so they can be re-run in minutes without the detectors.
Infrastructure: `infra/terraform` (plan/apply logs kept). Scripts: `scripts/real_clip_perception.py`,
`scripts/open_detectors.py`, `scripts/danger_zone.py`, `scripts/cvsafe.py`, `scripts/floor_calibration.py`,
`scripts/floor_retest.py`, `scripts/demo_moment.py`, `scripts/export_app_replay.py`, `scripts/second_look.py`,
`scripts/privacy_crosscheck.py`, `scripts/detector_probe.py`, `scripts/compare_runs.py`, `scripts/measured_twice.py`,
`scripts/motion_filter.py`, `scripts/audit_motion_noise.py`, `scripts/smoothing_report.py`, `floorsight/interlock.py`,
`floorsight/sim/live.py`, `scripts/interlock_sim.py`,
`scripts/benchmark/cool_bench.py`, `scripts/benchmark/bench_chart.py`, `scripts/video/build_video.py`.

**Addendum, 27 Oct: independent code review.** An outside reviewer reported 8 bugs and 6 risks; we found all of them
real and fixed them in code r56 (full suite on AWS with the COOL build: 275 passed, 2 skipped, 0 failed). The list is in
the [repository audit](../AUDIT.md#upstream-review-incorporated). One fix touches a result in this report. The interlock's resume rule
(section 4c) now also holds for the real-camera floor stage: a stop waits until the person it was for is seen out of
the truck's path. Before, the stop could end 2 s after the floor warnings stopped, and they stop once the truck
stands. Replaying the recorded floor data (`docs/receipts/interlock-replay-check-20261027.txt`) gives the same slow
(37.6 s) and stop (38.0 s) on clip-dev-002. The resume at 40.4 s no longer happens, because the person was still in
the path the truck had been driving on. The app recording was made before the fix. The simulated-warehouse results
(section 9) are unchanged.

## 13. Credits and licences

Footage: Önal, O. & Dandıl, E. (2024), "Video Dataset for Safe and Unsafe Behaviours", Mendeley Data, V1,
https://doi.org/10.17632/xjmtb22pff.1, CC BY 4.0. Changes: clips trimmed; heads and faces pixelated. No endorsement
implied. Models and libraries: `docs/publish/THIRD_PARTY_LICENSES.md` (Apache-2.0, MIT, BSD; no AGPL dependency).
OpenCV (Apache-2.0); COOL AMI under its AWS Marketplace terms (not redistributed).
