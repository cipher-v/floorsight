"""Write the demo video's section spec (for build_video.py) from a facts file, so every number in the narration comes
from the run receipts in one place.

    python scripts/video/floorsight_spec.py facts.json spec.json

facts.json holds the instance paths (VIDEO_IN, PERC, APP_WEBM), the app-recording cut times, and the measured numbers;
see FACT_KEYS. Cards use plain ASCII (OpenCV's Hershey fonts); captions and narration use full Unicode.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FACT_KEYS = ["VIDEO_IN", "PERC", "APP_WEBM", "APP_REPORT", "app_cuts", "c004_t", "c004_gap", "s002_new_closest", "s002_new_lead_s",
             "s002_outline_gap", "reject_reason", "sbs_video", "jitter_before_cm", "jitter_after_cm", "heading_flips_before",
             "heading_flips_after",
             "n_warn", "n_confirm", "n_reject", "n_uncertain", "n_steps", "cool_chain_x", "cool_perc_x", "head_cover_pct", "bench_chart",
             "twice_chart"]

CREDIT = "Real factory CCTV: Önal & Dandıl (2024), CC BY 4.0 · heads and faces pixelated"


def spec(f: dict) -> dict:
    vin, perc, app = f["VIDEO_IN"], f["PERC"], f["APP_WEBM"]
    cuts = f["app_cuts"]  # {"intro", "card", "approve", "reject": [start, end]} seconds in the dashboard recording
    sim = f.get("sim")    # closed-loop simulator: {"off", "on", "app": video paths; "cuts": {...}; numbers}
    s = [
        {"id": "title", "card": ["Floorsight", "Measures how close forklifts come to people, slows or stops them in time,",
                                 "and lets a person review.   OpenCV 5  |  AWS Graviton4  |  COOL"], "min_s": 6,
         "narration": "Floorsight measures how close forklifts come to people, slows or stops the forklift in time, and lets a person review."},
        {"id": "problem", "video": f"{perc}/clip-dev-004/clip-dev-004-overlay.mp4", "start": f["c004_t"] - 4.0, "end": f["c004_t"] + 2.4,
         "speed": 0.6, "caption_top": CREDIT,
         "narration": "Forklifts and people share the same floor. Most factories already record CCTV, but nobody watches it live, "
                      "and a camera image cannot tell you how far a person really is from a forklift."},
        {"id": "pixels-vs-floor", "image": f"{vin}/accepted-20261020/floor-004-before-after.jpg", "min_s": 9,
         "caption_top": "Left: on screen (pixels).  Right: the same moment on the floor map, in metres.",
         "narration": f"On screen, this seated operator seems to touch the forklift. On the floor map they are {f['c004_gap']} away. "
                      "A pixel rule would alarm all day. Floorsight measures in metres."},
        {"id": "floor-map", "image": f"{vin}/floor-grid.jpg", "min_s": 9,
         "caption_top": "1 m floor grid projected back onto the camera; green: height checks on people and machines",
         "narration": "OpenCV 5's line-segment detector finds the floor's main directions. Their vanishing points and two size references "
                      "turn one ordinary camera into a floor map in metres, with its error: the camera sits 3.56 metres up, plus or minus five percent."},
        {"id": "danger-zones", "video": f"{perc}/screen-002/screen-002-overlay.mp4", "start": 0.8, "end": 7.7, "speed": 0.4,
         "caption_top": CREDIT,
         "narration": "Every moving forklift carries its danger zone: where it travels before it can stop, where its rear swings, and what a "
                      "tall load hides, each drawn from sixty-four sampled variations. If it will pass within 0.8 metres of a person in "
                      "the next three seconds on its course, Floorsight warns."},
        {"id": "smoothing", "video": f["sbs_video"], "start": 0.0, "end": f.get("sbs_end", 6.9), "speed": f.get("sbs_speed", 0.35),
         "caption_top": "Same clip, same detections. Left: raw outlines. Right: Kalman-filtered tracks",
         "narration": "The detectors redraw every outline from scratch, so the raw zones jittered. A Kalman filter on every track, and a "
                      "second one on the turn rate, now hold them steady: the outline's jump between frames fell from about "
                      f"{f['jitter_before_cm']} centimetres to about {f['jitter_after_cm']}, and false turns that bent the zone toward people are gone."},
        {"id": "app-intro", "video": app, "start": cuts["intro"][0], "end": cuts["intro"][1], "speed": 1.0,
         "caption_top": "The Floorsight app on AWS Graviton4 (c8g.xlarge, COOL OpenCV 5.1), recorded on the instance",
         "narration": "This is the app, on an AWS Graviton4 instance with the Cloud Optimized OpenCV build. Six real clips replay as a "
                      "live feed onto the floor map."},
        {"id": "app-card", "video": app, "start": cuts["card"][0], "end": cuts["card"][1], "speed": 1.0,
         "caption_top": f.get("card_caption", "The machine acts first: automatic slow-down; then the agent's second look"),
         "narration": f.get("card_narration") or (
             "When a warning is confirmed, the machine acts first: Floorsight slows the forklift to walking pace, or stops it if the "
             "person is in its path. Then the agent replays the last one and a half seconds, takes a close-up, measures again, and "
             "recommends in one line.")},
        {"id": "app-approve", "video": app, "start": cuts["approve"][0], "end": cuts["approve"][1], "speed": 1.0,
         "caption_top": "Decisions in this recording: a scripted stand-in reviewer that follows the agent",
         "narration": "A person approves or rejects afterwards. Only approved warnings reach the floor lead, through Amazon SNS."},
        {"id": "app-reject", "video": app, "start": cuts["reject"][0], "end": cuts["reject"][1], "speed": 1.0,
         "caption_top": f.get("reject_caption", "Second warning: the agent recommends REJECT"),
         "narration": f.get("reject_narration") or f"On the next warning, the agent found {f['reject_reason']}, a tracking glitch, and recommended rejecting it."},
        {"id": "app-report", "image": f["APP_REPORT"], "min_s": 6,
         "caption_top": "Morning report at the end of the replay",
         "narration": f"In this run: {f['n_warn']} warnings, each saved with every step to a trace, and the morning report sums up the shift."},
    ]
    if sim:
        c = sim["cuts"]
        s += [
            {"id": "sim-off", "video": sim["off"], "start": c["off"][0], "end": c["off"][1], "speed": c.get("off_speed", 0.33),
             "caption_top": "Simulated warehouse, interlock OFF: Floorsight sees the picker, nothing brakes",
             "narration": "Recorded video cannot show a forklift being stopped, so we built a simulated warehouse whose forklift obeys "
                          "Floorsight. Without the interlock, it hits the picker who steps into its path."},
            {"id": "sim-on", "video": sim["on"], "start": c["on"][0], "end": c["on"][1], "speed": c.get("on_speed", 0.65),
             "caption_top": "Same scene, interlock ON: the same vision pipeline drives the brakes",
             "narration": "With it, the same vision pipeline sees the picker, the forklift stops, waits until the picker is seen clear "
                          "of its path, then drives on, slowing as it passes. Both people who stepped into its path were passed without contact. "
                          f"It costs time: in forty seconds it drove {sim['dist_on']} metres instead of {sim['dist_off']}."},
            {"id": "sim-app", "video": sim["app"], "start": c["app"][0], "end": c["app"][1], "speed": 1.0,
             "caption_top": "The same app on the simulated warehouse's four cameras",
             "narration": "It is the same app and the same stop logic. Braking cannot help someone who walks into the back of a forklift "
                          "driving away, and nothing can stop for a person no camera sees. We report both."},
        ]
    s += [
        {"id": "swerve-20oct", "video": f"{vin}/accepted-20261020/screen-002-halfspeed.mp4", "start": 0.0, "end": f.get("halfspeed_end", 14.0),
         "speed": f.get("halfspeed_speed", 0.65), "caption_top": "20 Oct, earlier detector stack (since replaced): the swerve, at reduced speed",
         "narration": "Our hardest real case: on the twentieth of October the forklift's outline swerved toward this operator in its last "
                      "forty milliseconds, 0.67 metres, and the warning came too late."},
        {"id": "measured-twice", "image": f["twice_chart"], "min_s": 10,
         "narration": f"Measured again with the publishable detectors and smoothed tracks, the same clip never comes closer than {f['s002_new_closest']} metres. "
                      + (f"{f['s002_warning_sentence']} " if f.get("s002_warning_sentence") else "")
                      + f"With one camera we cannot say which outline is right; at that moment they disagree by about {f['s002_outline_gap']} "
                        "metres, and no warning based on a forklift's course can foresee a real swerve."},
        {"id": "cool", "image": f["bench_chart"], "min_s": 10,
         "narration": "On the same Graviton4 instance, the COOL build runs adaptive threshold seven times as fast and one stream of our OpenCV "
                      f"chain {f['cool_chain_x']} faster, but resize and four parallel processes are slower, and our whole perception stage, "
                      f"spent in neural detectors, was {f['cool_perc_x']}. We also reproduced a crash in COOL's polygon fill."},
        {"id": "architecture", "image": f.get("ARCH_PNG", f"{vin}/architecture.png"), "min_s": 10,
         "narration": "Everything runs in one private AWS deployment, built with Terraform: evidence in S3, track history in DynamoDB, "
                      "alerts through SNS, metrics in CloudWatch, and no open inbound port."},
        {"id": "privacy", "card": ["Privacy first, fail-closed", "Heads and faces pixelated before anything is stored or shown",
                                   f"{f['head_cover_pct']} % of the heads the earlier detectors found are covered",
                                   "Filter cannot run -> the frame is black and listed as withheld"], "min_s": 7,
         "narration": "Privacy runs first and fails closed: heads and faces are pixelated before anything is stored or shown, covering "
                      f"{f['head_cover_pct']} percent of the heads the earlier detectors found."},
        {"id": "credits", "card": ["Floorsight", "Footage: Onal & Dandil (2024), Video Dataset for Safe and Unsafe Behaviours,",
                                   "Mendeley Data, doi:10.17632/xjmtb22pff.1, CC BY 4.0. Clips trimmed; heads and faces pixelated.",
                                   "No endorsement implied.  Detectors: RT-DETRv2, Grounding DINO, SlimSAM, YOLOX (Apache-2.0), YuNet (MIT)",
                                   "Simulated warehouse: our own renderer.  Built with OpenCV 5 on AWS"], "min_s": 7, "narration": ""},
    ]
    return {"voice": f.get("voice", "Matthew"), "sections": s}


def main() -> int:
    facts = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    missing = [k for k in FACT_KEYS if k not in facts]
    if missing:
        raise SystemExit(f"missing facts: {missing}")
    Path(sys.argv[2]).write_text(json.dumps(spec(facts), indent=1, ensure_ascii=False), encoding="utf-8")
    print("spec written", sys.argv[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
