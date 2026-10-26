# Agent traces: how camera evidence changes what Floorsight does next

Two unedited traces from the app on AWS (Graviton4, COOL OpenCV 5.1), written live, one JSON object per line:

| File | Run | What it shows |
|---|---|---|
| [`app-run-09-trace-20261024.jsonl`](app-run-09-trace-20261024.jsonl) | final run, 24 Oct, code r55: six real clips, smoothing and the interlock on | the interlock slows, stops and resumes a forklift; the agent checks 3 warnings and recommends confirm; 3 alerts |
| [`app-run-02-trace-20261024.jsonl`](app-run-02-trace-20261024.jsonl) | earlier run, 24 Oct, before smoothing | the agent **rejects** a warning built on a tracking glitch |

The `*-snapshot.json` files hold the dashboard's state at the end of each run. Line kinds: `interlock` (slow / stop /
horn / resume commands, with the stopping distance), `event` (a warning and its status changes), `agent_step` (tool,
input, output, decision and reason), `approval` (what the person was asked and what they decided), `alert`.
Decisions came from a scripted stand-in reviewer that follows the agent; the agent's planner was the local policy
(Claude on Amazon Bedrock is built but was not open to our account).

## Example 1: OpenCV measurements make the agent reject a warning (app-run-02, EV-0002, clip screen-032)

The rule fired: forklift S032:F1 would pass within 0.8 m of person S032:P11 in 1.7 s.

| Step | Tool | What it found | Effect |
|---|---|---|---|
| 1 | `short_replay` | the last 1.5 s of floor measurements: the forklift's measured speed jumped 1.0 → 2.1 m/s, then 2.3 → 1.3 m/s, between frames | "not sustained / artefact" |
| 2 | `sharper_crop` | a full-resolution close-up of the privacy-filtered frame for the reviewer | evidence stored |
| 3-4 | `redetect` | the detectors run again on a sharper crop of the raw frame, inside the private stage, and the floor gap is re-measured independently of the tracker | "second detector pass disagrees" |
| 5 | `recommend` | | **reject**: "likely a tracking glitch (speed jumped 1.0 -> 2.1 m/s)" |
| 6 | `request_approval` | the person is asked with the agent's REJECT recommendation and its numbers | |
| 7-8 | `human_decision`, `reject_event` | rejected as a false alarm | **no alert sent** |

## Example 2: a real pull-away, confirmed (app-run-09, EV-0003, clip clip-dev-003)

| Step | Tool or command | What it found | Effect |
|---|---|---|---|
| 0 | interlock `slow` | D003:F1 will pass within 0.8 m of D003:P35 in 2.5 s; "a stop now halts it 4.1 m short" | forklift limited to walking pace at once (recorded video: logged, not applied) |
| 1 | `short_replay` | the condition held 3 frames; the forklift **pulled away in a new direction after a stop** (heading 164° from its stopped one), real motion, not a tracking jump | "sustained" |
| 2 | `sharper_crop` | close-up stored | |
| 3-4 | `redetect` | second detector pass inconclusive (person not re-detected on the crop) | |
| 5 | `recommend` | | **confirm** |
| 6-9 | `request_approval`, `human_decision`, `confirm_event`, `send_alert` | approved | alert to the floor lead (Amazon SNS) |

In an earlier run (app-run-04) the agent called this kind of pull-away a tracking glitch; the perception stage now flags a
restart in a new direction after a stop, and the agent reads it as real motion.
