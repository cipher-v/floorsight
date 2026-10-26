"""GCP-only diagnostic: why the stop-hold (tracker step 3b) lets go, and how a candidate
pixel test would have scored on the same frames.

Production decisions are unchanged: the tracker source is instrumented by inserting log
lines next to each existing `continue` (needle counts are asserted). The pipeline consumes
only filtered pixels, calibration and the public site plan. Simulator truth is read only
after the run, by the evaluation section at the bottom, to label already-recorded frames.
"""
import argparse
import json
import math
import os
import sys
import types
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

if os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED') != 'yes':
    raise SystemExit('GCP lease required; do not run this job on the owner PC')

import floorsight.vision.pipeline as pipeline_module
from floorsight.agent.language import LocalRuleParser
from floorsight.agent.planner import LocalPlanner
from floorsight.agent.runtime import FloorsightAgent
from floorsight.edge.privacy import blur_faces
from floorsight.interfaces import FramePacket
from floorsight.server.components import load_setup_plates
from floorsight.vision.feed import VideoFeed

VERSION = 'hold-evidence-diagnostic-v1'


def instrumented_tracker():
    src = Path('floorsight/vision/tracker.py').read_text()
    edits = [
        ('        self.new_confirmed = []\n', '        self.new_confirmed = []\n        self.hold_log = []\n'),
        ('                if not tr.confirmed or tr.last_t == t or tr.blind or t - tr.last_t > 1.0:\n                    continue\n',
         '                if not tr.confirmed or tr.last_t == t or tr.blind or t - tr.last_t > 1.0:\n'
         '                    if tr.confirmed and tr.last_t != t:\n'
         "                        self.hold_log.append(dict(tid=tr.tid, reason='blind' if tr.blind else 'stale_over_1s'))\n"
         '                    continue\n'),
        ('                    continue  # only someone who has really walked here can be standing here\n',
         "                    self.hold_log.append(dict(tid=tr.tid, reason='forklift_class' if tr.cls == TrackClass.forklift else 'short_walk',"
         ' max_disp=tr.max_disp, walked_s=tr.last_move_t - tr.first_t))\n'
         '                    continue\n'),
        ('                if any(np.linalg.norm(at - o) < 1.0 for o in parked + bulky):\n                    continue\n',
         '                if any(np.linalg.norm(at - o) < 1.0 for o in parked + bulky):\n'
         "                    self.hold_log.append(dict(tid=tr.tid, reason='near_parked_or_bulky'))\n"
         '                    continue\n'),
        ('                if any(np.linalg.norm(at - f) < 2.0 for f in forklifts if f is not tr.xy):\n                    continue\n',
         '                if any(np.linalg.norm(at - f) < 2.0 for f in forklifts if f is not tr.xy):\n'
         "                    self.hold_log.append(dict(tid=tr.tid, reason='forklift_within_2m'))\n"
         '                    continue\n'),
        ('np.linalg.norm(at - o.xy) < 1.2 for o in movers):\n                    continue\n',
         'np.linalg.norm(at - o.xy) < 1.2 for o in movers):\n'
         "                    self.hold_log.append(dict(tid=tr.tid, reason='other_mover_within_1.2m'))\n"
         '                    continue\n'),
        ('                    c.person_like and np.linalg.norm(c.xy - at) < 0.7 for c in static\n                ):\n                    continue\n',
         '                    c.person_like and np.linalg.norm(c.xy - at) < 0.7 for c in static\n                ):\n'
         "                    self.hold_log.append(dict(tid=tr.tid, reason='hold_expired_no_static'))\n"
         '                    continue\n'),
        ('                frac, cams = self.presence(at)\n',
         '                frac, cams = self.presence(at)\n'
         "                self.hold_log.append(dict(tid=tr.tid, reason='held' if frac >= 0.15 else 'presence_below_0.15', presence=frac))\n"),
    ]
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    module = types.ModuleType('instrumented_tracker')
    sys.modules[module.__name__] = module
    exec(compile(src, 'instrumented-tracker.py', 'exec'), module.__dict__)
    return module.Tracker


def box_metrics(vision, cam, xy, history):
    box = vision._box_px(cam, (float(xy[0]), float(xy[1])), 0.3, 1.7)
    det = cam.det
    if box is None or det.diff_mask is None or det.moving_mask is None:
        return None
    s = det.scale
    x, y, w, h = (int(round(v * s)) for v in box)
    H, W = det.diff_mask.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    sl = (slice(y0, y1), slice(x0, x1))
    diff = det.diff_mask[sl]
    moving = det.moving_mask[sl]
    static = det.static_mask[sl] if det.static_mask is not None else np.zeros_like(diff)
    out = dict(box=list(box), diff=float(diff.mean()), unexplained=float((diff & ~moving).mean()),
               static=float(static.mean()), upper=float(diff[: max(1, (y1 - y0) // 2)].mean()))
    if len(det.prev_gray) >= 3:
        change = cv2.absdiff(det.prev_gray[-1], det.prev_gray[-3])[sl] > 14
        out['still'] = float((diff & ~change).mean())
    if len(history) >= 6:
        old = history[0][sl]
        union = int((diff | old).sum())
        out['stable_iou'] = float((diff & old).sum() / union) if union else None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    a = p.parse_args()
    cv2.setNumThreads(1)
    source = Path('data/sim/demo_001')
    out = Path('eval/diagnostics') / a.name
    out.mkdir(parents=True, exist_ok=False)
    pipeline_module.Tracker = instrumented_tracker()
    feed = VideoFeed(str(source))
    vision = pipeline_module.RealVision(feed.site)
    vision.set_background_plates(load_setup_plates(str(source), feed.site))
    agent = FloorsightAgent(feed.site, vision, planner=LocalPlanner(), rule_parser=LocalRuleParser(feed.site))
    history = {cid: deque(maxlen=6) for cid in vision.cams}
    frames = []
    try:
        for packets in feed:
            filtered = {}
            for cid, packet in packets.items():
                image, n = blur_faces(packet.image)
                filtered[cid] = FramePacket(packet.meta.model_copy(update={'blur_applied': True, 'faces_blurred': n}), image)
            world = vision.process(filtered)
            tr_by_id = {tr.tid: tr for tr in vision.tracker.tracks}
            holds = []
            for entry in vision.tracker.hold_log:
                tr = tr_by_id.get(entry['tid'])
                if tr is None:
                    continue
                cams = {}
                for cid, cam in vision.cams.items():
                    if not cam.usable or cam.visible is None:
                        continue
                    c, r = int(tr.seen_xy[0] // pipeline_module.CELL_M), int(tr.seen_xy[1] // pipeline_module.CELL_M)
                    if not (0 <= r < vision.rows and 0 <= c < vision.cols) or not cam.visible[r, c]:
                        continue
                    m = box_metrics(vision, cam, tr.seen_xy, history[cid])
                    if m is not None:
                        cams[cid] = m
                holds.append(dict(entry, xy=[float(v) for v in tr.seen_xy], last_t=tr.last_t,
                                  hold_since=tr.hold_since, cls=tr.cls.value, cameras=cams))
            for cid, cam in vision.cams.items():
                if cam.det.diff_mask is not None:
                    history[cid].append(cam.det.diff_mask.copy())
            frames.append(dict(t=world.t, holds=holds,
                               tracks=[dict(id=k.track_id, cls=k.cls.value, x=k.x, y=k.y, state=k.state.value) for k in world.tracks],
                               internal=[dict(tid=tr.tid, xy=[float(v) for v in tr.xy], last_t=tr.last_t, confirmed=tr.confirmed,
                                              kind=tr.kind) for tr in vision.tracker.tracks]))
            agent.on_world(world)
    finally:
        vision.close()
    with (out / 'frames.jsonl').open('w') as f:
        for fr in frames:
            f.write(json.dumps(fr) + '\n')
    evaluate(out, frames, source)
    print(out)


# ---------------- evaluation only: truth is read after the run is complete ---------------- #

def evaluate(out, frames, source):
    from eval.e2e.visibility import TruthVisibility
    truth = {round(d['t'], 1): d['entities'] for d in map(json.loads, (source / 'truth_frames.jsonl').read_text().splitlines())}
    vis = TruthVisibility(str(source))
    expected = matched = 0
    miss_reasons, miss_motion, window_reasons = Counter(), Counter(), {}
    rule_rows = []
    for fr in frames:
        t = fr['t']
        ents = [e for e in truth.get(round(t, 1), []) if e['cls'] in ('person', 'forklift') and vis.placeable(e, t)]
        dots = [d for d in fr['tracks'] if d['state'] == 'seen' and d['cls'] in ('person', 'forklift', 'unknown')]
        good = set()
        if ents and dots:
            cost = np.array([[math.dist((e['x'], e['y']), (d['x'], d['y'])) for d in dots] for e in ents])
            rows, cols = linear_sum_assignment(cost)
            good = {int(i) for i, j in zip(rows, cols) if cost[i, j] <= .75}
        expected += len(ents)
        matched += len(good)
        holds = {h['tid']: h for h in fr['holds']}
        for i, e in enumerate(ents):
            if i in good:
                continue
            speed = math.hypot(e.get('vx', 0.0), e.get('vy', 0.0))
            motion = 'still' if speed < 0.3 else 'moving'
            near = [m for m in fr['internal'] if m['kind'] == 'mover' and m['confirmed']]
            near = min(near, key=lambda m: math.dist(m['xy'], (e['x'], e['y'])), default=None)
            if near is None or math.dist(near['xy'], (e['x'], e['y'])) > 1.5:
                reason = 'no_confirmed_track_within_1.5m'
            elif near['tid'] in holds:
                reason = holds[near['tid']]['reason']
            elif near['last_t'] == t:
                reason = 'track_updated_but_not_counted'
            else:
                reason = 'track_not_logged'
            miss_reasons[(e['cls'], motion, reason)] += 1
            miss_motion[(str(e['id']), motion)] += 1
        # Label every logged hold decision by what truth says is at the held point.
        for h in fr['holds']:
            if h['reason'] == 'blind':
                continue
            people = [e for e in truth.get(round(t, 1), []) if e['cls'] == 'person']
            d = min((math.dist(h['xy'], (e['x'], e['y'])), e) for e in people) if people else (9e9, None)
            label = 'absent'
            if d[1] is not None and d[0] < 0.5:
                label = 'present_still' if math.hypot(d[1].get('vx', 0), d[1].get('vy', 0)) < 0.3 else 'present_moving'
            rule_rows.append(dict(t=t, tid=h['tid'], reason=h['reason'], label=label, cams=h['cameras']))

    def rule(name, fn):
        c = Counter()
        for r in rule_rows:
            c[(r['label'], bool(fn(list(r['cams'].values()))))] += 1
        pos = c[('present_still', True)] + c[('present_still', False)]
        neg = c[('absent', True)] + c[('absent', False)]
        return dict(rule=name, still_true=c[('present_still', True)], still_total=pos,
                    absent_true=c[('absent', True)], absent_total=neg,
                    moving_true=c[('present_moving', True)], moving_total=c[('present_moving', True)] + c[('present_moving', False)])

    def need(cs, k, pred):
        return len(cs) > 0 and sum(pred(c) for c in cs) >= min(k, len(cs))

    rules = [
        rule('existing: diff>=.15 in min(2,n) cams', lambda cs: need(cs, 2, lambda c: c['diff'] >= .15)),
        rule('diff>=.15 & stable>=.8, min(2,n) cams', lambda cs: need(cs, 2, lambda c: c['diff'] >= .15 and (c.get('stable_iou') or 0) >= .8)),
        rule('diff>=.15 & stable>=.8, any cam', lambda cs: need(cs, 1, lambda c: c['diff'] >= .15 and (c.get('stable_iou') or 0) >= .8)),
        rule('unexplained>=.15 & stable>=.8, any cam', lambda cs: need(cs, 1, lambda c: c['unexplained'] >= .15 and (c.get('stable_iou') or 0) >= .8)),
        rule('still>=.15 & stable>=.8, any cam', lambda cs: need(cs, 1, lambda c: c.get('still', 0) >= .15 and (c.get('stable_iou') or 0) >= .8)),
        rule('upper>=.15 & stable>=.8, any cam', lambda cs: need(cs, 1, lambda c: c['upper'] >= .15 and (c.get('stable_iou') or 0) >= .8)),
        rule('unexplained>=.15 & stable>=.9, min(2,n) cams', lambda cs: need(cs, 2, lambda c: c['unexplained'] >= .15 and (c.get('stable_iou') or 0) >= .9)),
    ]
    by_reason = Counter((r['reason'], r['label']) for r in rule_rows)
    summary = dict(
        version=VERSION,
        limitation='Diagnostic replay (offline, agent in loop, full-resolution filtering) approximates but is not the frozen e2e scorer run. '
                   'Truth labels are evaluation-only and were read after the replay finished.',
        replay_recall=round(matched / max(expected, 1), 4), expected=expected, matched=matched,
        misses_by_class_motion_reason=sorted(([*k, v] for k, v in miss_reasons.items()), key=lambda x: -x[-1]),
        misses_by_entity_motion=sorted(([*k, v] for k, v in miss_motion.items()), key=lambda x: -x[-1]),
        hold_decisions_by_reason_label=sorted(([*k, v] for k, v in by_reason.items()), key=lambda x: -x[-1]),
        candidate_rules=rules,
    )
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (out / 'hold_rows.jsonl').open('w') as f:
        for r in rule_rows:
            f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
