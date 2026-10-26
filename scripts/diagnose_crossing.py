"""GCP-only diagnostic: where does the crossing worker's detection get lost near the forklift?

Traces every camera's floor detections, the fused clusters and the track updates through the
three Aisle-4 crossings. Production code is unchanged and consumes only filtered pixels,
calibration and the site plan. Simulator truth is read only after the replay, to label
already-recorded detections and to draw evaluation crosses on a few saved frames.
"""
import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

if os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED') != 'yes':
    raise SystemExit('GCP lease required; do not run this job on the owner PC')

from floorsight.agent.language import LocalRuleParser
from floorsight.agent.planner import LocalPlanner
from floorsight.agent.runtime import FloorsightAgent
from floorsight.edge.privacy import blur_faces
from floorsight.interfaces import FramePacket
from floorsight.server.components import load_setup_plates
from floorsight.vision.feed import VideoFeed
from floorsight.vision.pipeline import RealVision
from floorsight.vision.tracker import cluster

VERSION = 'crossing-trace-v1'
WINDOWS = [(46.0, 56.0), (107.0, 117.0), (148.0, 158.0)]
SNAPSHOTS = {48.5, 49.5, 50.5, 51.5, 53.0, 110.0, 111.0, 112.5, 150.5, 151.5, 153.0}


def in_window(t):
    return any(a <= t <= b for a, b in WINDOWS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    a = p.parse_args()
    cv2.setNumThreads(1)
    source = Path('data/sim/demo_001')
    out = Path('eval/diagnostics') / a.name
    out.mkdir(parents=True, exist_ok=False)
    feed = VideoFeed(str(source))
    vision = RealVision(feed.site)
    vision.set_background_plates(load_setup_plates(str(source), feed.site))
    agent = FloorsightAgent(feed.site, vision, planner=LocalPlanner(), rule_parser=LocalRuleParser(feed.site))
    frames, shots = [], {}
    try:
        for packets in feed:
            filtered = {}
            for cid, packet in packets.items():
                image, n = blur_faces(packet.image)
                filtered[cid] = FramePacket(packet.meta.model_copy(update={'blur_applied': True, 'faces_blurred': n}), image)
            world = vision.process(filtered)
            t = round(world.t, 1)
            if in_window(t):
                dets = vision.last_dets
                moving = cluster([d for d in dets if d.kind == 'moving'], 0.9, 1.6)
                static = cluster([d for d in dets if d.kind == 'static'], 1.2)
                raw = {cid: [dict(kind=b.kind, bbox=[int(v) for v in b.bbox], foot_px=[float(v) for v in b.foot_px])
                             for b in cam.blobs] for cid, cam in vision.cams.items()}
                frames.append(dict(
                    t=t,
                    raw_blobs=raw,
                    dets=[dict(cam=d.camera_id, kind=d.kind, xy=[float(v) for v in d.xy], sigma=d.sigma, footprint=d.footprint,
                               height=d.height_m, bbox=[int(v) for v in d.bbox]) for d in dets],
                    clusters=[dict(kind=k, xy=[float(v) for v in c.xy], footprint=c.footprint, cams=c.camera_ids,
                                   person_like=c.person_like, height=c.height_m)
                              for k, cs in (('moving', moving), ('static', static)) for c in cs],
                    tracks=[dict(tid=tr.tid, cls=tr.cls.value, kind=tr.kind, confirmed=tr.confirmed, xy=[float(v) for v in tr.xy],
                                 updated=tr.last_t == world.t, hold=tr.hold_since is not None) for tr in vision.tracker.tracks],
                    world=[dict(id=k.track_id, cls=k.cls.value, x=k.x, y=k.y, state=k.state.value) for k in world.tracks],
                    calib={cid: None if cam.calib is None else dict(K=cam.calib.K.tolist(), R=cam.calib.R.tolist(),
                                                                     tvec=np.asarray(cam.calib.tvec).ravel().tolist())
                           for cid, cam in vision.cams.items()},
                ))
                if t in SNAPSHOTS:
                    shots[t] = {cid: filtered[cid].image.copy() for cid in filtered}
            agent.on_world(world)
            if world.t > WINDOWS[-1][1] + 0.5:
                break
    finally:
        vision.close()
    with (out / 'frames.jsonl').open('w') as f:
        for fr in frames:
            f.write(json.dumps(fr) + '\n')
    evaluate(out, frames, shots, source, vision)
    print(out)


# ---------------- evaluation only: truth is read after the replay ---------------- #

def evaluate(out, frames, shots, source, vision):
    truth = {round(d['t'], 1): {e['id']: e for e in d['entities']}
             for d in map(json.loads, (source / 'truth_frames.jsonl').read_text().splitlines())}
    rows = []
    for fr in frames:
        ents = truth.get(fr['t'], {})
        p4, fork = ents.get('p4'), ents.get('forklift1')
        if p4 is None:
            continue
        here = (p4['x'], p4['y'])
        per_cam = {}
        for d in fr['dets']:
            dist = math.dist(d['xy'], here)
            if dist < 0.6:
                per_cam.setdefault(d['cam'], []).append(dict(kind=d['kind'], dist=round(dist, 2), footprint=round(d['footprint'], 2),
                                                            height=round(d['height'], 2)))
        clusters = sorted(((math.dist(c['xy'], here), c) for c in fr['clusters']), key=lambda x: x[0])
        tracks = sorted(((math.dist(k['xy'], here), k) for k in fr['tracks'] if k['kind'] == 'mover' and k['confirmed']),
                        key=lambda x: x[0])
        seen = [w for w in fr['world'] if w['state'] == 'seen' and math.dist((w['x'], w['y']), here) <= 0.75]
        if per_cam and not any(c[0] < 0.75 for c in clusters):
            stage = 'det_near_but_no_cluster_near'
        elif not per_cam:
            stage = 'no_det_within_0.6m'
        elif not any(k[0] < 0.75 for k in tracks):
            stage = 'cluster_near_but_no_track_near'
        elif not seen:
            stage = 'track_near_but_not_seen_dot'
        else:
            stage = 'matched'
        near_cluster = clusters[0] if clusters else None
        rows.append(dict(
            t=fr['t'], stage=stage, p4=[round(v, 2) for v in here],
            p4_speed=round(math.hypot(p4.get('vx', 0), p4.get('vy', 0)), 2),
            forklift_dist=round(math.dist(here, (fork['x'], fork['y'])), 2) if fork else None,
            dets_within_0_6=per_cam,
            nearest_cluster=None if near_cluster is None else dict(dist=round(near_cluster[0], 2), **{k: near_cluster[1][k] for k in ('kind', 'footprint', 'cams', 'person_like')}),
            nearest_track=None if not tracks else dict(dist=round(tracks[0][0], 2), tid=tracks[0][1]['tid'], cls=tracks[0][1]['cls'],
                                                       updated=tracks[0][1]['updated'], hold=tracks[0][1]['hold'])))
    from collections import Counter
    summary = dict(version=VERSION,
                   limitation='Diagnostic replay; truth labels added after the run. Stage is the first pipeline stage where the worker p4 is not represented within the stated distance.',
                   stages_by_window={f'{a}-{b}': Counter(r['stage'] for r in rows if a <= r['t'] <= b) for a, b in WINDOWS},
                   moving_vs_still={f'{a}-{b}': Counter((r['stage'], 'moving' if r['p4_speed'] >= 0.3 else 'still') for r in rows if a <= r['t'] <= b) for a, b in WINDOWS})
    summary['moving_vs_still'] = {k: {f'{s}|{m}': n for (s, m), n in v.items()} for k, v in summary['moving_vs_still'].items()}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (out / 'rows.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    # Evaluation overlays: blue boxes = product blobs (moving solid, static thin); green cross = truth (evaluation only).
    for t, imgs in shots.items():
        fr = next(f for f in frames if f['t'] == t)
        tiles = []
        for cid in sorted(imgs):
            img = imgs[cid].copy()
            for b in fr['raw_blobs'].get(cid, []):
                x, y, w, h = b['bbox']
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 160, 0), 2 if b['kind'] == 'moving' else 1)
            cal = fr['calib'].get(cid)
            if cal is not None:
                for eid, e in truth.get(t, {}).items():
                    if eid not in ('p4', 'forklift1'):
                        continue
                    pts, _ = cv2.projectPoints(np.array([[e['x'], e['y'], 0.0]]), cv2.Rodrigues(np.array(cal['R']))[0],
                                               np.array(cal['tvec']), np.array(cal['K']), None)
                    u, v = pts.ravel()
                    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                        cv2.drawMarker(img, (int(u), int(v)), (60, 220, 60), cv2.MARKER_CROSS, 18, 2)
                        cv2.putText(img, f'TRUTH {eid}', (int(u) + 8, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, .5, (60, 220, 60), 1)
            cv2.putText(img, f'{cid} t={t:.1f}s  blue=product blobs  green=TRUTH (evaluation only)', (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 255), 2)
            tiles.append(cv2.resize(img, (768, 432)))
        cv2.imwrite(str(out / f't{t:.1f}.jpg'), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])]))


if __name__ == '__main__':
    main()
