"""GCP-only debug replay: preserve camera observations at difficult timestamps.

No truth files, labels, or positions are read. Full-resolution filtering precedes
the vision pipeline and every stored frame. This is not the throughput benchmark.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import cv2
import numpy as np
from floorsight.edge.privacy import blur_faces
from floorsight.agent.runtime import FloorsightAgent
from floorsight.agent.planner import LocalPlanner
from floorsight.agent.language import LocalRuleParser
from floorsight.interfaces import FramePacket
from floorsight.server.components import load_setup_plates
from floorsight.vision.feed import VideoFeed
from floorsight.vision.pipeline import RealVision


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    if os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED') != 'yes':
        raise SystemExit('GCP lease required; do not run this job on the owner PC')
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--times', nargs='+', type=float, default=[47.3, 55.9, 104., 108.5, 149.3])
    a = p.parse_args()
    cv2.setNumThreads(1)
    root = Path(__file__).resolve().parents[1]
    source = root / 'data/sim/demo_001'
    out = root / 'eval/diagnostics' / a.name
    out.mkdir(parents=True, exist_ok=False)
    feed = VideoFeed(str(source))
    vision = RealVision(feed.site)
    vision.set_background_plates(load_setup_plates(str(source), feed.site))
    agent = FloorsightAgent(feed.site, vision, planner=LocalPlanner(), rule_parser=LocalRuleParser(feed.site))
    try:
        for packets in feed:
            filtered = {}
            for cid, packet in packets.items():
                image, n = blur_faces(packet.image)
                filtered[cid] = FramePacket(packet.meta.model_copy(update={'blur_applied': True, 'faces_blurred': n}), image)
            world = vision.process(filtered)
            if any(abs(world.t - target) <= .4 for target in a.times):
                record = {'t':world.t, 'world':world.model_dump(mode='json'),
                          'detections':[asdict(d) for d in vision.last_dets], 'cameras':{},
                          'internal_tracks':[{'track_id':f'T{tr.tid}', 'confirmed':tr.confirmed,
                              'class_votes':{k.value:v for k,v in tr.votes.items()},
                              'measurement_sigma':tr.measurement_sigma, 'kalman_sigma':tr.sigma,
                              'last_observed_xy':tr.seen_xy, 'last_observed_velocity':tr.seen_v,
                              'last_t':tr.last_t, 'blind':tr.blind} for tr in vision.tracker.tracks]}
                for cid, cam in vision.cams.items():
                    record['cameras'][cid] = {'status':cam.status.value, 'last_t':cam.last_t,
                        'calibration':None if cam.calib is None else {'K':cam.calib.K, 'R':cam.calib.R, 'tvec':cam.calib.tvec},
                        'blobs':[{'bbox':b.bbox, 'foot_px':b.foot_px, 'kind':b.kind, 'bottom_px':b.bottom_px} for b in cam.blobs]}
                    cv2.imwrite(str(out / f'{world.t:.1f}_{cid}.jpg'), filtered[cid].image)
                (out / f'{world.t:.1f}.json').write_text(json.dumps(record, default=serial, indent=2), encoding='utf-8')
            # Match Runtime._tick ordering: vision first, then agent. In particular,
            # the real agent repairs the knocked camera; omitting it invalidates
            # diagnostics after the knock even when the source pixels are identical.
            agent.on_world(world)
            if world.t > max(a.times) + .5:
                break
    finally:
        vision.close()
    (out / 'agent-state.json').write_text(json.dumps(agent.snapshot(), indent=2), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
