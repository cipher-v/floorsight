"""Controlled camera-evidence evaluation of the production review agent.

This is a synthetic challenge set, not a real-footage accuracy claim. Scene
geometry supplies labels; OpenCV measures independently rendered second-look
pixels. The production ReplayVision and FloorsightAgent make the decisions.
No label or scene-family name is passed to the agent. No reviewer is simulated
in the primary metrics, and no notification is sent.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from floorsight.agent.planner import LocalPlanner
from floorsight.agent.rules import Candidate
from floorsight.agent.runtime import FloorsightAgent
from floorsight.contracts import Evidence, EventType, SitePlan
from floorsight.realclip import ReplayVision

FAMILIES = (
    'hazard_stable', 'hazard_restart', 'hazard_heading_noise',
    'hazard_closer_than_tracker', 'hazard_missing_second_look',
    'clear_heading_noise', 'clear_tracking_gap', 'clear_stable_wrong_gap',
    'clear_missing_second_look', 'clear_frame_edge',
)


def scene(family, seed):
    rng = np.random.default_rng(seed)
    hazard = family.startswith('hazard_')
    gap = float(rng.uniform(.22, .64) if hazard else rng.uniform(1.6, 2.8))
    # Rendered top-down geometry at 100 px/m; gap is measured edge-to-edge.
    img = np.zeros((240, 720, 3), np.uint8)
    vx = int(rng.integers(40, 80)); right = vx + 180
    cv2.rectangle(img, (vx, 80), (right, 180), (255, 0, 0), -1)
    person_x = right + round(gap * 100)
    cv2.rectangle(img, (person_x, 105), (person_x + 25, 155), (0, 0, 255), -1)
    actual_gap = (person_x - right) / 100
    if 'missing_second_look' in family:
        img[:] = 0  # missing pixels, not an oracle returning "inconclusive"
    tracked_gap = 1.5 if family == 'hazard_closer_than_tracker' else .55
    history = []
    for f in range(15):
        heading = 0.0
        restart = family == 'hazard_restart'
        if ('heading_noise' in family and f == 12) or (restart and f >= 12):
            heading = math.radians(75)
        vehicle = {'id': 'F1', 'heading': heading, 'speed': [1.8, 2.0, 2.2],
                   'est': family == 'clear_tracking_gap' and 8 <= f < 12,
                   'restart': restart and f == 12}
        alerts = [] if f < 12 else [{'person': 'P1', 'vehicle': 'F1',
            'clearance': {'median': tracked_gap}, 'pass': {'median': .3},
            'cut': family == 'clear_frame_edge'}]
        history.append((f / 25, {'vehicles': [vehicle], 'alerts': alerts}))
    return {'image': img, 'history': history, 'fps': 25, 't': 14 / 25,
            'tracked_gap': tracked_gap}, {'hazard': actual_gap < .8, 'gap_m': actual_gap}


class CameraEvidence:
    def __init__(self, observations, evidence_dir=None):
        self.history = observations['history']
        self.fps, self.t = observations['fps'], observations['t']
        self.image, self.tracked_gap = observations['image'], observations['tracked_gap']
        self.frames = []
        self.evidence_dir = evidence_dir

    def short_replay(self, track_ids, seconds=1.5):
        return ReplayVision.short_replay(self, track_ids, seconds)

    def sharper_crop(self, camera_id, xy, radius_m=1.5):
        ref = None
        if self.evidence_dir:
            self.evidence_dir.mkdir(parents=True, exist_ok=True)
            path = self.evidence_dir / 'second-look.png'
            cv2.imwrite(str(path), self.image)
            ref = str(path)
        return Evidence(kind='crop', camera_id=camera_id, t=self.t, supports=None,
                        image_ref=ref, note='Synthetic top-down camera crop; no real people.')

    def redetect(self, track_ids):
        # Actual OpenCV inference on the pixels, without scene labels or coordinates.
        boxes = []
        for low, high in [((200, 0, 0), (255, 30, 30)), ((0, 0, 200), (30, 30, 255))]:
            mask = cv2.inRange(self.image, low, high)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return Evidence(kind='redetect', t=self.t, supports=None, note='Second-look object missing.')
            boxes.append(cv2.boundingRect(max(contours, key=cv2.contourArea)))
        a, b = boxes
        gap = max(0, b[0] - (a[0] + a[2] - 1)) / 100
        agrees = abs(gap - self.tracked_gap) <= max(.4, .3 * self.tracked_gap)
        return Evidence(kind='redetect', t=self.t, supports=agrees, score=gap,
            note=f'OpenCV re-measured {gap:.2f} m; tracker reported {self.tracked_gap:.2f} m.')


def run_case(observations, planner=None, evidence_dir=None):
    site = SitePlan.model_validate({'site_id': 'controlled-eval', 'name': 'Controlled evaluation',
        'width_m': 10, 'depth_m': 10, 'floor_stage': True, 'zones': [], 'obstacles': [], 'markers': [],
        'cameras': [{'camera_id': 'camA', 'width': 720, 'height': 240, 'fps': 25,
                     'K': [[100, 0, 360], [0, 100, 120], [0, 0, 1]], 'nominal_position': [5, 5, 4]}]})
    agent = FloorsightAgent(site, CameraEvidence(observations, evidence_dir), planner=planner or LocalPlanner())
    agent.t = observations['t']
    candidate = Candidate(key=('evaluation',), type=EventType.floor_warning, rule=None,
        t=agent.t, t_start=12/25, location=(5, 5), zone_id=None, track_ids=['P1', 'F1'],
        camera_ids=['camA'], severity='warning', summary='Candidate: predicted close pass.',
        data={'level': 'warning', 'clearance_m': observations['tracked_gap']})
    agent._open_case(candidate)
    case = next(iter(agent.cases.values()))
    return agent, case


class ExitCameraEvidence:
    def __init__(self, image):
        self.image = image

    def sharper_crop(self, camera_id, xy, radius_m=1.):
        return Evidence(kind='crop', camera_id=camera_id, t=20, supports=None,
                        note='Primary camera candidate; synthetic scene.')

    def second_view(self, xy, exclude, kind):
        if self.image is None:
            return []  # unavailable camera must not become negative evidence
        mask = cv2.inRange(self.image, (200, 0, 0), (255, 30, 30))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        occupied = any(cv2.contourArea(c) >= 100 for c in contours)
        return [Evidence(kind='second_view', camera_id='camB', t=20, supports=occupied,
                         note='OpenCV measured occupancy in the independent exit view.')]


def evaluate_blocked_exits(variants, seed):
    rows = []
    site = SitePlan.model_validate({'site_id': 'exit-eval', 'name': 'Two-camera exit',
        'width_m': 10, 'depth_m': 10, 'zones': [], 'obstacles': [], 'markers': [],
        'cameras': [{'camera_id': c, 'width': 160, 'height': 120, 'fps': 25,
            'K': [[100, 0, 80], [0, 100, 60], [0, 0, 1]], 'nominal_position': [5, 5, 4]}
            for c in ['camA', 'camB']]})
    for hazard, visible in [(True, True), (False, True), (True, False), (False, False)]:
        for i in range(variants):
            rng = np.random.default_rng(seed + i)
            img = np.zeros((120, 160, 3), np.uint8)
            if hazard:
                x, y = rng.integers(20, 50, size=2)
                cv2.rectangle(img, (x, y), (x + 30, y + 25), (255, 0, 0), -1)
            agent = FloorsightAgent(site, ExitCameraEvidence(img if visible else None), planner=LocalPlanner())
            agent.t = 20
            agent._open_case(Candidate(key=('exit',), type=EventType.blocked_exit, rule=None,
                t=20, t_start=0, location=(5, 5), zone_id=None, track_ids=['O1'], camera_ids=['camA'],
                severity='warning', summary='Primary camera reports a blocked exit.'))
            case = next(iter(agent.cases.values()))
            rows.append({'hazard': hazard, 'second_camera_available': visible, 'variant': i,
                'alerts': len(agent.alerts), 'status': case.event.status.value,
                'pending_reviews': sum(a.status.value == 'pending' for a in agent.approvals.values()),
                'trace': [s.model_dump(mode='json') for s in agent.trace]})
    observable = [r for r in rows if r['second_camera_available']]
    return {'scope': 'Synthetic exit-occupancy cases. Production agent alert records; no external notifier.',
        'metrics': {'cases': len(rows), 'visible_cases': len(observable),
            'baseline_false_alerts_visible': sum(not r['hazard'] for r in observable),
            'agent_false_alerts_visible': sum(r['alerts'] for r in observable if not r['hazard']),
            'hazards_visible': sum(r['hazard'] for r in observable),
            'hazards_alerted_visible': sum(r['alerts'] for r in observable if r['hazard']),
            'blind_cases': sum(not r['second_camera_available'] for r in rows),
            'blind_cases_pending_review': sum(r['pending_reviews'] for r in rows if not r['second_camera_available']),
            'hazards_rejected_all': sum(r['hazard'] and r['status'] == 'rejected' for r in rows)}, 'cases': rows}


def evaluate(variants=8, seed=20261027, evidence_dir=None):
    rows = []
    for fi, family in enumerate(FAMILIES):
        for i in range(variants):
            case_seed = seed + fi * 1000 + i
            obs, truth = scene(family, case_seed)
            start = time.perf_counter()
            agent, case = run_case(obs, evidence_dir=(evidence_dir / f'{family}-{i}' if evidence_dir else None))
            elapsed = time.perf_counter() - start
            verdict = (case.facts.get('recommendation') or {}).get('verdict', 'uncertain')
            rows.append({'case_id': f'{family}-{i}', 'seed': case_seed, 'family': family,
                'label': truth, 'recommendation': verdict, 'retained_for_review': verdict != 'reject',
                'tool_calls': len(agent.trace), 'elapsed_ms': round(elapsed * 1000, 3),
                'pending_approvals': sum(a.status.value == 'pending' for a in agent.approvals.values()),
                'actual_alerts_before_approval': len(agent.alerts),
                'evidence': [e.model_dump(mode='json') for e in case.event.evidence],
                'trace': [s.model_dump(mode='json') for s in agent.trace]})
    real = [r for r in rows if r['label']['hazard']]
    clear = [r for r in rows if not r['label']['hazard']]
    fp = sum(r['retained_for_review'] for r in clear)
    missed = sum(not r['retained_for_review'] for r in real)
    result = {'schema': 'floorsight.agent-eval.v1', 'seed': seed, 'variants_per_family': variants,
        'scope': 'Synthetic candidate-review evaluation. Neither detector recall nor real-world safety accuracy.',
        'label_method': 'Rendered scene edge-to-edge clearance < 0.8 m; labels withheld from tools/planner.',
        'baseline': 'Send an alert for every candidate entering review (all cases are candidates).',
        'agent_metric': 'Conservative routing: confirm and uncertain remain actionable; only reject is removed. Human approval remains required.',
        'environment': {'python': platform.python_version(), 'opencv': cv2.__version__},
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), ROOT/'floorsight/agent/planner.py', ROOT/'floorsight/agent/runtime.py', ROOT/'floorsight/realclip.py']},
        'metrics': {'cases': len(rows), 'hazards': len(real), 'clear': len(clear),
            'baseline_false_flags': len(clear), 'agent_false_flags_retained': fp,
            'false_flags_removed': len(clear)-fp,
            'false_flag_reduction': (len(clear)-fp)/len(clear) if clear else None,
            'hazards_rejected': missed, 'hazards_retained': len(real)-missed,
            'recommendations': dict(Counter(r['recommendation'] for r in rows)),
            'hazard_recommendations': dict(Counter(r['recommendation'] for r in real)),
            'clear_recommendations': dict(Counter(r['recommendation'] for r in clear)),
            'actual_alerts_before_approval': sum(r['actual_alerts_before_approval'] for r in rows),
            'tool_calls': sum(r['tool_calls'] for r in rows),
            'p50_review_ms': float(np.median([r['elapsed_ms'] for r in rows]))}, 'cases': rows,
        'blocked_exit': evaluate_blocked_exits(variants, seed)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--variants', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261027)
    p.add_argument('--evidence-dir', type=Path)
    p.add_argument('--check', action='store_true', help='fail if hazards are rejected, no false flags removed, or alerts bypass approval')
    args = p.parse_args()
    if args.variants < 1:
        p.error('--variants must be positive')
    result = evaluate(args.variants, args.seed, args.evidence_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['metrics'], indent=2))
    print(json.dumps(result['blocked_exit']['metrics'], indent=2))
    m = result['metrics']
    if args.check and (m['hazards_rejected'] or not m['false_flags_removed'] or m['actual_alerts_before_approval']):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
