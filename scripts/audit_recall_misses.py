"""Evaluation-only recall attribution; truth never enters production tracking."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from eval.e2e.visibility import TruthVisibility


def main():
    if os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED') != 'yes':
        raise RuntimeError('GCP lease required')
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    a = p.parse_args()
    sim = Path('data/sim/demo_001')
    truth = {round(d['t'], 1): d['entities'] for d in
             map(json.loads, (sim/'truth_frames.jsonl').read_text().splitlines())}
    vis = TruthVisibility(str(sim))
    counts, entities, misses = Counter(), Counter(), []
    expected = matched = 0
    for record in map(json.loads, (Path('data/runtime')/a.run/'worlds.jsonl').read_text().splitlines()):
        w = record['world']
        ents = [e for e in truth.get(round(w['t'], 1), [])
                if e['cls'] in ('person', 'forklift') and vis.placeable(e, w['t'])]
        tracks = [d for d in w['tracks'] if d['cls'] in ('person', 'forklift', 'unknown')]
        dots = [d for d in tracks if d['state'] == 'seen']
        good = set()
        if ents and dots:
            cost = np.array([[math.dist((e['x'], e['y']), (d['x'], d['y'])) for d in dots] for e in ents])
            rows, cols = linear_sum_assignment(cost)
            good = {int(i) for i, j in zip(rows, cols) if cost[i, j] <= .75}
        expected += len(ents)
        matched += len(good)
        for i, e in enumerate(ents):
            if i in good:
                continue
            nearest = min(tracks, key=lambda d: math.dist((e['x'], e['y']), (d['x'], d['y'])), default=None)
            distance = math.dist((e['x'], e['y']), (nearest['x'], nearest['y'])) if nearest else None
            reason = ('no_track' if nearest is None else 'position_or_missing_track' if distance > .75
                      else 'assignment_competition' if nearest['state'] == 'seen' else nearest['state'])
            counts[reason] += 1
            entities[str(e['id'])] += 1
            misses.append({'t': w['t'], 'entity': e['id'], 'reason': reason,
                           'nearest_distance_m': distance, 'nearest_track': nearest})
    intervals = []
    for entity in entities:
        current = None
        for miss in (m for m in misses if str(m['entity']) == entity):
            if current is None or miss['t'] - current['end'] > .21:
                current = {'entity': entity, 'start': miss['t'], 'end': miss['t'], 'samples': 0}
                intervals.append(current)
            current['end'] = miss['t']
            current['samples'] += 1
    result = {'version': 'recall-attribution-v1', 'run': a.run,
              'scorer_definition': 'floorsight-e2e-legacy8-v1 placeable visibility, seen-only Hungarian match within 0.75m',
              'limitation': 'Nearest track attribution is diagnostic, not identity ground truth. No scorer or production changes.',
              'expected': expected, 'matched': matched, 'recall': matched/expected,
              'miss_reasons': dict(counts), 'miss_entities': dict(entities),
              'longest_intervals': sorted(intervals, key=lambda x: x['samples'], reverse=True)[:20],
              'misses': misses}
    out = Path('eval/results')/(a.run+'_recall_attribution_v1.json')
    with out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != 'misses'}, indent=2))


if __name__ == '__main__':
    main()
