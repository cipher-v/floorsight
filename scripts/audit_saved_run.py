"""Supplementary lifecycle audit; does not change legacy e2e acceptance metrics."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    dest = root / 'eval/results' / (a.run + '_lifecycle_v1.json')
    if dest.exists():
        raise RuntimeError('Preserve earlier audit; choose a new run')
    worlds = [json.loads(x)['world'] for x in (root/'data/runtime'/a.run/'worlds.jsonl').read_text().splitlines()]
    truth = {round(x['t'], 1): x['entities'] for x in map(json.loads, (root/'data/sim/demo_001/truth_frames.jsonl').read_text().splitlines())}
    counts, unmatched = Counter(), Counter()
    max_age = defaultdict(float)
    at_times = {}
    for world in worlds:
        entities = truth.get(round(world['t'], 1), [])
        for tr in world['tracks']:
            if tr['cls'] not in ('person', 'forklift', 'unknown'):
                continue
            state = tr['state']
            counts[state] += 1
            max_age[state] = max(max_age[state], world['t'] - tr['last_seen_t'])
            candidates = [e for e in entities if e['cls'] in ('person', 'forklift')]
            dist = min((math.hypot(tr['x']-e['x'], tr['y']-e['y']) for e in candidates), default=math.inf)
            if dist > .75:
                unmatched[state] += 1
    for t in (55.9, 104., 149.3):
        w = min(worlds, key=lambda w: abs(w['t']-t))
        at_times[str(t)] = {'actual_t': w['t'], 'tracks': w['tracks']}
    state = json.loads((root/'data/runtime'/a.run/'state.json').read_text())
    events = {e['event_id']: e for e in state['agent']['events']}
    stale = [p for p in state['agent']['approvals'] if p['status']=='pending' and p['action']!='create_rule' and events.get(p['event_id'], {}).get('status') in ('resolved','rejected')]
    report = {'audit_version':'floorsight-lifecycle-v1', 'run':a.run, 'observed_update_span_s':worlds[-1]['t']-worlds[0]['t'],
              'state_samples':dict(counts), 'max_observation_age_s':dict(max_age), 'unmatched_samples_by_state':dict(unmatched),
              'definition':'Unmatched means no truth person/forklift within 0.75m. Unseen/estimated samples are hypotheses, not asserted false observed dots. This audit is supplementary and never changes legacy scoring.',
              'pending_operational_approvals_on_closed_events':stale, 'review_times':at_times}
    dest.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='review_times'}, indent=2))


if __name__ == '__main__':
    main()
