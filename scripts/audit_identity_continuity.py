"""Product-only identity audit; no evaluation oracle imports."""
import argparse
import json
import os
from pathlib import Path


def main():
    if os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED') != 'yes':
        raise RuntimeError('GCP lease required')
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    worlds = [json.loads(line)['world'] for line in
              (Path('data/runtime')/args.run/'worlds.jsonl').read_text().splitlines()]
    previous, transitions = {}, []
    for world in worlds:
        for tr in world['tracks']:
            prior = previous.get(tr['track_id'])
            if prior and prior['track']['cls'] != tr['cls']:
                transitions.append({'t':world['t'], 'before':prior, 'after':tr})
            previous[tr['track_id']] = {'t':world['t'], 'track':tr}
    data = {'version':'identity-continuity-v1','run':args.run,
            'class_transition_count':len(transitions),'transitions':transitions,
            'definition':'Class changes within one published track ID. Product observations only; not a truth-matched identity-switch score.'}
    with (Path('eval/results')/(args.run+'_identity_v1.json')).open('x',encoding='utf-8') as stream:
        json.dump(data,stream,indent=2)
    print(json.dumps(data,indent=2))


if __name__ == '__main__':
    main()
