"""Offline diagnostic on product observations only; never tunes against truth."""
import argparse
import json
import math
import os
from pathlib import Path
from floorsight.site import box_distance, FORKLIFT_LENGTH_M, FORKLIFT_WIDTH_M


def main():
    if os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED') != 'yes':
        raise RuntimeError('Run through the assigned GCP lease')
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    worlds = [json.loads(line)['world'] for line in
              (Path('data/runtime') / args.run / 'worlds.jsonl').read_text().splitlines()]
    rows = []
    for target in (47.3, 55.9, 108.5, 149.3):
        w = min(worlds, key=lambda w: abs(w['t'] - target))
        for f in (x for x in w['tracks'] if x['cls'] == 'forklift'):
            previous = [(old['t'], tr) for old in worlds if w['t']-.9 <= old['t'] <= w['t']-.5
                        for tr in old['tracks'] if tr['track_id'] == f['track_id'] and tr['state']=='seen']
            heading = math.atan2(f['vy'], f['vx'])
            history_heading = None
            baseline = None
            if previous:
                old_t, old = previous[0]
                dx, dy = f['x']-old['x'], f['y']-old['y']
                baseline = math.hypot(dx,dy)
                if baseline >= .2:
                    history_heading = math.atan2(dy,dx)
            people = []
            for p in w['tracks']:
                if p['cls'] != 'person' or math.hypot(p['x']-f['x'],p['y']-f['y']) > 3:
                    continue
                def gap(h):
                    return box_distance(p['x'],p['y'],f['x'],f['y'],h,FORKLIFT_LENGTH_M,FORKLIFT_WIDTH_M)
                people.append({'track':p, 'instant_gap_m':gap(heading),
                               'history_gap_m':gap(history_heading) if history_heading is not None else None})
            rows.append({'t':w['t'], 'forklift':f, 'instant_heading_deg':math.degrees(heading),
                         'history_heading_deg':math.degrees(history_heading) if history_heading is not None else None,
                         'history_baseline_m':baseline, 'people':people})
    out = Path('eval/results') / (args.run+'_heading_audit_v1.json')
    with out.open('x', encoding='utf-8') as stream:
        json.dump({'version':'product-heading-diagnostic-v1','run':args.run,'rows':rows},stream,indent=2)
    print(json.dumps(rows,indent=2))


if __name__ == '__main__':
    main()
