"""GCP-only compact observation report, without reading simulator truth."""
import argparse
import json
import math
import os
from pathlib import Path


def main():
    if os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED') != 'yes':
        raise SystemExit('Run only through the assigned GCP runner')
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    a = p.parse_args()
    directory = Path('eval/diagnostics') / a.name
    rows = []
    for target in (47.3,55.9,104.,108.5,149.3):
        record = json.loads((directory / f'{target:.1f}.json').read_text())
        tracks = [t for t in record['world']['tracks'] if t['cls']=='forklift']
        for track in tracks:
            near = [d for d in record['detections'] if math.dist(d['xy'], [track['x'],track['y']]) < 2.]
            rows.append({'t':target, 'track':track,
                         'internal':[t for t in record['internal_tracks'] if t['track_id']==track['track_id']],
                         'nearby_detections':[{'camera_id':d['camera_id'],'xy':d['xy'],'sigma':d['sigma'],
                             'footprint':d['footprint'],'height_m':d['height_m'],'kind':d['kind'],'bbox':d['bbox']} for d in near]})
    output = directory / 'observation-summary-v1.json'
    with output.open('x', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
