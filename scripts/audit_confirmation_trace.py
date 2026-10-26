"""Supplement frozen incident score with explicit confirmation evidence."""
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
    agent = json.loads((Path('data/runtime')/args.run/'state.json').read_text())['agent']
    rows = []
    for event in agent['events']:
        steps = [s for s in agent['trace'] if s['event_id']==event['event_id']]
        confirms = [s for s in steps if s['tool']=='confirm_event' and s['tool_output'].get('status')=='confirmed']
        rows.append({'event_id':event['event_id'],'type':event['type'],'t_detect':event['t_detect'],
                     'final_status':event['status'],'explicit_confirmation':bool(confirms),
                     'confirmation_times':[s['t'] for s in confirms],
                     'verification_steps':[s for s in steps if s['tool'] in ('check_trajectory','second_view','sharper_crop','confirm_event','reject_event')]})
    data = {'audit_version':'explicit-confirmation-trace-v1','run':args.run,
            'definition':'Successful confirm_event tool output required. Final resolved status alone is not confirmation. Legacy scorer unchanged.',
            'near_miss_detected':sum(r['type']=='near_miss' for r in rows),
            'near_miss_explicitly_confirmed':sum(r['type']=='near_miss' and r['explicit_confirmation'] for r in rows),
            'events':rows}
    with (Path('eval/results')/(args.run+'_confirmation_trace_v1.json')).open('x',encoding='utf-8') as stream:
        json.dump(data,stream,indent=2)
    print(json.dumps({k:v for k,v in data.items() if k!='events'},indent=2))
    for r in rows:
        print(json.dumps({k:v for k,v in r.items() if k!='verification_steps'}))


if __name__ == '__main__':
    main()
