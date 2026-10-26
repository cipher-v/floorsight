"""GCP-generated static review of actual saved decisions, not a mock dashboard."""
import argparse
import html
import json
import os
from pathlib import Path


def main():
    if os.environ.get('FLOORSIGHT_GCP_LEASE_CONFIRMED') != 'yes':
        raise RuntimeError('GCP lease required')
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    root = Path.cwd()
    state_path = root/'data/runtime'/args.run/'state.json'
    agent = json.loads(state_path.read_text())['agent']
    out = root/'eval/reviews'/(args.run+'_decisions')
    out.mkdir(exist_ok=False)
    esc = html.escape
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Floorsight decision evidence</title>',
            '<style>body{background:#101820;color:#eff6fa;font:16px/1.5 system-ui;margin:0}main{max-width:1100px;margin:36px auto;padding:0 24px}h1{font-size:34px}section{background:#1a2936;padding:24px;margin:24px 0;border-left:4px solid #71d8d0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#101820;padding:16px}summary{cursor:pointer}small{color:#b7cad6}.warning{color:#ffd494}a{color:#9acfff}</style><main>',
            '<p>PRIVATE · ACTUAL SAVED AGENT TRACE · SIMULATED FOOTAGE</p><h1>What the agent confirmed, and why</h1>',
            f'<p>Run <strong>{esc(args.run)}</strong>. This static review reads saved event and tool outputs. It is not a live dashboard or proof of real-camera safety.</p>',
            '<p class="warning">A rejected event means insufficient evidence under this policy; it does not prove the scene was safe. Second-view presence is not an independent clearance measurement.</p>']
    rows = []
    for event in agent['events']:
        if event['type'] != 'near_miss':
            continue
        steps = [s for s in agent['trace'] if s['event_id']==event['event_id']]
        confirmed = any(s['tool']=='confirm_event' and s['tool_output'].get('status')=='confirmed' for s in steps)
        row = {'event':event,'explicitly_confirmed':confirmed,'steps':steps}
        rows.append(row)
        page.append(f'<section><h2>{event["t_detect"]:.1f}s · {esc(event["event_id"])}</h2><p>Final state: <strong>{esc(event["status"])}</strong> · Explicit confirmation: <strong>{str(confirmed).lower()}</strong></p><p>{esc(event["summary"])}</p>')
        for step in steps:
            if step['tool'] in ('check_trajectory','second_view','sharper_crop','confirm_event','reject_event'):
                page.append(f'<details><summary>{step["t"]:.1f}s · {esc(step["tool"])} — {esc(step["decision"])}</summary><pre>{esc(json.dumps(step,indent=2))}</pre></details>')
        page.append('</section>')
    page.append(f'<p><a href="../{esc(args.run)}/index.html">Source camera overlays</a> · <a href="manifest.json">Machine-readable trace and exact source path</a></p></main></html>')
    (out/'index.html').write_text('\n'.join(page),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps({'version':'decision-review-v1','run':args.run,'source_path':str(state_path),'representation':'static2Ddecisionreview','events':rows},indent=2),encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
