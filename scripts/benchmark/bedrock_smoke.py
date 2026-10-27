"""Opt-in live Bedrock agent check; synthetic pixels only, no notifications or approvals."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.benchmark.agent_eval import scene, run_case
from floorsight.llm.bedrock import BedrockPlanner, make_bedrock_client, MODEL_ID


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', default=MODEL_ID)
    p.add_argument('--region', default='us-east-1')
    p.add_argument('--cases', type=int, choices=[1, 4], default=1)
    a = p.parse_args()
    client = make_bedrock_client(a.region).with_options(timeout=30, max_retries=0)
    results = []
    for family in ['hazard_stable', 'clear_heading_noise', 'hazard_closer_than_tracker', 'hazard_missing_second_look'][:a.cases]:
        planner = BedrockPlanner(client=client, model=a.model, region=a.region, max_tokens=2048)
        obs, truth = scene(family, 20261028)
        start = time.perf_counter()
        agent, case = run_case(obs, planner)
        steps = [s.model_dump(mode='json') for s in agent.trace]
        results.append({'family': family, 'label': truth, 'calls': planner.calls,
            'fallbacks': planner.fallbacks, 'elapsed_s': time.perf_counter()-start,
            'recommendation': case.facts.get('recommendation'), 'alerts': len(agent.alerts),
            'approvals_pending': sum(x.status.value == 'pending' for x in agent.approvals.values()),
            'steps': steps})
        # Access denied should not cause another full set of repeated calls.
        if not any(s['mode'] == 'bedrock' for s in steps):
            break
    success = len(results) == a.cases and all(any(s['mode'] == 'bedrock' for s in r['steps'])
        and r['recommendation'] and r['approvals_pending'] == 1 and r['alerts'] == 0 for r in results)
    report = {'schema': 'floorsight.bedrock-live.v1', 'model': a.model, 'region': a.region,
        'live': True, 'success': success, 'scope': 'Synthetic tool evidence; real Bedrock calls; no human review or delivery.',
        'cases': results}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['model','success']}, indent=2))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
