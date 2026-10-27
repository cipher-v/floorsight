"""Behaviour checks on geometry-labelled camera evidence, without cloud calls."""
import pytest
from scripts.benchmark.agent_eval import evaluate, scene, run_case
from floorsight.agent.planner import Action, guard


def test_agent_reduces_false_flags_without_rejecting_hazards():
    # Different scene offsets/gaps from the published development receipt.
    result = evaluate(variants=3, seed=739391)
    m = result['metrics']
    assert m['hazards_rejected'] == 0
    assert 0 < m['false_flags_removed'] < m['baseline_false_flags']
    assert m['actual_alerts_before_approval'] == 0
    assert m['recommendations']['uncertain'] > 0
    assert all(r['pending_approvals'] == 1 for r in result['cases'])
    exits = result['blocked_exit']['metrics']
    assert exits['agent_false_alerts_visible'] == 0 < exits['baseline_false_alerts_visible']
    assert exits['hazards_alerted_visible'] == exits['hazards_visible']
    assert exits['blind_cases_pending_review'] == exits['blind_cases']
    assert exits['hazards_rejected_all'] == 0


@pytest.mark.parametrize('family', ['hazard_heading_noise', 'hazard_closer_than_tracker', 'hazard_missing_second_look'])
def test_bad_tracking_or_missing_evidence_does_not_establish_safety(family):
    obs, label = scene(family, 735)
    agent, case = run_case(obs)
    assert label['hazard']
    assert case.facts['recommendation']['verdict'] == 'uncertain'
    view = agent._view(case, 'new', [])
    assert not guard(view, Action('recommend', {'verdict': 'reject'}))[0]
    assert not guard(view, Action('recommend', {'verdict': 'confirm'}))[0]
    assert not agent.alerts


def test_reviewer_can_override_recommendation_and_duplicate_decisions_do_not_alert_twice():
    obs, _ = scene('clear_heading_noise', 735)
    agent, case = run_case(obs)
    assert case.facts['recommendation']['verdict'] == 'reject'
    ap = next(iter(agent.approvals.values()))
    agent.decide(ap.approval_id, True, agent.t + 1, reviewer='test reviewer')
    assert len(agent.alerts) == 1
    agent.decide(ap.approval_id, True, agent.t + 2, reviewer='test reviewer')
    assert len(agent.alerts) == 1
