from types import SimpleNamespace
from floorsight.agent.runtime import FloorsightAgent
from floorsight.contracts import EventType, EventStatus


def test_unconfirmed_near_miss_does_not_claim_false_alarm_or_safety():
    event = SimpleNamespace(type=EventType.near_miss, status=None, zone_id=None)
    case = SimpleNamespace(event=event, t_end=None)
    out, decision = FloorsightAgent._do_reject_event(SimpleNamespace(t=12.), case, None, [])
    assert event.status == EventStatus.rejected
    assert case.t_end == 12.
    assert out['reason_category'] == 'insufficient_evidence'
    assert out['safety_conclusion'] == 'unknown'
    assert 'false alarm' not in decision
    assert 'safety remains unknown' in decision
