"""A notifier failure (SNS outage) must not lose an alert: the dashboard still gets it, the audience counts as alerted
only once delivery succeeds, and the retry re-sends the same alert instead of recording a second one."""

from collections import Counter

from floorsight.agent.runtime import ALERT_RETRY_S, FloorsightAgent
from floorsight.fake import FakeFeed, FakeVision


class FlakyNotifier:
    """Raises on the first `fail` calls, then delivers."""

    def __init__(self, fail: int):
        self.fail = fail
        self.calls: list[str] = []  # alert ids, every attempt
        self.sent: list[str] = []  # alert ids, delivered

    def send(self, alert):
        self.calls.append(alert.alert_id)
        if len(self.calls) <= self.fail:
            raise ConnectionError("SNS unreachable")
        self.sent.append(alert.alert_id)


def _drive(sim_dir, notifier):
    """floorsight.agent.demo.run (every approval auto-approved), with a notifier."""
    feed = FakeFeed(sim_dir)
    vision = FakeVision(feed)
    agent = FloorsightAgent(feed.site, vision, notifier=notifier)
    log = []

    def handle(msgs, t):
        queue = list(msgs)
        while queue:
            m = queue.pop(0)
            log.append(m)
            if m.kind == "approval" and m.data["status"] == "pending":
                queue += agent.decide(m.data["approval_id"], True, t)

    for packets in feed:
        world = vision.process(packets)
        handle(agent.on_world(world), world.t)
    return agent, log


def _key(a):
    return (a.alert_id, a.event_id, a.audience, a.message)


def test_failed_send_is_retried_with_the_same_alert_and_delivered_once(sim_dir, full_run, monkeypatch):
    monkeypatch.delenv("FLOORSIGHT_LLM", raising=False)
    clean, _ = full_run
    notifier = FlakyNotifier(fail=1)
    agent, log = _drive(sim_dir, notifier)

    assert clean.alerts and [_key(a) for a in agent.alerts] == [_key(a) for a in clean.alerts]  # no extra alert records
    first = agent.alerts[0].alert_id
    assert notifier.calls[:2] == [first, first]  # the failed alert was sent again under the same id
    assert sorted(notifier.sent) == sorted(a.alert_id for a in agent.alerts)  # every alert delivered exactly once
    assert len(notifier.sent) == len(set(notifier.sent))
    # the dashboard got every alert once, the one whose delivery failed included
    assert [m.data["alert_id"] for m in log if m.kind == "alert"] == [a.alert_id for a in agent.alerts]
    for c in agent.cases.values():
        assert not c.undelivered
        assert c.alerted == {a.audience for a in agent.alerts if a.event_id == c.event.event_id}

    failed = next(s for s in agent.trace if s.tool == "send_alert" and s.tool_output.get("delivered") is False)
    assert failed.tool_output["alert"]["alert_id"] == first and "will retry" in failed.decision
    redo = next(s for s in agent.trace if s.tool == "deliver_alert")
    assert redo.tool_input["alert_id"] == first and redo.tool_output == {"delivered": True}
    assert redo.t >= failed.t + ALERT_RETRY_S
    assert agent.snapshot()["alerts_undelivered"] == []


def test_alerts_stay_on_the_dashboard_and_queued_while_the_notifier_is_down(sim_dir, full_run, monkeypatch):
    monkeypatch.delenv("FLOORSIGHT_LLM", raising=False)
    clean, _ = full_run
    notifier = FlakyNotifier(fail=10**9)
    agent, log = _drive(sim_dir, notifier)

    ids = [a.alert_id for a in agent.alerts]
    assert [_key(a) for a in agent.alerts] == [_key(a) for a in clean.alerts]  # same decisions, no duplicates
    assert notifier.sent == []
    assert [m.data["alert_id"] for m in log if m.kind == "alert"] == ids
    assert all(not c.alerted for c in agent.cases.values())  # nobody counts as alerted: nothing was delivered
    assert agent.snapshot()["alerts_undelivered"] == sorted(ids)
    # retried on a timer, not on every frame (180 s shift at 10 fps = 1800 frames)
    assert max(Counter(notifier.calls).values()) <= 180 / ALERT_RETRY_S + 2
