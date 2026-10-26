"""Print the agent's decision trace for a simulated shift (approvals auto-approved).

    python -m floorsight.agent.demo --sim-dir data/sim/demo_001
    python -m floorsight.agent.demo --sim-dir data/sim/demo_001 --rule "tell me when Exit B is blocked for 30 seconds"
"""

from __future__ import annotations

import argparse
import json

from floorsight.agent.runtime import FloorsightAgent
from floorsight.contracts import LiveMessage
from floorsight.fake import FakeFeed, FakeVision


def run(sim_dir: str, rules: list[str] | None = None, auto_approve: bool = True, verbose: bool = False, planner=None):
    """Drive the agent over FakeFeed/FakeVision. Returns (agent, all LiveMessages)."""
    feed = FakeFeed(sim_dir)
    vision = FakeVision(feed)
    agent = FloorsightAgent(feed.site, vision, planner=planner)
    log: list[LiveMessage] = []

    def handle(msgs: list[LiveMessage], t: float):
        queue = list(msgs)
        while queue:
            m = queue.pop(0)
            log.append(m)
            if verbose:
                print_message(m)
            if auto_approve and m.kind == "approval" and m.data["status"] == "pending":
                queue += agent.decide(m.data["approval_id"], True, t)

    for text in rules or []:
        handle(agent.add_rule_from_text(text, 0.0), 0.0)
    for packets in feed:
        world = vision.process(packets)
        handle(agent.on_world(world), world.t)
    return agent, log


def print_message(m: LiveMessage) -> None:
    d = m.data
    if m.kind == "agent_step":
        print(f"  t={d['t']:6.1f}  {d['event_id']:<10} #{d['step']:<2} {d['tool']:<18} -> {d['decision']}")
        if d["reason"]:
            print(f"  {'':8}  {'':10}     why: {d['reason']}")
    elif m.kind == "event":
        pass
    elif m.kind == "approval":
        verb = "ASK " if d["status"] == "pending" else d["status"].upper()
        print(f"  {'':8}  {d.get('event_id') or '-':<10}     [{verb}] {d['approval_id']} {d['action']}: {d['summary']}")
    elif m.kind == "alert":
        print(f"  {'':8}  {d['event_id']:<10}     [ALERT -> {d['audience']}] {d['message']}")
    elif m.kind == "rule":
        if d.get("ok") is False:
            print(f"  [RULE] not created: {d['message']}")
        else:
            print(f"  [RULE] {d['rule_id']} ({d['source']}, enabled={d['enabled']}): {d['text']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", default="data/sim/demo_001")
    ap.add_argument("--rule", action="append", default=[], help="plain-language rule to add at t=0 (repeatable)")
    ap.add_argument("--no-approve", action="store_true", help="leave approvals pending")
    ap.add_argument("--report", action="store_true", help="print the full shift report as JSON")
    a = ap.parse_args()
    print(f"Floorsight agent trace for {a.sim_dir}\n")
    agent, _ = run(a.sim_dir, a.rule, auto_approve=not a.no_approve, verbose=True)
    rep = agent.shift_report()
    print("\nEvents:")
    for c in agent.cases.values():
        e = c.event
        print(f"  {e.event_id}  {e.type.value:<15} t={e.t_detect:6.1f}  {e.status.value:<10} {e.summary}")
    print("\nShift report:")
    print(f"  counts:            {json.dumps(rep['counts_by_type'])}")
    print(f"  hotspots:          {[(h['zone_name'], h['count']) for h in rep['near_miss_hotspots']]}")
    print(f"  Exit B blocked:    {rep['exit_b_blocked_minutes']} min")
    print(f"  false alarms:      {[(f['event_id'], f['camera_ids']) for f in rep['false_alarms_rejected']]}")
    ap_ = rep["approvals"]
    print(f"  approvals:         asked {ap_['asked']}, answered {ap_['answered']} ({ap_['by_action']})")
    print(f"  alerts:            {rep['alerts']}")
    print(f"  rules added:       {[r['rule_id'] + ': ' + r['text'] for r in rep['rules_added']]}")
    print(f"  open items:        {[(o.get('event_id'), o.get('type') or o.get('action')) for o in rep['open_items']]}")
    if a.report:
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
