"""End-of-shift report built from the agent's events, approvals and trace."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from floorsight.contracts import ApprovalStatus, EventType
from floorsight.site import zone_by_id

if TYPE_CHECKING:
    from floorsight.agent.runtime import Case, FloorsightAgent

HOTSPOT_RADIUS_M = 3.0


def _refs(case: Case) -> dict:
    ev = case.event
    return {
        "event_ids": [ev.event_id],
        "evidence_refs": [e.image_ref for e in ev.evidence if e.image_ref],
        "evidence": [{"kind": e.kind, "camera_id": e.camera_id, "t": e.t, "supports": e.supports, "note": e.note} for e in ev.evidence],
    }


def build_shift_report(agent: FloorsightAgent) -> dict:
    cases = list(agent.cases.values())
    now = agent.t

    counts: dict[str, dict[str, int]] = {}
    for c in cases:
        row = counts.setdefault(c.event.type.value, {"total": 0, "confirmed": 0, "rejected": 0, "resolved": 0, "open": 0})
        row["total"] += 1
        st = c.event.status.value
        if c.was_confirmed:
            row["confirmed"] += 1
        if st == "rejected":
            row["rejected"] += 1
        elif st == "resolved":
            row["resolved"] += 1
        else:
            row["open"] += 1

    # near-miss hotspots: greedy clustering of confirmed near-misses
    hotspots: list[dict] = []
    for c in sorted((c for c in cases if c.event.type == EventType.near_miss and c.was_confirmed), key=lambda c: c.event.t_detect):
        loc = c.event.location
        spot = next((h for h in hotspots if math.dist(h["location"], loc) <= HOTSPOT_RADIUS_M), None)
        if spot is None:
            zone = agent._hotspot_zone(loc)
            spot = {"location": list(loc), "zone_id": zone.zone_id if zone else None, "zone_name": zone.name if zone else None, "count": 0, "event_ids": [], "evidence_refs": []}
            hotspots.append(spot)
        spot["count"] += 1
        spot["event_ids"].append(c.event.event_id)
        spot["evidence_refs"] += _refs(c)["evidence_refs"]
    hotspots.sort(key=lambda h: -h["count"])
    for h in hotspots:
        prop = next((p for p in agent.proposals if math.dist(p["xy"], h["location"]) <= HOTSPOT_RADIUS_M), None)
        h["fix_proposed"] = prop["rule_id"] if prop else None
        h["fix_added"] = bool(prop and agent.engine.get_rule(prop["rule_id"]))

    # blocked exits (confirmed incidents only; rejected false alarms don't count)
    blocked: dict[str, float] = {}
    blocked_items = []
    for c in cases:
        if c.event.type != EventType.blocked_exit or not c.was_confirmed:
            continue
        end = c.t_end if c.t_end is not None else now
        secs = max(0.0, end - c.event.t_start)
        blocked[c.event.zone_id or "?"] = blocked.get(c.event.zone_id or "?", 0.0) + secs
        blocked_items.append({"t_start": c.event.t_start, "t_end": c.t_end, "status": c.event.status.value, **_refs(c)})
    blocked_minutes = {z: round(s / 60.0, 2) for z, s in blocked.items()}

    false_alarms = [
        {
            "event_id": c.event.event_id,
            "type": c.event.type.value,
            "t": c.event.t_detect,
            "zone_id": c.event.zone_id,
            "camera_ids": c.detect_cams,
            "reason": next((s.reason for s in reversed(agent.trace) if s.event_id == c.event.event_id and s.tool == "reject_event"), ""),
            **_refs(c),
        }
        for c in cases
        if c.event.status.value == "rejected"
    ]

    aps = list(agent.approvals.values())
    approvals = {
        "asked": len(aps),
        "answered": sum(a.status in (ApprovalStatus.approved, ApprovalStatus.rejected) for a in aps),
        "expired": sum(a.status == ApprovalStatus.expired for a in aps),
        "approved": sum(a.status == ApprovalStatus.approved for a in aps),
        "rejected": sum(a.status == ApprovalStatus.rejected for a in aps),
        "pending": sum(a.status == ApprovalStatus.pending for a in aps),
        "by_action": {},
        "items": [
            {"approval_id": a.approval_id, "action": a.action, "event_id": a.event_id, "status": a.status.value, "created_t": a.created_t, "decided_t": a.decided_t, "summary": a.summary}
            for a in aps
        ],
    }
    for a in aps:
        approvals["by_action"][a.action] = approvals["by_action"].get(a.action, 0) + 1

    open_items = [
        {"kind": "event", "event_id": c.event.event_id, "type": c.event.type.value, "status": c.event.status.value, "summary": c.event.summary, **_refs(c)}
        for c in cases
        if c.event.status.value not in ("resolved", "rejected")
    ] + [{"kind": "approval", "approval_id": a.approval_id, "action": a.action, "event_id": a.event_id, "summary": a.summary} for a in aps if a.status == ApprovalStatus.pending]

    exit_b = zone_by_id(agent.site, "exit_b")
    return {
        "t_start": agent.t_first,
        "t_end": now,
        "mode": getattr(agent.planner, "mode", "local"),
        "counts_by_type": counts,
        "near_miss_hotspots": hotspots,
        "blocked_minutes_by_zone": blocked_minutes,
        "exit_b_blocked_minutes": blocked_minutes.get("exit_b", 0.0) if exit_b else None,
        "blocked_incidents": blocked_items,
        "false_alarms_rejected": false_alarms,
        "remembered_spots": [{"zone_id": s.zone_id, "xy": list(s.xy), "camera_ids": s.camera_ids, "extra_s": s.extra_s, "event_id": s.event_id} for s in agent.engine.spots],
        "approvals": approvals,
        "alerts": {"total": len(agent.alerts), "floor_lead": sum(a.audience == "floor_lead" for a in agent.alerts), "manager": sum(a.audience == "manager" for a in agent.alerts)},
        "rules_added": [{"rule_id": r.rule_id, "source": r.source, "text": r.text, "enabled": r.enabled} for r in agent.engine.rules if r.source != "default"],
        "open_items": open_items,
        "trace_steps": len(agent.trace),
    }
