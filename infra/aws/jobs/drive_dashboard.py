"""Headless dashboard session for the AWS real-clip run (runs on the instance, never on the owner PC).

Opens the dashboard served on 127.0.0.1, records the screen, and acts as a SCRIPTED STAND-IN reviewer, labelled as
such in every decision ("reviewer=scripted stand-in"):
    card verdict confirm -> Approve      reject -> Reject      uncertain -> left pending
On the first card it opens "Agent's evidence and full reasoning" for a few seconds so the recording shows the replay
strip and the close-up. Writes screenshots, the recording and a session log to OUT. Real human verdicts are separate.
Usage: python drive_dashboard.py URL OUT [--max-s 900] [--width 1600 --height 1000]
"""
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


url, out = sys.argv[1], Path(sys.argv[2])
max_s, W, H = arg("--max-s", 900.0), arg("--width", 1600), arg("--height", 1000)
out.mkdir(parents=True, exist_ok=True)
log = []
REVIEWER = "scripted stand-in"


def note(**k):
    k["wall"] = round(time.time(), 2)
    log.append(k)
    print(json.dumps(k), flush=True)


def safe(aid: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in aid)


with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(out / "video"), record_video_size={"width": W, "height": H})
    note(event="recording_started")  # the recording's t=0 (within ~0.1 s); later wall times map onto it
    page = ctx.new_page()
    page.goto(f"{url}/?reviewer={REVIEWER.replace(' ', '%20')}", wait_until="domcontentloaded")
    page.wait_for_timeout(2500)
    page.screenshot(path=str(out / "dashboard-start.png"))
    t0 = time.time()
    decided, shots, ended_at, opened = set(), 0, None, False
    while time.time() - t0 < max_s:
        st = page.evaluate("async () => { const r = await fetch('/api/state'); return r.json(); }")
        mode = st.get("mode")
        aps = st.get("agent", {}).get("approvals", [])
        for a in [a for a in aps if a.get("status") == "pending" and a.get("action") == "confirm_incident"]:
            aid = a["approval_id"]
            if aid in decided:
                continue
            if aid not in [x.get("card_seen") for x in log]:
                note(card_seen=aid)
            card_data = (a.get("payload") or {}).get("card") or {}
            v = card_data.get("verdict")
            if v is None:  # older payloads: read the summary
                s = a.get("summary", "")
                v = "confirm" if "recommends CONFIRM" in s else "reject" if "recommends REJECT" in s else "uncertain"
            if v == "uncertain":
                if aid not in [x.get("left_pending") for x in log]:
                    note(left_pending=aid, headline=card_data.get("headline"))
                continue
            card = page.locator(f"#ap-{safe(aid)}-verdict").locator("xpath=..")
            try:
                card.wait_for(timeout=8000)
            except Exception as exc:
                note(error=f"card {aid} not on screen: {exc}")
                continue
            page.wait_for_timeout(3500)  # let the recording show the verdict line and the three numbers
            shots += 1
            page.screenshot(path=str(out / f"approval-{shots:02d}-{aid}.png"))
            if not opened:
                card.locator("details.ap-details > summary").click()
                note(evidence_open=aid)
                page.wait_for_timeout(5000)
                page.screenshot(path=str(out / f"approval-{shots:02d}-{aid}-evidence.png"))
                card.locator("details.ap-details > summary").click()
                opened = True
            btn = card.locator(".btn-approve" if v == "confirm" else ".btn-reject")
            try:
                btn.click(timeout=5000)
                decided.add(aid)
                note(decided=aid, verdict=v, headline=card_data.get("headline"), numbers=card_data.get("numbers"), reviewer=REVIEWER)
            except Exception as exc:
                note(error=f"click {aid}: {exc}")
            page.wait_for_timeout(2500)
            page.screenshot(path=str(out / f"after-{shots:02d}-{aid}.png"))
        if mode == "ended":
            if ended_at is None:
                note(event="replay_ended")
            ended_at = ended_at or time.time()
            if time.time() - ended_at > 6:
                break
        page.wait_for_timeout(500)
    page.screenshot(path=str(out / "dashboard-end.png"))
    try:
        page.click("#report-btn", timeout=3000)
        note(event="report_open")
        page.wait_for_timeout(2500)
        page.screenshot(path=str(out / "report.png"), full_page=True)
    except Exception as exc:
        note(error=f"report: {exc}")
    ctx.close()
    browser.close()
(out / "session-log.json").write_text(json.dumps(log, indent=1))
print("SESSION_DONE", len(decided), "decisions")
