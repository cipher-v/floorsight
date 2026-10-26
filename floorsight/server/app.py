"""Floorsight HTTP + websocket API and the dashboard.

    GET  /                          dashboard (web/index.html)
    GET  /api/site                  {"site": SitePlan, "camera_views": {...}}
    GET  /api/state                 agent snapshot + cameras + world + metrics
    GET  /api/coverage              CoverageMap
    POST /api/rules                 {"text"} -> agent.add_rule_from_text
    POST /api/approvals/{id}        {"approved": bool} -> agent.decide
    GET  /api/evidence/{ref}        blurred evidence JPEG
    GET  /api/cameras/{id}/frame.jpg latest blurred camera thumbnail
    GET  /api/replay?t0=&t1=        world snapshots (2 Hz) in a time range
    GET  /api/report                shift report
    GET  /api/metrics               fps, latency, dropped frames ...
    GET  /healthz                   liveness (never needs the token)
    WS   /ws                        LiveMessage JSON stream

If the env var FLOORSIGHT_TOKEN is set, every route except /healthz needs it,
as ``?token=``, the ``floorsight_token`` cookie, or ``Authorization: Bearer``.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from http.cookies import SimpleCookie
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from floorsight.server.runtime import Hub, Runtime, RuntimeConfig, to_jsonable

log = logging.getLogger("floorsight.server.app")

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "web")
TOKEN_COOKIE = "floorsight_token"
OPEN_PATHS = {"/healthz"}


class RuleIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class DecisionIn(BaseModel):
    approved: bool
    reviewer: str | None = Field(default=None, max_length=80, description="Who decided (shown in the saved trace)")


# --------------------------------------------------------------------------- #
# Optional judge token (pure ASGI so it also guards the websocket)
# --------------------------------------------------------------------------- #


_LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Floorsight - access</title>
<style>body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#0e131a;color:#e9eef4;
font:17px system-ui,sans-serif}}form{{background:#161e28;padding:32px;border-radius:12px;width:min(92vw,380px)}}
h1{{font-size:22px;margin:0 0 6px}}p{{color:#aab6c3;margin:0 0 18px}}label{{display:block;margin-bottom:6px}}
input{{width:100%;box-sizing:border-box;padding:12px;font-size:17px;border-radius:8px;border:1px solid #3a4756;
background:#0e131a;color:#e9eef4}}button{{margin-top:14px;width:100%;padding:12px;font-size:17px;border:0;
border-radius:8px;background:#5aa9ff;color:#06111f;font-weight:700;cursor:pointer}}
input:focus,button:focus{{outline:3px solid #ffd166;outline-offset:2px}}.err{{color:#ff8a8a}}</style></head>
<body><main><form method="get" action="/"><h1>Floorsight</h1><p>This demo is protected. Enter the access token you were given.</p>
{error}<label for="token">Access token</label><input id="token" name="token" type="password" autocomplete="current-password" required autofocus>
<button type="submit">Open dashboard</button></form></main></body></html>"""


class TokenGuard:
    def __init__(self, app, token: str):
        self.app = app
        self.token = token

    def _extract(self, scope) -> tuple[str | None, bool]:
        qs = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        if qs.get("token"):
            return qs["token"][0], True
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip(), False
        if "cookie" in headers:
            jar = SimpleCookie()
            try:
                jar.load(headers["cookie"])
            except Exception:
                return None, False
            if TOKEN_COOKIE in jar:
                return jar[TOKEN_COOKIE].value, False
        return None, False

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or scope.get("path") in OPEN_PATHS:
            return await self.app(scope, receive, send)
        given, from_query = self._extract(scope)
        ok = given is not None and hmac.compare_digest(given.encode(), self.token.encode())
        if not ok:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4401})
                return
            if scope.get("path") == "/" and scope.get("method") == "GET":
                err = '<p class="err" role="alert">That token is not right. Try again.</p>' if given else ""
                resp = Response(_LOGIN_PAGE.format(error=err), status_code=401, media_type="text/html")
            else:
                resp = JSONResponse({"detail": "access token required"}, status_code=401)
            return await resp(scope, receive, send)
        if not from_query or scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_with_cookie(message):
            if message["type"] == "http.response.start":
                cookie = f"{TOKEN_COOKIE}={self.token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=86400"
                message = {**message, "headers": list(message.get("headers", [])) + [(b"set-cookie", cookie.encode())]}
            await send(message)

        return await self.app(scope, receive, send_with_cookie)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #


def create_app(cfg: RuntimeConfig | None = None, runtime: Runtime | None = None, token: str | None = None, origin_secret: str | None = None) -> FastAPI:
    rt = runtime or Runtime(cfg or RuntimeConfig(), hub=Hub())
    token = os.environ.get("FLOORSIGHT_TOKEN") if token is None else token

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        rt.hub.attach_loop(asyncio.get_running_loop())
        rt.start()
        try:
            yield
        finally:
            rt.stop()

    app = FastAPI(title="Floorsight", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.runtime = rt
    no_store = {"Cache-Control": "no-store"}

    @app.middleware("http")
    async def revalidate_static(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"  # always revalidate dashboard assets
        return response

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "running": rt.running, "mode": rt.mode, "t": rt.current_t()}

    @app.get("/")
    def index():
        path = os.path.join(WEB_DIR, "index.html")
        if not os.path.exists(path):
            return Response("<h1>Floorsight</h1><p>web/index.html is missing.</p>", media_type="text/html")
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    @app.get("/api/site")
    def site():
        return {"site": to_jsonable(rt.site), "camera_views": rt.camera_views, "privacy": rt.privacy()}

    @app.get("/api/state")
    def state():
        return JSONResponse(rt.state(), headers=no_store)

    @app.get("/api/coverage")
    def coverage():
        if rt.coverage is None:
            raise HTTPException(503, "coverage not available yet")
        return rt.coverage

    @app.post("/api/rules")
    def add_rule(body: RuleIn):
        text = body.text.strip()
        if not text:
            raise HTTPException(422, "rule text is empty")
        try:
            msgs = rt.add_rule(text)
        except Exception as exc:
            log.exception("add_rule failed")
            raise HTTPException(500, f"agent could not add the rule: {exc}") from exc
        return {"ok": True, "agent": rt.agent_label, "messages": msgs}

    @app.post("/api/approvals/{approval_id}")
    def decide(approval_id: str, body: DecisionIn):
        try:
            msgs = rt.decide(approval_id, body.approved, reviewer=body.reviewer)
        except KeyError as exc:
            raise HTTPException(404, f"unknown approval {approval_id}") from exc
        except Exception as exc:
            log.exception("decide failed")
            raise HTTPException(500, f"agent could not record the decision: {exc}") from exc
        return {"ok": True, "agent": rt.agent_label, "approval_id": approval_id, "approved": body.approved, "messages": msgs}

    @app.get("/api/trace")
    def trace():
        path = rt.save_snapshot("api request")
        return JSONResponse({"trace_path": rt.trace_path, "snapshot_path": path, "agent": rt.agent_snapshot()}, headers=no_store)

    @app.get("/api/evidence/{ref}")
    def evidence(ref: str):
        data = rt.evidence.get(ref)
        if data is None:
            raise HTTPException(404, "evidence not found (it may have been auto-deleted)")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/cameras/{camera_id}/frame.jpg")
    def camera_frame(camera_id: str):
        if camera_id not in {c.camera_id for c in rt.site.cameras}:
            raise HTTPException(404, "unknown camera")
        data = rt.thumbnail(camera_id)
        if data is None:
            return Response(status_code=204, headers=no_store)  # no frame yet (or a feed without video)
        return Response(data, media_type="image/jpeg", headers=no_store)

    @app.get("/api/replay")
    def replay(t0: float | None = Query(None), t1: float | None = Query(None)):
        return JSONResponse(rt.replay(t0, t1), headers=no_store)

    @app.get("/api/report")
    def report():
        return JSONResponse(build_report(rt), headers=no_store)

    @app.get("/api/metrics")
    def metrics():
        return JSONResponse(rt.metrics_snapshot(), headers=no_store)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        q = rt.hub.register()
        try:
            while True:
                text = await q.get()
                await websocket.send_text(text)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            rt.hub.unregister(q)

    if os.path.isdir(WEB_DIR):
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    guarded = TokenGuard(app, token) if token else app
    origin_secret = os.environ.get("FLOORSIGHT_ORIGIN_SECRET", "") if origin_secret is None else origin_secret
    if origin_secret:  # public mode: only requests that came through the CloudFront distribution get in
        return OriginGuard(guarded, origin_secret)  # type: ignore[return-value]
    return guarded


class OriginGuard:
    """Public mode (Terraform public_endpoint = true): CloudFront adds X-Floorsight-Origin: <secret> to every request
    it forwards; anything that reaches the port without it (a direct hit on the instance) gets 403. /healthz stays open."""

    def __init__(self, app, secret: str):
        self.app, self.secret = app, secret.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or scope.get("path") in OPEN_PATHS:
            return await self.app(scope, receive, send)
        given = next((v for k, v in scope.get("headers", []) if k.lower() == b"x-floorsight-origin"), b"")
        if hmac.compare_digest(given, self.secret):
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4403})
            return
        return await JSONResponse({"detail": "direct access refused; use the published link"}, status_code=403)(scope, receive, send)


def build_report(rt: Runtime) -> dict:
    """Agent's shift report plus facts the server can count on its own."""
    snap = rt.agent_snapshot()
    events = snap.get("events", [])
    by_status: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for e in events:
        by_status[str(e.get("status", "?"))] = by_status.get(str(e.get("status", "?")), 0) + 1
        by_type[str(e.get("type", "?"))] = by_type.get(str(e.get("type", "?")), 0) + 1
    approvals = snap.get("approvals", [])
    m = rt.metrics_snapshot()
    cams = rt.camera_list()
    return {
        "site": rt.site.name,
        "duration_s": m.get("t", 0.0),
        "agent_mode": rt.agent_label,
        "agent": rt.shift_report(),
        "facts": {
            "events": len(events),
            "events_by_status": by_status,
            "events_by_type": by_type,
            "false_alarms_rejected": by_status.get("rejected", 0),
            "approvals_total": len(approvals),
            "approvals_pending": sum(1 for a in approvals if a.get("status") == "pending"),
            "alerts": len(snap.get("alerts", [])),
            "rules": len(snap.get("rules", [])),
            "frames_processed": m.get("frames", 0),
            "frames_dropped": m.get("dropped", 0),
            "faces_blurred": sum(int(c.get("faces_blurred_total", 0)) for c in cams),
        },
        "cameras": [
            {"camera_id": c["camera_id"], "status": (c.get("health") or {}).get("status", "unknown"),
             "calib_error_m": (c.get("health") or {}).get("calib_error_m")}
            for c in cams
        ],
        "events": [
            {k: e.get(k) for k in ("event_id", "type", "status", "severity", "summary", "t_detect", "zone_id")}
            for e in sorted(events, key=lambda e: e.get("t_detect") or 0)
        ],
        "privacy": rt.privacy(),
    }
