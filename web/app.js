/* Floorsight dashboard. Vanilla JS, no build step, works offline.
 *
 * Data: GET /api/site and /api/state on load (and on every reconnect), then the
 * /ws LiveMessage stream ({kind, data}). Every field is treated as optional.
 */
(function () {
  "use strict";

  // ------------------------------------------------------------------ helpers
  const $ = (id) => document.getElementById(id);
  const arr = (v) => (Array.isArray(v) ? v : []);
  const num = (v, d = 0) => (typeof v === "number" && isFinite(v) ? v : d);
  const str = (v, d = "") => (v === null || v === undefined ? d : String(v));
  const obj = (v) => (v && typeof v === "object" && !Array.isArray(v) ? v : {});
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  function el(tag, attrs, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : String(v));
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined || c === false) continue;
      e.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
    }
    return e;
  }

  function fmtT(t) {
    t = Math.max(0, num(t));
    const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
    const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0");
    return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
  }
  const titleCase = (s) => str(s).replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  const sentence = (s) => { s = str(s).replace(/_/g, " "); return s.charAt(0).toUpperCase() + s.slice(1); };

  async function api(path, opts) {
    const o = Object.assign({ credentials: "same-origin", headers: {} }, opts || {});
    if (o.body && typeof o.body !== "string") {
      o.body = JSON.stringify(o.body);
      o.headers["Content-Type"] = "application/json";
    }
    const r = await fetch(path, o);
    let data = null;
    try { data = await r.json(); } catch (_) { data = null; }
    if (!r.ok) {
      const detail = data && data.detail;
      const msg = typeof detail === "string" ? detail : Array.isArray(detail) ? "That input was not accepted." : `Request failed (${r.status})`;
      const err = new Error(msg);
      err.status = r.status;
      err.data = data; // e.g. a 409 carries the server's own record of the approval
      throw err;
    }
    return data;
  }

  // ------------------------------------------------------------------ labels
  const EVENT_LABEL = {
    blocked_exit: "Blocked exit", near_miss: "Near miss", camera_moved: "Camera moved",
    unseen_in_zone: "Someone out of view", dwell: "Stayed too long", floor_warning: "Forklift warning (floor)",
  };
  const STATUS_LABEL = {
    candidate: "Spotted", verifying: "Checking", confirmed: "Confirmed",
    escalated: "Escalated", rejected: "False alarm", resolved: "Resolved",
  };
  const ACTION_LABEL = {
    escalate: "Escalate to the shift manager", create_rule: "Create a new rule", recalibrate: "Re-align a camera",
    dispatch_check: "Send someone to check", confirm_rule: "Turn on a new rule",
    confirm_incident: "Real hazard or false alarm? (your verdict)",
  };
  const CAM_LABEL = { ok: "OK", moved: "Moved", repairing: "Repairing", needs_human: "Needs a person", offline: "Offline" };
  const AUDIENCE_LABEL = { floor_lead: "Floor lead", manager: "Shift manager" };
  const CLASS_LABEL = { person: "Person", forklift: "Forklift", object: "Object", unknown: "Unknown" };
  const ACTIVE = new Set(["candidate", "verifying", "confirmed", "escalated"]);

  const C = {
    floor: "#151d26", grid: "#1d2733", grid5: "#263342", border: "#3a4858",
    person: "#5aaeff", forklift: "#ff8c2e", object: "#c48d5c", unknown: "#c3c9d1",
    exit: "#4cd38a", lane: "#f2c94c", restricted: "#ff6b6b", area: "#8d9bab",
    rack: "#2a3441", rackLine: "#46566a", marker: "#dfe6ee", text: "#ecf1f6", muted: "#a8b4c1",
    cam: { ok: "#5aaeff", moved: "#f5b33d", repairing: "#5aaeff", needs_human: "#ff6b6b", offline: "#9aa6b2" },
    status: { candidate: "#9aa6b2", verifying: "#5aaeff", confirmed: "#f5b33d", escalated: "#ff6b6b", rejected: "#7d8996", resolved: "#4cd38a" },
  };

  // ------------------------------------------------------------------ state
  const S = {
    site: null, views: {}, labelPos: {}, privacy: {}, components: {}, warnings: [],
    world: null, coverage: null, cameras: [], metrics: {}, history: {},
    events: new Map(), approvals: new Map(), rules: new Map(), trace: new Map(),
    trails: new Map(), display: new Map(), headings: new Map(),
    eventSeenAt: new Map(), decidedAt: new Map(),
    mode: "live", ended: false, replay: { t: null, snap: null, trails: new Map(), playing: false },
    coverageOn: false, focusEvent: null, focusUntil: 0,
    conn: "connecting", lastSummaryAt: 0, lastStateFetch: 0, sessionResetPending: false,
  };
  const ui = { eventCards: new Map(), approvalCards: new Map(), camTiles: new Map(), openTraces: new Set() };

  // ------------------------------------------------------------------ site + state
  async function loadSite() {
    try {
      const d = await api("/api/site");
      S.site = obj(d).site || null;
      S.views = obj(obj(d).camera_views);
      if (obj(d).privacy) applyPrivacy(d.privacy);
      if (S.site) {
        $("site-name").textContent = str(S.site.name, "Unnamed site");
        document.title = `Floorsight - ${str(S.site.name, "live floor map")}`;
        computeLabelPositions();
        $("map-empty").hidden = true;
        $("map").setAttribute("aria-label", `Floor map of ${str(S.site.name, "the site")}, ${num(S.site.width_m)} by ${num(S.site.depth_m)} metres`);
      }
      layoutMap();
    } catch (e) {
      $("map-empty").textContent = "Could not load the floor plan. Retrying...";
      setTimeout(loadSite, 3000);
    }
  }

  async function loadState() {
    S.lastStateFetch = Date.now();
    try {
      applyState(await api("/api/state"));
    } catch (e) {
      /* websocket status already shows the problem */
    }
    try {
      const cov = await api("/api/coverage");
      S.coverage = cov;
      staticDirty = true;
    } catch (_) { /* coverage optional */ }
  }

  function applyState(st) {
    st = obj(st);
    S.components = obj(st.components);
    S.config = obj(st.config);
    S.warnings = arr(st.warnings);
    if (st.privacy) applyPrivacy(st.privacy);
    renderWarnings();
    renderComponents();
    S.ended = st.mode === "ended";
    const agent = obj(st.agent);
    S.events.clear(); S.approvals.clear(); S.rules.clear(); S.trace.clear();
    for (const e of arr(agent.events)) if (e && e.event_id) S.events.set(e.event_id, e);
    for (const a of arr(agent.approvals)) if (a && a.approval_id) S.approvals.set(a.approval_id, a);
    for (const r of arr(agent.rules)) if (r && r.rule_id) S.rules.set(r.rule_id, r);
    for (const s of arr(agent.trace)) addStep(s, true);
    S.cameras = arr(st.cameras);
    S.metrics = obj(st.metrics);
    S.history = obj(st.history);
    if (st.world) applyWorld(st.world, true);
    renderAll();
  }

  function applyPrivacy(p) {
    S.privacy = obj(p);
    const h = S.privacy.retention_h ?? 24;
    const retention = S.privacy.preservation_hold ? "evidence preserved locally" : `auto-delete ${h} h`;
    const box = $("privacy"), text = $("privacy-text");
    if (S.privacy.faces_blurred === false) {
      box.classList.add("privacy-warn");
      text.textContent = `Face blur not running · ${retention}`;
    } else {
      box.classList.remove("privacy-warn");
      text.textContent = `Face filter running · ${retention}`;
    }
  }

  // ------------------------------------------------------------------ live messages
  function handleMessage(m) {
    m = obj(m);
    const d = obj(m.data);
    switch (m.kind) {
      case "world": applyWorld(d, false); break;
      case "event": upsertEvent(d); break;
      case "agent_step": addStep(d, false); scheduleRender("events"); break;
      case "approval": upsertApproval(d); break;
      case "alert": showAlert(d); break;
      case "rule": if (d.rule_id) { S.rules.set(d.rule_id, d); scheduleRender("rules"); } break;
      case "coverage": S.coverage = d; staticDirty = true; break;
      default: break;
    }
  }

  // ---- forklift interlock: the machine slows or stops a forklift itself; people review afterwards
  const LOCK_TEXT = {
    stop: ["AUTO-STOP", "stop requested"],
    slow: ["AUTO-SLOW", "speed limit requested"],
    horn: ["HORN", "warning requested"],
    resume: ["RESUME", "resume requested"],
  };
  function updateInterlock(lock, t) {
    const box = $("interlock-banner");
    if (!box) return;
    const vehicles = Object.entries(obj(lock.vehicles)).filter(([, v]) => v && (v.state === "stop" || v.state === "slow"));
    const recent = arr(lock.recent).filter((c) => c && t - num(c.t) <= 2.5);
    const rows = [];
    for (const [vid, v] of vehicles) rows.push({ cmd: v.state, vehicle: vid, reason: v.waiting ? `Stopped and ${v.waiting}` : v.reason });
    for (const c of recent) if ((c.command === "horn" || c.command === "resume") && !rows.some((r) => r.vehicle === c.vehicle && r.cmd === c.command)) rows.push({ cmd: c.command, vehicle: c.vehicle, reason: c.reason, applied: c.applied });
    if (!rows.length) { box.hidden = true; box.replaceChildren(); return; }
    const top = rows.find((r) => r.cmd === "stop") || rows.find((r) => r.cmd === "slow") || rows[0];
    const last = arr(lock.recent).slice().reverse().find((c) => c.vehicle === top.vehicle && c.command === top.cmd) || {};
    box.className = `interlock-banner lock-${top.cmd}`;
    const [title, what] = LOCK_TEXT[top.cmd] || [top.cmd, ""];
    const note = last.applied === true ? "Applied in the simulator. Incident review follows separately."
      : last.applied === false ? "Command logged; no vehicle receiver applied it."
      : "Command logged; no receiver confirmation.";
    const short = typeof last.stopsShortBy_m === "number" && (top.cmd === "stop" || top.cmd === "slow")
      ? ` · braking model estimates ${last.stopsShortBy_m >= 0 ? `${last.stopsShortBy_m.toFixed(1)} m to spare` : `${(-last.stopsShortBy_m).toFixed(1)} m overrun`}` : "";
    box.replaceChildren(
      el("b", {}, `${title} · ${top.vehicle} ${what}`),
      el("span", {}, `${str(top.reason, "")}${short}`),
      el("small", {}, note),
    );
    box.hidden = false;
  }

  function applyWorld(w, fromState) {
    w = obj(w);
    const t = num(w.t, NaN);
    if (!isFinite(t)) return;
    if (S.world && t < num(S.world.t) - 1 && !fromState) {
      // Footage looped: the server started a fresh session. Re-sync everything.
      S.trails.clear(); S.display.clear(); S.events.clear(); S.approvals.clear(); S.trace.clear();
      for (const c of ui.eventCards.values()) c.el.remove();
      ui.eventCards.clear();
      backToLive();
      if (Date.now() - S.lastStateFetch > 1500) loadState();
    }
    S.world = w;
    for (const tr of arr(w.tracks)) {
      if (!tr || !tr.track_id) continue;
      let trail = S.trails.get(tr.track_id);
      if (!trail) { trail = []; S.trails.set(tr.track_id, trail); }
      const last = trail[trail.length - 1];
      if (!last || t - last.t >= 0.09) trail.push({ t, x: num(tr.x), y: num(tr.y), state: tr.state });
    }
    for (const [id, trail] of S.trails) {
      while (trail.length && trail[0].t < t - 5) trail.shift();
      if (!trail.length) S.trails.delete(id);
    }
    const health = new Map(arr(w.cameras).map((c) => [c && c.camera_id, c]));
    for (const cam of S.cameras) if (health.has(cam.camera_id)) cam.health = health.get(cam.camera_id);
    if (S.mode === "live") updateTopbarCounts(arr(w.tracks), t);
    if (!fromState) updateInterlock(obj(w.interlock), t);
    updateSlider();
    if (!fromState && Date.now() - S.lastSummaryAt > 4000) updateMapSummary();
  }

  function upsertEvent(e) {
    if (!e || !e.event_id) return;
    const prev = S.events.get(e.event_id);
    S.events.set(e.event_id, e);
    if (!S.eventSeenAt.has(e.event_id)) S.eventSeenAt.set(e.event_id, performance.now());
    if (!prev || prev.status !== e.status) {
      if (e.status === "confirmed" || e.status === "escalated") announce(`${EVENT_LABEL[e.type] || sentence(e.type)}: ${str(e.summary)}`);
      if (e.status === "rejected") announce(`False alarm rejected: ${str(e.summary)}`);
    }
    scheduleRender("events");
  }

  function addStep(s, quiet) {
    if (!s || !s.event_id) return;
    let list = S.trace.get(s.event_id);
    if (!list) { list = []; S.trace.set(s.event_id, list); }
    const i = list.findIndex((x) => x.step === s.step && x.tool === s.tool && x.t === s.t);
    if (i >= 0) list[i] = s; else list.push(s);
    list.sort((a, b) => num(a.step) - num(b.step));
  }

  function upsertApproval(a) {
    if (!a || !a.approval_id) return;
    const prev = S.approvals.get(a.approval_id);
    S.approvals.set(a.approval_id, a);
    if (a.status === "pending" && !prev) announce(`Needs your decision: ${str(a.summary)}`);
    if (a.status !== "pending" && (!prev || prev.status === "pending")) S.decidedAt.set(a.approval_id, Date.now());
    scheduleRender("approvals");
    staticDirty = true;
  }

  function showAlert(a) {
    a = obj(a);
    const box = $("toasts");
    const who = AUDIENCE_LABEL[a.audience] || "Alert";
    const t = el("div", { class: `toast toast-${str(a.audience, "floor_lead")}`, role: "alert" },
      el("b", { text: `Alert sent to ${who}` }), str(a.message, "New alert"));
    box.appendChild(t);
    while (box.children.length > 3) box.firstChild.remove();
    setTimeout(() => t.remove(), 9000);
  }

  function announce(text) {
    const a = $("announcer");
    a.textContent = "";
    setTimeout(() => { a.textContent = text; }, 30);
  }

  // ------------------------------------------------------------------ websocket
  let ws = null, wsRetry = 0, wsTimer = null;
  function connect() {
    clearTimeout(wsTimer);
    setConn("connecting");
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    try { ws = new WebSocket(`${proto}//${location.host}/ws`); } catch (_) { return scheduleReconnect(); }
    ws.onopen = () => { wsRetry = 0; setConn("ok"); loadState(); };
    ws.onmessage = (ev) => {
      let m = null;
      try { m = JSON.parse(ev.data); } catch (_) { return; }
      try { handleMessage(m); } catch (err) { console.warn("Floorsight: could not apply message", err); }
    };
    ws.onclose = () => scheduleReconnect();
    ws.onerror = () => { try { ws.close(); } catch (_) { /* ignore */ } };
  }
  function scheduleReconnect() {
    if (wsTimer) return;
    const delay = Math.min(8000, 500 * Math.pow(2, wsRetry++));
    setConn("down", delay);
    wsTimer = setTimeout(() => { wsTimer = null; connect(); }, delay);
  }
  function setConn(state, delay) {
    S.conn = state;
    const box = $("conn"), text = $("conn-text");
    box.className = "conn " + (state === "ok" ? "conn-ok" : state === "down" ? "conn-down" : "conn-wait");
    text.textContent = state === "ok" ? "Connected" : state === "down" ? `Reconnecting in ${Math.ceil((delay || 0) / 1000)} s` : "Connecting...";
  }

  // ------------------------------------------------------------------ top bar
  function updateTopbarCounts(tracks, t) {
    const n = { person: 0, forklift: 0, object: 0 };
    for (const tr of tracks) if (tr && tr.state !== "lost" && n[tr.cls] !== undefined) n[tr.cls]++;
    $("count-person").textContent = n.person;
    $("count-forklift").textContent = n.forklift;
    $("count-object").textContent = n.object;
    $("clock").textContent = fmtT(t);
  }

  function renderMetrics() {
    const m = S.metrics;
    const fps = num(m.fps, NaN);
    const lat = num(m.latency_ms, NaN);
    $("stat-fps").textContent = isFinite(fps) ? `${fps.toFixed(fps >= 100 ? 0 : 1)} fps` : "-- fps";
    $("stat-latency").textContent = isFinite(lat) ? `${Math.round(lat)} ms` : "-- ms";
    $("stat-fps").title = `Frames processed per second. Dropped so far: ${num(m.dropped)}`;
    $("stat-latency").title = `Tracking, control and review after face filtering. Vision: ${Math.round(num(m.vision_latency_ms))} ms. Excludes camera, filtering and receiver delay.`;
    S.ended = m.mode === "ended";
    renderModePill();
  }

  function renderModePill() {
    const pill = $("mode-pill"), text = $("mode-text");
    if (S.mode === "replay") {
      pill.className = "pill pill-replay";
      text.textContent = `Replay ${fmtT(S.replay.t)}`;
    } else if (S.ended) {
      pill.className = "pill pill-ended";
      text.textContent = "Recording ended";
    } else if (S.config && S.config.source === "simlive") {
      pill.className = "pill pill-recorded";
      text.textContent = "Live simulation";
    } else if (S.components && S.components.vision === "replay") {
      pill.className = "pill pill-recorded";  // recorded real clips, not a live camera
      text.textContent = "Recorded clips";
    } else {
      pill.className = "pill pill-live";
      text.textContent = "Live";
    }
  }

  function renderWarnings() {
    const box = $("warnings");
    box.replaceChildren(...S.warnings.map((w) => el("span", { text: str(w) })));
    box.hidden = S.warnings.length === 0;
  }

  function renderComponents() {
    const c = S.components;
    const vision = { real: "Camera measurements", replay: "Saved detections", fake: "Simulator truth" }[c.vision] || str(c.vision, "Unknown input");
    const agent = { real: "Review workflow", demo: "Scripted review", stub: "Review disabled" }[c.agent] || str(c.agent, "Unknown review mode");
    $("components").textContent = `${vision} · ${agent}`;
  }

  async function pollMetrics() {
    try {
      S.metrics = obj(await api("/api/metrics"));
      S.history = obj(S.metrics.history);
      renderMetrics();
      updateSlider();
    } catch (_) { /* shown by connection indicator */ }
  }

  // ------------------------------------------------------------------ rendering scheduler
  const dirty = new Set();
  let renderQueued = false;
  function scheduleRender(what) {
    dirty.add(what);
    if (renderQueued) return;
    renderQueued = true;
    requestAnimationFrame(() => {
      renderQueued = false;
      const d = new Set(dirty); dirty.clear();
      try {
        if (d.has("approvals")) renderApprovals();
        if (d.has("events")) renderEvents();
        if (d.has("rules")) renderRules();
        if (d.has("cameras")) renderCameras();
      } catch (err) { console.warn("Floorsight: render failed", err); }
    });
  }
  function renderAll() { ["approvals", "events", "rules", "cameras"].forEach(scheduleRender); renderMetrics(); }

  // ------------------------------------------------------------------ approvals
  function zoneName(id) {
    const z = arr(S.site && S.site.zones).find((z) => z && z.zone_id === id);
    return z ? str(z.name, id) : id ? titleCase(id) : "";
  }

  function renderApprovals() {
    const box = $("approvals");
    const now = Date.now();
    const show = [...S.approvals.values()].filter((a) =>
      a.status === "pending" || now - num(S.decidedAt.get(a.approval_id), 0) < 4000);
    show.sort((a, b) => num(a.created_t) - num(b.created_t));
    const keep = new Set(show.map((a) => a.approval_id));
    for (const [id, card] of ui.approvalCards) if (!keep.has(id)) { card.remove(); ui.approvalCards.delete(id); }
    show.forEach((a, i) => {
      let card = ui.approvalCards.get(a.approval_id);
      if (!card) { card = buildApproval(a); ui.approvalCards.set(a.approval_id, card); }
      updateApproval(card, a);
      if (box.children[i] !== card) box.insertBefore(card, box.children[i] || null);
    });
    const pending = show.filter((a) => a.status === "pending").length;
    $("approvals-count").textContent = pending;
    $("approvals-count").hidden = pending === 0;
    $("no-approvals").hidden = show.length > 0;
    if (show.some((a) => a.status !== "pending")) setTimeout(() => scheduleRender("approvals"), 4100);
  }

  function buildApproval(a) {
    const id = `ap-${str(a.approval_id).replace(/[^\w-]/g, "_")}`;
    // Floor warnings carry a decision card (payload.card): verdict line + 3 numbers up front, the rest folded away.
    const hasCard = !!obj(obj(a.payload).card).headline;
    const details = [el("p", { class: "ap-summary", id: `${id}-sum` }), el("p", { class: "ap-meta" }), el("div", { class: "ap-evidence" })];
    const card = el("div", { class: "approval", role: "group", "aria-labelledby": hasCard ? `${id}-verdict` : `${id}-sum` },
      el("p", { class: "ap-action" }),
      ...(hasCard
        ? [el("p", { class: "ap-verdict", id: `${id}-verdict` }), el("dl", { class: "ap-numbers" }),
           el("details", { class: "ap-details" }, el("summary", { text: "Agent's evidence and full reasoning" }), ...details)]
        : details),
      el("div", { class: "ap-buttons" },
        el("button", { type: "button", class: "btn btn-approve", onclick: () => decide(a.approval_id, true, card) }, "Approve"),
        el("button", { type: "button", class: "btn btn-reject", onclick: () => decide(a.approval_id, false, card) }, "Reject")),
      el("p", { class: "ap-status", "aria-live": "polite" }));
    return card;
  }

  function updateApproval(card, a) {
    card.querySelector(".ap-action").textContent = ACTION_LABEL[a.action] || sentence(a.action || "Decision needed");
    card.querySelector(".ap-summary").textContent = str(a.summary, "The agent is asking for a decision.");
    const ev = a.event_id ? S.events.get(a.event_id) : null;
    const payload = obj(a.payload);
    const dc = obj(payload.card);
    const verdictEl = card.querySelector(".ap-verdict");
    if (verdictEl) {
      verdictEl.textContent = str(dc.headline);
      verdictEl.className = `ap-verdict v-${str(dc.verdict, "uncertain")}`;
      const nums = arr(dc.numbers).slice(0, 3);
      card.querySelector(".ap-numbers").replaceChildren(...nums.map((n) =>
        el("div", { class: "ap-num" }, el("dt", { text: str(obj(n).label) }), el("dd", { text: str(obj(n).value) }))));
    }
    const zone = payload.zone_id || obj(payload.rule).zone_id || (ev && ev.zone_id);
    const bits = [];
    if (ev) bits.push(`About: ${str(ev.summary, EVENT_LABEL[ev.type] || "an event")}`);
    if (zone) bits.push(`Zone: ${zoneName(zone)}`);
    bits.push(`Asked at ${fmtT(a.created_t)}`);
    card.querySelector(".ap-meta").textContent = bits.join(" · ");
    const evBox = card.querySelector(".ap-evidence");
    const refs = ev ? arr(ev.evidence).filter((x) => x && x.image_ref) : [];
    if (evBox && evBox.dataset.n !== String(refs.length)) {
      evBox.dataset.n = String(refs.length);
      evBox.replaceChildren(...refs.map((x) => el("figure", { class: "ap-ev" },
        el("img", { src: `/api/evidence/${encodeURIComponent(x.image_ref)}`, alt: `${x.kind} evidence`, loading: "lazy" }),
        el("figcaption", { text: `${x.kind}: ${str(x.note).slice(0, 140)}` }))));
    }
    const decided = a.status && a.status !== "pending";
    card.classList.toggle("decided", !!decided);
    card.querySelectorAll(".ap-buttons button").forEach((b) => { if (decided) b.disabled = true; });
    card.querySelector(".ap-buttons").hidden = !!decided;
    const st = card.querySelector(".ap-status");
    if (decided) {
      st.classList.remove("err");
      const said = a.status === "approved" ? "Approved. The agent is acting on it."
        : a.status === "expired" ? "Expired: the incident closed before a decision. Nothing will be done."
        : "Rejected. Nothing will be done.";
      st.textContent = a.note ? `${a.note} ${said}` : said;
    }
  }

  async function decide(id, approved, card) {
    const buttons = card.querySelectorAll(".ap-buttons button");
    buttons.forEach((b) => (b.disabled = true));
    const st = card.querySelector(".ap-status");
    st.classList.remove("err");
    st.textContent = approved ? "Approving..." : "Rejecting...";
    try {
      const reviewer = new URLSearchParams(location.search).get("reviewer") || "dashboard user";
      const res = await api(`/api/approvals/${encodeURIComponent(id)}`, { method: "POST", body: { approved, reviewer } });
      for (const m of arr(obj(res).messages)) handleMessage(m);
      const rec = obj(obj(res).approval);
      if (rec.approval_id) upsertApproval(rec); // the server's record of the decision, not this click
      scheduleRender("approvals");
      $("panel").focus({ preventScroll: true });
    } catch (e) {
      st.classList.add("err");
      const rec = obj(obj(e.data).approval);
      if (e.status === 409 && rec.approval_id) {
        // Someone else answered first, or the incident closed: show what the server holds.
        upsertApproval(Object.assign({}, rec, { note: "Your decision was not applied: this request was already answered or closed." }));
        announce(`Your decision was not applied. The request is ${str(rec.status)}.`);
        return;
      }
      if (e.status === 404) {
        st.textContent = "The server no longer knows this request (the recording restarted). Reloading.";
        setTimeout(() => { S.approvals.delete(id); scheduleRender("approvals"); loadState(); }, 2500);
        return;
      }
      st.textContent = `Could not send your decision: ${e.message}`;
      buttons.forEach((b) => (b.disabled = false));
    }
  }

  // ------------------------------------------------------------------ events
  function visibleEvents() {
    let list = [...S.events.values()];
    if (S.mode === "replay" && S.replay.t !== null) list = list.filter((e) => num(e.t_detect, num(e.t_start)) <= S.replay.t + 0.01);
    list.sort((a, b) => num(b.t_detect, num(b.t_start)) - num(a.t_detect, num(a.t_start)));
    return list;
  }

  function renderEvents() {
    const box = $("events");
    const list = visibleEvents();
    const keep = new Set(list.map((e) => e.event_id));
    for (const [id, c] of ui.eventCards) if (!keep.has(id)) { c.el.remove(); ui.eventCards.delete(id); }
    list.forEach((e, i) => {
      let c = ui.eventCards.get(e.event_id);
      if (!c) { c = buildEventCard(e); ui.eventCards.set(e.event_id, c); }
      updateEventCard(c, e);
      if (box.children[i] !== c.el) box.insertBefore(c.el, box.children[i] || null);
    });
    $("no-events").hidden = list.length > 0;
    const rejected = list.filter((e) => e.status === "rejected").length;
    $("events-count").textContent = list.length ? `${list.length}${rejected ? ` · ${rejected} false alarm${rejected > 1 ? "s" : ""} caught` : ""}` : "";
  }

  function buildEventCard(e) {
    const key = str(e.event_id).replace(/[^\w-]/g, "_");
    const traceId = `trace-${key}`;
    const c = {};
    c.type = el("span", { class: "ev-type" });
    c.sev = el("span", { class: "ev-sev" });
    c.badge = el("span", { class: "badge" });
    c.summary = el("p", { class: "ev-summary" });
    c.meta = el("p", { class: "ev-meta" });
    c.evidence = el("ul", { class: "ev-evidence", "aria-label": "Evidence" });
    c.thumbs = el("div", { class: "ev-thumbs" });
    c.why = el("button", { type: "button", class: "btn", "aria-expanded": "false", "aria-controls": traceId });
    c.locate = el("button", { type: "button", class: "btn", title: "Highlight this place on the floor map" }, "Show on map");
    c.trace = el("ol", { class: "trace", id: traceId, hidden: true, "aria-label": "Agent decision steps" });
    c.why.addEventListener("click", () => {
      const open = c.trace.hidden;
      c.trace.hidden = !open;
      c.why.setAttribute("aria-expanded", String(open));
      if (open) ui.openTraces.add(e.event_id); else ui.openTraces.delete(e.event_id);
      updateEventCard(c, S.events.get(e.event_id) || e);
    });
    c.locate.addEventListener("click", () => { S.focusEvent = e.event_id; S.focusUntil = performance.now() + 5000; });
    c.el = el("li", { class: "event-card", "data-event-id": e.event_id },
      el("div", { class: "ev-head" }, c.type, c.sev, c.badge),
      c.summary, c.meta, c.evidence, c.thumbs,
      el("div", { class: "ev-actions" }, c.why, c.locate),
      c.trace);
    c.el.addEventListener("mouseenter", () => { S.focusEvent = e.event_id; S.focusUntil = Infinity; });
    c.el.addEventListener("mouseleave", () => { if (S.focusUntil === Infinity) S.focusEvent = null; });
    c.sig = "";
    return c;
  }

  function evidenceMark(ev) {
    if (ev.supports === true) return el("span", { class: "ev-mark yes" }, "Supports");
    if (ev.supports === false) return el("span", { class: "ev-mark no" }, "Against");
    return el("span", { class: "ev-mark info" }, "Info");
  }

  function summarizeOutput(o) {
    if (o === null || o === undefined) return "";
    if (typeof o !== "object") return str(o);
    if (typeof o.note === "string") return o.note;
    const views = arr(o.views || o.evidence);
    if (views.length) return views.map((v) => str(obj(v).note, `${str(obj(v).camera_id, "camera")}: ${obj(v).supports ? "sees it" : "does not see it"}`)).join("; ");
    const parts = [];
    for (const [k, v] of Object.entries(o)) {
      if (v === null || typeof v === "object") continue;
      parts.push(`${k.replace(/_/g, " ")}: ${typeof v === "number" ? +v.toFixed(3) : v}`);
      if (parts.length >= 4) break;
    }
    if (!parts.length && Array.isArray(o.cameras)) return o.cameras.length ? `cameras: ${o.cameras.join(", ")}` : "no camera covers this spot";
    return parts.join(", ") || "(see details)";
  }

  function updateEventCard(c, e) {
    const steps = S.trace.get(e.event_id) || [];
    const ev = arr(e.evidence);
    const open = ui.openTraces.has(e.event_id);
    const sig = JSON.stringify([e.status, e.summary, e.severity, ev.length, steps.length, open, e.camera_ids, e.zone_id]);
    if (sig === c.sig) return;
    c.sig = sig;
    const status = str(e.status, "candidate");
    c.el.className = `event-card st-${status}`;
    c.type.textContent = EVENT_LABEL[e.type] || sentence(e.type || "Event");
    c.sev.textContent = str(e.severity);
    c.sev.className = `ev-sev sev-${str(e.severity)}`;
    c.badge.textContent = STATUS_LABEL[status] || sentence(status);
    c.summary.textContent = str(e.summary, "No summary yet.");
    const meta = [];
    if (e.zone_id) meta.push(zoneName(e.zone_id));
    meta.push(`at ${fmtT(num(e.t_detect, num(e.t_start)))}`);
    const cams = arr(e.camera_ids);
    if (cams.length) meta.push(cams.join(", "));
    c.meta.textContent = meta.join(" · ");

    c.evidence.replaceChildren(...ev.filter((x) => x && x.note).slice(0, 4).map((x) =>
      el("li", { class: "ev-note" }, evidenceMark(x), el("span", { text: str(x.note) }))));
    c.evidence.hidden = c.evidence.children.length === 0;
    const imgs = ev.filter((x) => x && x.image_ref).slice(0, 6);
    c.thumbs.replaceChildren(...imgs.map((x) => {
      const src = `/api/evidence/${encodeURIComponent(x.image_ref)}`;
      const alt = `Evidence from ${str(x.camera_id, "a camera")} at ${fmtT(x.t)}${x.note ? `: ${x.note}` : ""}`;
      const img = el("img", { src, alt, loading: "lazy" });
      img.addEventListener("error", () => { img.replaceWith(el("span", { class: "cam-placeholder", text: "Image expired" })); }, { once: true });
      return el("button", { type: "button", class: "thumb-btn", "aria-label": `Enlarge: ${alt}`, onclick: () => openEvidence(src, alt) }, img);
    }));
    c.thumbs.hidden = imgs.length === 0;

    c.why.textContent = steps.length ? `${open ? "Hide" : "Why?"} (${steps.length} step${steps.length > 1 ? "s" : ""})` : "Why? (no steps yet)";
    c.why.disabled = steps.length === 0;
    if (open) c.trace.replaceChildren(...steps.map(buildStep));
    c.locate.hidden = !Array.isArray(e.location);
  }

  function buildStep(s) {
    let raw = "";
    try { raw = JSON.stringify({ input: s.tool_input, output: s.tool_output }, null, 2); } catch (_) { raw = ""; }
    if (raw.length > 2500) raw = raw.slice(0, 2500) + "\n...";
    return el("li", null,
      el("div", { class: "step-head" },
        el("span", { class: "step-n", "aria-label": `Step ${num(s.step)}` }, String(num(s.step))),
        el("span", { class: "step-tool", text: str(s.tool, "step") }),
        el("span", { "aria-hidden": "true" }, "→"),
        el("span", { class: "step-decision", text: sentence(str(s.decision, "no decision")) }),
        el("span", { class: "step-mode", text: `${fmtT(s.t)}${s.mode ? ` · ${s.mode}` : ""}` })),
      el("p", { class: "step-line" }, el("b", null, "Result: "), summarizeOutput(s.tool_output)),
      el("p", { class: "step-line" }, el("b", null, "Reason: "), str(s.reason, "-")),
      raw ? el("details", null, el("summary", null, "Raw tool data"), el("pre", { text: raw })) : null);
  }

  // ------------------------------------------------------------------ rules
  function renderRules() {
    const list = [...S.rules.values()];
    $("rules-count").textContent = `(${list.length})`;
    $("rules").replaceChildren(...list.map((r) => {
      const waiting = r.confirmed === false;
      const on = !waiting && r.enabled !== false;
      return el("li", null,
        el("span", { class: `badge ${waiting ? "rule-wait" : on ? "rule-on" : "rule-off"}` }, waiting ? "Needs OK" : on ? "On" : "Off"),
        el("span", { text: str(r.text, r.rule_id) }));
    }));
    staticDirty = true;
  }

  function setupRuleForm() {
    const input = $("rule-input"), fb = $("rule-feedback"), btn = $("rule-submit");
    document.querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => {
      input.value = chip.textContent.trim();
      input.focus();
    }));
    $("rule-form").addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const text = input.value.trim();
      if (!text) { fb.className = "feedback err"; fb.textContent = "Type a rule first, or pick an example."; input.focus(); return; }
      btn.disabled = true;
      fb.className = "feedback"; fb.textContent = "Sending to the agent...";
      try {
        const res = obj(await api("/api/rules", { method: "POST", body: { text } }));
        const msgs = arr(res.messages);
        for (const m of msgs) handleMessage(m);
        if (res.agent === "stub") {
          fb.className = "feedback err";
          fb.textContent = "The agent is not connected (stub mode), so the rule was not read. Start the server with --agent real.";
        } else if (msgs.some((m) => obj(m).kind === "approval")) {
          fb.className = "feedback ok";
          fb.textContent = "Got it. Check what the agent understood at the top, then approve to turn it on.";
          input.value = "";
        } else {
          fb.className = "feedback ok";
          fb.textContent = msgs.length ? "Rule added." : "Sent. The agent did not reply with a rule yet.";
          input.value = "";
        }
      } catch (e) {
        fb.className = "feedback err";
        fb.textContent = `Could not add the rule: ${e.message}`;
      } finally {
        btn.disabled = false;
      }
    });
  }

  // ------------------------------------------------------------------ cameras
  function renderCameras() {
    const box = $("camera-list");
    const cams = S.cameras.length ? S.cameras : arr(S.site && S.site.cameras).map((c) => ({ camera_id: c.camera_id }));
    cams.forEach((cam, i) => {
      if (!cam || !cam.camera_id) return;
      let tile = ui.camTiles.get(cam.camera_id);
      if (!tile) { tile = buildCamTile(cam); ui.camTiles.set(cam.camera_id, tile); }
      updateCamTile(tile, cam);
      if (box.children[i] !== tile.el) box.insertBefore(tile.el, box.children[i] || null);
    });
  }

  function buildCamTile(cam) {
    const t = {};
    t.img = el("img", { alt: `Live view from ${cam.camera_id}, faces blurred`, hidden: true });
    t.ph = el("span", { class: "cam-placeholder", text: "Waiting for video..." });
    t.badge = el("span", { class: "badge cam-unknown", text: "..." });
    t.stats = el("div", { class: "cam-stats" });
    t.el = el("li", { class: "cam-tile", "aria-label": `Camera ${cam.camera_id}` },
      el("div", { class: "cam-img" }, t.img, t.ph),
      el("div", { class: "cam-body" },
        el("div", { class: "cam-info" }, el("span", { class: "cam-name", text: cam.camera_id }), t.badge),
        t.stats));
    t.loading = false;
    return t;
  }

  function updateCamTile(t, cam) {
    const h = obj(cam.health);
    const status = str(h.status, "unknown");
    t.badge.className = `badge cam-${status}`;
    t.badge.textContent = CAM_LABEL[status] || (status === "unknown" ? "No data" : sentence(status));
    t.el.classList.toggle("cam-bad", status === "needs_human" || status === "offline");
    t.el.classList.toggle("cam-warn", status === "moved" || status === "repairing");
    const lines = [];
    if (typeof h.fps === "number") lines.push(`${h.fps.toFixed(0)} fps`);
    if (typeof h.calib_error_m === "number" && Number.isFinite(h.calib_error_m)) lines.push(`Floor error ${h.calib_error_m.toFixed(2)} m`);
    // real clips have no floor markers: their floor comes from the vanishing-point calibration (the tile is ~20 characters wide)
    else if (h.calib_error_m === null) lines.push(cam.blur_mode === "prefiltered" ? "Floor: calibrated" : "Floor alignment unavailable");
    if (cam.blur_mode === "prefiltered") lines.push("Heads pre-blurred");
    else if (typeof cam.faces_blurred_total === "number") lines.push(cam.blur_applied === false ? "Face blur off" : `${cam.faces_blurred_total} faces blurred`);
    t.stats.replaceChildren(...(lines.length ? lines : ["No health data yet"]).map((x) => el("div", { text: x })));
    t.stats.title = lines.join(" · ");
    if (cam.has_video === false) { t.ph.textContent = "Simulated feed (no video)"; t.ph.hidden = false; t.img.hidden = true; }
    t.hasVideo = cam.has_video === true;
    t.privacyOk = cam.blur_applied !== false;
    t.img.alt = `Live view from ${cam.camera_id}${t.privacyOk ? ", faces blurred" : ""}`;
  }

  let thumbN = 0;
  function refreshThumbs() {
    if (document.hidden) return;
    thumbN++;
    for (const [id, t] of ui.camTiles) {
      if (!t.hasVideo || t.loading) continue;
      t.loading = true;
      const pre = new Image();
      pre.onload = () => { t.img.src = pre.src; t.img.hidden = false; t.ph.hidden = true; t.loading = false; };
      pre.onerror = () => { t.loading = false; if (t.img.hidden) t.ph.textContent = "Waiting for video..."; };
      pre.src = `/api/cameras/${encodeURIComponent(id)}/frame.jpg?n=${thumbN}`;
    }
  }

  async function pollCameras() {
    try {
      const st = obj(await api("/api/state"));
      S.cameras = arr(st.cameras);
      if (st.privacy) applyPrivacy(st.privacy);
      const w = arr(st.warnings);
      if (w.join("|") !== S.warnings.join("|")) { S.warnings = w; renderWarnings(); }
      scheduleRender("cameras");
    } catch (_) { /* ignore */ }
  }

  // ------------------------------------------------------------------ replay
  const slider = $("replay");
  function sliderRange() {
    const tMin = num(S.history.t_min, 0);
    const tMax = Math.max(num(S.history.t_max, 0), num(S.world && S.world.t, 0));
    return [tMin, tMax];
  }
  function updateSlider() {
    const [a, b] = sliderRange();
    slider.min = String(a);
    slider.max = String(Math.max(a, b));
    if (S.mode === "live") {
      slider.value = slider.max;
      const edge = S.components && S.components.vision === "replay" ? "Latest" : "Live";
      slider.setAttribute("aria-valuetext", `${edge}, ${fmtT(b)}`);
      $("replay-time").textContent = edge;
    }
  }

  let replayTimer = null, replayReq = 0;
  function enterReplay(t) {
    S.mode = "replay";
    S.replay.t = t;
    $("live-btn").disabled = false;
    $("replay-time").textContent = fmtT(t);
    slider.setAttribute("aria-valuetext", `Replay at ${fmtT(t)}`);
    $("clock").textContent = fmtT(t);
    renderModePill();
    scheduleRender("events");
    clearTimeout(replayTimer);
    replayTimer = setTimeout(() => fetchReplay(t), 60);
  }

  async function fetchReplay(t) {
    const id = ++replayReq;
    try {
      const d = obj(await api(`/api/replay?t0=${(t - 5).toFixed(2)}&t1=${t.toFixed(2)}`));
      if (id !== replayReq || S.mode !== "replay") return;
      const snaps = arr(d.snapshots);
      S.replay.snap = snaps.length ? snaps[snaps.length - 1] : { t, tracks: [] };
      const trails = new Map();
      for (const s of snaps) for (const tr of arr(s.tracks)) {
        if (!tr || !tr.track_id) continue;
        if (!trails.has(tr.track_id)) trails.set(tr.track_id, []);
        trails.get(tr.track_id).push({ t: num(s.t), x: num(tr.x), y: num(tr.y), state: tr.state });
      }
      S.replay.trails = trails;
      updateTopbarCounts(arr(S.replay.snap.tracks), t);
      $("clock").textContent = fmtT(t);
    } catch (_) { /* keep last frame */ }
  }

  function backToLive() {
    S.mode = "live";
    S.replay = { t: null, snap: null, trails: new Map(), playing: false };
    setPlaying(false);
    $("live-btn").disabled = true;
    updateSlider();
    renderModePill();
    scheduleRender("events");
    if (S.world) updateTopbarCounts(arr(S.world.tracks), num(S.world.t));
  }

  let playTimer = null;
  function setPlaying(on) {
    S.replay.playing = on;
    $("play-icon").setAttribute("d", on ? "M7 5h4v14H7zM13 5h4v14h-4z" : "M7 5v14l12-7z");
    $("play-btn").setAttribute("aria-label", on ? "Pause replay" : "Play replay");
    $("play-btn").title = on ? "Pause replay" : "Play replay";
    clearInterval(playTimer);
    if (!on) return;
    playTimer = setInterval(() => {
      const [, b] = sliderRange();
      const next = num(S.replay.t, num(slider.value)) + 0.5;
      if (next >= b) { backToLive(); return; }
      slider.value = String(next);
      enterReplay(next);
      S.replay.playing = true;
    }, 500);
  }

  function setupReplay() {
    slider.addEventListener("input", () => enterReplay(parseFloat(slider.value)));
    $("live-btn").addEventListener("click", backToLive);
    $("play-btn").addEventListener("click", () => {
      if (S.replay.playing) { setPlaying(false); return; }
      if (S.mode === "live") {
        const [a, b] = sliderRange();
        const start = Math.max(a, b - 30);
        slider.value = String(start);
        enterReplay(start);
      }
      setPlaying(true);
    });
  }

  // ------------------------------------------------------------------ modal
  let lastFocus = null;
  function openModal(title, body, printable) {
    lastFocus = document.activeElement;
    $("modal-title").textContent = title;
    $("modal-body").replaceChildren(body);
    $("modal-print").hidden = !printable;
    $("modal-backdrop").hidden = false;
    $("modal-close").focus();
  }
  function closeModal() {
    if ($("modal-backdrop").hidden) return;
    $("modal-backdrop").hidden = true;
    $("modal-body").replaceChildren();
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }
  function setupModal() {
    $("modal-close").addEventListener("click", closeModal);
    $("modal-print").addEventListener("click", () => window.print());
    $("modal-backdrop").addEventListener("mousedown", (e) => { if (e.target === $("modal-backdrop")) closeModal(); });
    document.addEventListener("keydown", (e) => {
      if ($("modal-backdrop").hidden) return;
      if (e.key === "Escape") { e.preventDefault(); closeModal(); return; }
      if (e.key === "Tab") {
        const f = [...$("modal").querySelectorAll("button:not([hidden]), [href], summary, [tabindex]:not([tabindex='-1'])")].filter((x) => x.offsetParent !== null);
        if (!f.length) return;
        const first = f[0], last = f[f.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    });
  }
  function openEvidence(src, alt) {
    openModal("Evidence", el("figure", { style: "margin:0" },
      el("img", { class: "evidence-full", src, alt }),
      el("figcaption", { class: "caption", text: `${alt} (faces blurred on device)` })), false);
  }

  // ------------------------------------------------------------------ morning report
  async function openReport() {
    openModal("Morning report", el("p", { class: "calm", text: "Putting the report together..." }), true);
    try {
      const r = obj(await api("/api/report"));
      $("modal-body").replaceChildren(renderReport(r));
    } catch (e) {
      $("modal-body").replaceChildren(el("p", { class: "feedback err", text: `Could not load the report: ${e.message}` }));
    }
  }

  function renderReport(r) {
    const f = obj(r.facts), agent = obj(r.agent), by = obj(f.events_by_status);
    const real = num(by.confirmed) + num(by.escalated) + num(by.resolved);
    const frag = document.createDocumentFragment();
    const headline = agent.headline || agent.summary || agent.title ||
      (num(f.events) ? `${real} confirmed warning${real === 1 ? "" : "s"}, ${num(f.false_alarms_rejected)} false alarm${num(f.false_alarms_rejected) === 1 ? "" : "s"} caught.` : "A quiet shift: nothing was flagged.");
    const steps = typeof agent.trace_steps === "number" ? ` · ${agent.trace_steps} agent steps` : "";
    const brain = agent.mode ? ` (${agent.mode === "local" ? "on-site reasoning" : str(agent.mode)})` : "";
    frag.append(
      el("p", { class: "report-headline", text: str(headline) }),
      el("p", { class: "report-sub", text: `${str(r.site, "Site")} · ${fmtT(r.duration_s)} of footage · agent: ${r.agent_mode === "demo" ? "demo script" : str(r.agent_mode, "?")}${brain}${steps}` }));
    const tile = (v, k, cls) => el("div", { class: `tile ${cls || ""}` }, el("div", { class: "v", text: String(v) }), el("div", { class: "k", text: k }));
    const p = obj(r.privacy);
    const upstream = p.blur === "prefiltered";
    frag.append(el("div", { class: "tiles" },
      tile(real, "Confirmed by people", real ? "warn" : "good"),
      tile(num(f.false_alarms_rejected), "False alarms caught", "good"),
      tile(num(f.approvals_total) - num(f.approvals_pending), "Decisions by people"),
      tile(num(f.approvals_pending), "Still waiting for you", num(f.approvals_pending) ? "warn" : ""),
      tile(num(f.alerts), "Alerts sent"),
      upstream ? tile("On", "Heads blurred upstream", "good") : tile(num(f.faces_blurred), "Faces blurred")));

    frag.append(renderAgentReport(agent));

    const events = arr(r.events);
    if (events.length) {
      frag.append(el("section", { class: "report-section" }, el("h3", { text: "What happened" }),
        el("table", { class: "rtable" },
          el("thead", null, el("tr", null, el("th", { scope: "col" }, "Time"), el("th", { scope: "col" }, "What"), el("th", { scope: "col" }, "Outcome"), el("th", { scope: "col" }, "Details"))),
          el("tbody", null, events.map((e) => el("tr", null,
            el("td", { class: "num", text: fmtT(e.t_detect) }),
            el("td", { text: EVENT_LABEL[e.type] || sentence(e.type) }),
            el("td", null, el("span", { class: `badge st-badge st-${str(e.status)}`, text: STATUS_LABEL[e.status] || sentence(e.status) })),
            el("td", { text: str(e.summary) })))))));
    }
    const cams = arr(r.cameras);
    if (cams.length) {
      frag.append(el("section", { class: "report-section" }, el("h3", { text: "Cameras now" }),
        el("ul", null, cams.map((c) => el("li", { text: `${str(c.camera_id)}: ${CAM_LABEL[c.status] || sentence(c.status)}${typeof c.calib_error_m === "number" ? ` (floor error ${c.calib_error_m.toFixed(2)} m)` : ""}` })))));
    }
    const blurText = p.faces_blurred === false ? "face blur was NOT running"
      : upstream ? "heads and faces were pixelated upstream, before the app received the video (that stage fails closed)"
      : "faces are blurred on the camera device before anything leaves it";
    frag.append(el("p", { class: "privacy-note", text:
      `Privacy: ${blurText}; ` +
      (p.preservation_hold ? "History and filtered evidence images are preserved locally under the owner hold." : `History and evidence images are deleted after ${p.retention_h ?? 24} h.`) }));
    const wrap = el("div");
    wrap.append(frag);
    return wrap;
  }

  // Known fields of the agent's shift report get a friendly layout; anything else falls back to renderValue.
  function renderAgentReport(agent) {
    const frag = document.createDocumentFragment();
    const used = new Set(["headline", "summary", "title", "t_start", "t_end", "mode", "trace_steps", "alerts", "approvals"]);
    const section = (title, ...body) => el("section", { class: "report-section" }, el("h3", { text: title }), ...body);
    const textOf = (x) => (x && typeof x === "object" ? str(x.summary || x.text || x.note || x.zone_name || x.event_id, cellText(x)) : str(x));
    const evidenceThumb = (ref) => {
      const src = `/api/evidence/${encodeURIComponent(ref)}`;
      const img = el("img", { src, alt: "Evidence photo (faces blurred)", loading: "lazy" });
      img.addEventListener("error", () => img.closest("button") && img.closest("button").remove(), { once: true });
      return el("button", { type: "button", class: "thumb-btn", "aria-label": "Open evidence photo", onclick: () => window.open(src, "_blank") }, img);
    };

    const open = arr(agent.open_items);
    used.add("open_items");
    if (open.length) frag.append(section("Still needs attention", el("ul", null, open.map((x) => el("li", { text: textOf(x) })))));

    const counts = obj(agent.counts_by_type);
    used.add("counts_by_type");
    if (Object.keys(counts).length) {
      const cols = [["total", "Total"], ["confirmed", "Confirmed"], ["rejected", "False alarms"], ["resolved", "Resolved"], ["open", "Open"]];
      frag.append(section("Incidents by type", el("table", { class: "rtable" },
        el("thead", null, el("tr", null, el("th", { scope: "col" }, "Type"), cols.map(([, l]) => el("th", { scope: "col", text: l })))),
        el("tbody", null, Object.entries(counts).map(([type, c]) => el("tr", null,
          el("th", { scope: "row", text: EVENT_LABEL[type] || sentence(type) }),
          cols.map(([k]) => el("td", { class: "num", text: String(num(obj(c)[k])) }))))))));
    }

    const hot = arr(agent.near_miss_hotspots);
    used.add("near_miss_hotspots");
    if (hot.length) {
      frag.append(section("Near-miss hotspots", el("ul", null, hot.map((h) => {
        h = obj(h);
        const loc = Array.isArray(h.location) ? ` at (${num(h.location[0]).toFixed(1)}, ${num(h.location[1]).toFixed(1)}) m` : "";
        const fix = h.fix_added ? " · fix added" : h.fix_proposed ? ` · proposed fix: ${textOf(h.fix_proposed)}` : "";
        const refs = arr(h.evidence_refs).slice(0, 3);
        return el("li", null, `${str(h.zone_name, zoneName(h.zone_id) || "Unzoned spot")}: ${num(h.count)} near miss${num(h.count) === 1 ? "" : "es"}${loc}${fix}`,
          refs.length ? el("div", { class: "ev-thumbs" }, refs.map(evidenceThumb)) : null);
      }))));
    }

    const blockedZones = obj(agent.blocked_minutes_by_zone);
    const blockedList = arr(agent.blocked_incidents);
    ["blocked_minutes_by_zone", "blocked_incidents", "exit_b_blocked_minutes"].forEach((k) => used.add(k));
    const zoneRows = Object.entries(blockedZones).filter(([, m]) => num(m) > 0);
    if (zoneRows.length || blockedList.length) {
      frag.append(section("Blocked exits and lanes",
        zoneRows.length ? el("ul", null, zoneRows.map(([z, m]) => el("li", { text: `${zoneName(z)}: blocked for ${num(m).toFixed(1)} min` }))) : null,
        blockedList.length ? el("ul", null, blockedList.map((x) => el("li", { text: textOf(x) }))) : null));
    }

    for (const [key, title] of [["false_alarms_rejected", "False alarms the agent caught"], ["false_alarms", "False alarms the agent caught"],
      ["highlights", "Highlights"], ["rules_added", "Rules added this shift"], ["remembered_spots", "Spots the agent is watching"]]) {
      used.add(key);
      const list = arr(agent[key]);
      if (list.length) frag.append(section(title, el("ul", null, list.map((x) => el("li", { text: textOf(x) })))));
    }

    const rest = Object.entries(agent).filter(([k, v]) => !used.has(k) && v !== null && v !== undefined &&
      !(Array.isArray(v) && !v.length) && !(typeof v === "object" && !Array.isArray(v) && !Object.keys(v).length));
    for (const [k, v] of rest) frag.append(section(sentence(k), renderValue(v, 0)));
    return frag;
  }

  function renderValue(v, depth) {
    if (v === null || v === undefined) return el("p", { text: "-" });
    if (typeof v !== "object") return el("p", { text: typeof v === "number" ? String(+v.toFixed(3)) : str(v) });
    if (Array.isArray(v)) {
      if (v.every((x) => x === null || typeof x !== "object")) return el("ul", null, v.map((x) => el("li", { text: str(x) })));
      const rows = v.filter((x) => x && typeof x === "object" && !Array.isArray(x));
      const cols = [...new Set(rows.flatMap((x) => Object.keys(x)))].slice(0, 6);
      return el("table", { class: "rtable" },
        el("thead", null, el("tr", null, cols.map((c) => el("th", { scope: "col", text: sentence(c) })))),
        el("tbody", null, rows.map((x) => el("tr", null, cols.map((c) => el("td", { text: cellText(x[c]) }))))));
    }
    if (depth >= 2) return el("pre", { text: JSON.stringify(v, null, 2) });
    return el("dl", { class: "kv" }, Object.entries(v).flatMap(([k, x]) => [
      el("dt", { text: sentence(k) }),
      el("dd", null, x !== null && typeof x === "object" ? renderValue(x, depth + 1) : cellText(x))]));
  }
  function cellText(x) {
    if (x === null || x === undefined) return "-";
    if (typeof x === "number") return String(+x.toFixed(3));
    if (typeof x === "object") { try { return JSON.stringify(x); } catch (_) { return "?"; } }
    return str(x);
  }

  // ------------------------------------------------------------------ map: geometry
  const canvas = $("map"), wrapEl = $("map-wrap");
  const ctx = canvas.getContext("2d");
  const layer = document.createElement("canvas");
  const lctx = layer.getContext("2d");
  let staticDirty = true, W = 0, H = 0, dpr = 1, sc = 40, ox = 0, oy = 0, staticKey = "";

  const X = (x) => ox + x * sc;
  const Y = (y) => oy + (num(S.site && S.site.depth_m, 10) - y) * sc;

  function layoutMap() {
    const r = wrapEl.getBoundingClientRect();
    dpr = Math.max(1, window.devicePixelRatio || 1);
    W = Math.max(50, r.width); H = Math.max(50, r.height);
    for (const c of [canvas, layer]) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    const w = num(S.site && S.site.width_m, 16), d = num(S.site && S.site.depth_m, 10);
    const pad = 26;
    sc = Math.max(4, Math.min((W - 2 * pad) / w, (H - 2 * pad) / d));
    ox = (W - w * sc) / 2;
    oy = (H - d * sc) / 2;
    staticDirty = true;
  }

  function pointInPoly(x, y, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [x0, y0] = poly[i], [x1, y1] = poly[j];
      if ((y0 > y) !== (y1 > y) && x < x0 + ((y - y0) * (x1 - x0)) / (y1 - y0)) inside = !inside;
    }
    return inside;
  }
  function distToEdges(x, y, poly) {
    let best = Infinity;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [ax, ay] = poly[j], [bx, by] = poly[i];
      const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy || 1;
      const u = clamp(((x - ax) * dx + (y - ay) * dy) / L, 0, 1);
      best = Math.min(best, Math.hypot(x - (ax + u * dx), y - (ay + u * dy)));
    }
    return best;
  }
  function computeLabelPositions() {
    // label each zone at its most "inside" point, so U-shaped lanes get a readable spot
    S.labelPos = {};
    for (const z of arr(S.site && S.site.zones)) {
      const poly = arr(z && z.polygon).filter((p) => Array.isArray(p) && p.length >= 2);
      if (poly.length < 3) continue;
      const xs = poly.map((p) => p[0]), ys = poly.map((p) => p[1]);
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
      let best = null, bestScore = -Infinity;
      for (let i = 0; i <= 30; i++) for (let j = 0; j <= 20; j++) {
        const x = x0 + ((x1 - x0) * i) / 30, y = y0 + ((y1 - y0) * j) / 20;
        if (!pointInPoly(x, y, poly)) continue;
        // most "inside" point; ties (long thin lanes) go to the middle
        const score = Math.round(distToEdges(x, y, poly) * 10) / 10 - 0.01 * Math.hypot(x - mx, y - my);
        if (score > bestScore) { bestScore = score; best = [x, y]; }
      }
      S.labelPos[z.zone_id] = best || [(x0 + x1) / 2, (y0 + y1) / 2];
    }
  }

  function polyPath(c, poly) {
    c.beginPath();
    poly.forEach((p, i) => (i ? c.lineTo(X(p[0]), Y(p[1])) : c.moveTo(X(p[0]), Y(p[1]))));
    c.closePath();
  }

  function hatch(color, gap, width) {
    const p = document.createElement("canvas");
    const n = Math.max(4, Math.round(gap * dpr));
    p.width = p.height = n;
    const g = p.getContext("2d");
    g.strokeStyle = color; g.lineWidth = (width || 1.5) * dpr;
    g.beginPath(); g.moveTo(-1, n + 1); g.lineTo(n + 1, -1); g.stroke();
    g.beginPath(); g.moveTo(-1, 1); g.lineTo(1, -1); g.moveTo(n - 1, n + 1); g.lineTo(n + 1, n - 1); g.stroke();
    const pat = lctx.createPattern(p, "repeat");
    if (pat && pat.setTransform) pat.setTransform(new DOMMatrix().scale(1 / dpr));
    return pat;
  }

  function text(c, s, x, y, opt) {
    opt = opt || {};
    c.font = `${opt.weight || 600} ${opt.size || 13}px system-ui, "Segoe UI", sans-serif`;
    c.textAlign = opt.align || "center";
    c.textBaseline = opt.baseline || "middle";
    c.lineJoin = "round";
    c.lineWidth = 4;
    c.strokeStyle = "rgba(10,14,19,0.9)";
    c.strokeText(s, x, y);
    c.fillStyle = opt.color || C.text;
    c.fillText(s, x, y);
  }

  function camStatus(id) {
    const cams = arr(S.world && S.world.cameras);
    const h = cams.find((c) => c && c.camera_id === id);
    return h ? str(h.status, "ok") : "ok";
  }

  function pendingRuleZones() {
    const zones = new Set();
    for (const a of S.approvals.values()) {
      if (a.status !== "pending") continue;
      if (a.action !== "confirm_rule" && a.action !== "create_rule") continue;
      const p = obj(a.payload);
      const z = p.zone_id || obj(p.rule).zone_id;
      if (z) zones.add(z);
    }
    for (const r of S.rules.values()) if (r.confirmed === false && r.zone_id) zones.add(r.zone_id);
    return zones;
  }

  // ------------------------------------------------------------------ map: static layer
  function drawStatic() {
    const c = lctx;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    const site = S.site;
    if (!site) return;
    const w = num(site.width_m, 16), d = num(site.depth_m, 10);

    // floor + grid
    c.fillStyle = C.floor;
    c.fillRect(X(0), Y(d), w * sc, d * sc);
    c.lineWidth = 1;
    for (let gx = 1; gx < w; gx++) { c.strokeStyle = gx % 5 ? C.grid : C.grid5; c.beginPath(); c.moveTo(X(gx), Y(0)); c.lineTo(X(gx), Y(d)); c.stroke(); }
    for (let gy = 1; gy < d; gy++) { c.strokeStyle = gy % 5 ? C.grid : C.grid5; c.beginPath(); c.moveTo(X(0), Y(gy)); c.lineTo(X(w), Y(gy)); c.stroke(); }

    c.save();
    c.beginPath(); c.rect(X(0), Y(d), w * sc, d * sc); c.clip();

    // camera field-of-view wedges
    for (const cam of arr(site.cameras)) {
      const v = S.views[cam.camera_id];
      if (!v) continue;
      const st = camStatus(cam.camera_id);
      const yaw = (-num(v.yaw_deg) * Math.PI) / 180; // canvas y points down
      const half = (num(v.hfov_deg, 70) * Math.PI) / 360;
      const near = num(v.near_m, 1) * sc, far = Math.min(num(v.far_m, 20), 30) * sc;
      const cx = X(num(v.x)), cy = Y(num(v.y));
      const g = c.createRadialGradient(cx, cy, near, cx, cy, far);
      const col = C.cam[st] || C.cam.ok;
      g.addColorStop(0, hexA(col, st === "ok" ? 0.09 : 0.16)); g.addColorStop(1, hexA(col, 0));
      c.fillStyle = g;
      c.beginPath();
      c.arc(cx, cy, far, yaw - half, yaw + half);
      c.arc(cx, cy, near, yaw + half, yaw - half, true);
      c.closePath();
      c.fill();
      c.strokeStyle = hexA(col, 0.22); c.lineWidth = 1; c.setLineDash([4, 6]);
      c.beginPath(); c.moveTo(cx, cy); c.lineTo(cx + Math.cos(yaw - half) * far, cy + Math.sin(yaw - half) * far);
      c.moveTo(cx, cy); c.lineTo(cx + Math.cos(yaw + half) * far, cy + Math.sin(yaw + half) * far); c.stroke();
      c.setLineDash([]);
    }

    // coverage overlay
    const cov = S.coverage;
    if (S.coverageOn && cov && Array.isArray(cov.counts)) {
      const cell = num(cov.cell_m, 0.5), cols = num(cov.cols), rows = num(cov.rows);
      const blind = hatch("rgba(255,107,107,0.55)", 7, 1.5);
      for (let r = 0; r < rows; r++) for (let k = 0; k < cols; k++) {
        const n = num(cov.counts[r * cols + k], 0);
        const x = X(k * cell), y = Y((r + 1) * cell), s = cell * sc + 0.5;
        if (n <= 0) { c.fillStyle = "rgba(255,107,107,0.13)"; c.fillRect(x, y, s, s); if (blind) { c.fillStyle = blind; c.fillRect(x, y, s, s); } }
        else if (n === 1) { c.fillStyle = "rgba(245,179,61,0.17)"; c.fillRect(x, y, s, s); }
        else { c.fillStyle = "rgba(76,211,138,0.12)"; c.fillRect(x, y, s, s); }
      }
    }

    // zones
    const crossing = hatch("rgba(242,201,76,0.45)", 9, 2);
    for (const z of arr(site.zones)) {
      const poly = arr(z && z.polygon);
      if (poly.length < 3) continue;
      const kind = str(z.kind), isCrossing = /cross/i.test(str(z.zone_id) + " " + str(z.name));
      polyPath(c, poly);
      if (isCrossing) {
        c.fillStyle = crossing || "rgba(242,201,76,0.2)"; c.fill();
        c.strokeStyle = hexA(C.lane, 0.9); c.lineWidth = 2; c.stroke();
      } else if (kind === "exit") {
        c.fillStyle = hexA(C.exit, 0.2); c.fill();
        c.strokeStyle = C.exit; c.lineWidth = 2.5; c.stroke();
      } else if (kind === "lane") {
        c.fillStyle = hexA(C.lane, 0.04); c.fill();
        c.strokeStyle = hexA(C.lane, 0.85); c.lineWidth = 2; c.setLineDash([10, 6]); c.stroke(); c.setLineDash([]);
      } else if (kind === "restricted") {
        c.fillStyle = hexA(C.restricted, 0.15); c.fill();
        c.strokeStyle = hexA(C.restricted, 0.9); c.lineWidth = 2; c.stroke();
      } else {
        c.fillStyle = hexA(C.area, 0.06); c.fill();
        c.strokeStyle = hexA(C.area, 0.6); c.lineWidth = 1.2; c.stroke();
      }
    }

    // racks
    const rackHatch = hatch("rgba(120,140,160,0.25)", 8, 1);
    for (const o of arr(site.obstacles)) {
      const poly = arr(o && o.polygon);
      if (poly.length < 3) continue;
      polyPath(c, poly);
      c.fillStyle = C.rack; c.fill();
      if (rackHatch) { c.fillStyle = rackHatch; c.fill(); }
      c.strokeStyle = C.rackLine; c.lineWidth = 1.5; c.stroke();
    }

    // floor markers (small)
    for (const m of arr(site.markers)) {
      if (!m || !Array.isArray(m.center)) continue;
      const s = Math.max(4, num(m.size_m, 0.5) * sc * 0.6);
      c.save();
      c.translate(X(m.center[0]), Y(m.center[1]));
      c.rotate((-num(m.rotation_deg) * Math.PI) / 180);
      c.strokeStyle = hexA(C.marker, 0.55); c.lineWidth = 1;
      c.strokeRect(-s / 2, -s / 2, s, s);
      c.fillStyle = hexA(C.marker, 0.35); c.fillRect(-s / 6, -s / 6, s / 3, s / 3);
      c.restore();
    }
    c.restore();

    // floor border
    c.strokeStyle = C.border; c.lineWidth = 2;
    c.strokeRect(X(0), Y(d), w * sc, d * sc);

    // labels (after everything so they stay readable)
    const fs = clamp(sc * 0.3, 11, 17);
    for (const o of arr(site.obstacles)) {
      const poly = arr(o && o.polygon);
      if (poly.length < 3) continue;
      const cx = poly.reduce((a, p) => a + p[0], 0) / poly.length, cy = poly.reduce((a, p) => a + p[1], 0) / poly.length;
      text(c, str(o.name, o.obstacle_id), X(cx), Y(cy), { size: fs * 0.9, color: C.muted });
    }
    for (const z of arr(site.zones)) {
      const pos = S.labelPos[z.zone_id];
      if (!pos) continue;
      const kind = str(z.kind);
      const isCrossing = /cross/i.test(str(z.zone_id) + " " + str(z.name));
      const col = isCrossing ? C.lane : kind === "exit" ? C.exit : kind === "lane" ? C.lane : kind === "restricted" ? "#ff9a9a" : C.muted;
      let label = str(z.name, z.zone_id);
      if (kind === "exit") label = label.toUpperCase();
      if (isCrossing) label = label.replace(/ crossing$/i, "\ncrossing");
      const lines = label.split("\n");
      const size = kind === "exit" ? fs * 1.05 : fs * (isCrossing ? 0.8 : 0.9);
      const weight = kind === "exit" || kind === "restricted" ? 800 : 600;
      let lx = X(pos[0]), ly = Y(pos[1]);
      if (kind === "exit") {
        // exits are small: label just outside the zone so parked objects never cover it
        const ys = arr(z.polygon).map((p) => p[1]);
        const top = Y(Math.max(...ys)) - size * 0.9, bottom = Y(Math.min(...ys)) + size * 0.9;
        ly = top > Y(d) + size * 0.6 ? top : bottom;
      }
      c.font = `${weight} ${size}px system-ui, "Segoe UI", sans-serif`;
      const tw = Math.max(...lines.map((ln) => c.measureText(ln).width));
      lx = clamp(lx, X(0) + tw / 2 + 4, X(w) - tw / 2 - 4);
      lines.forEach((ln, i) => text(c, ln, lx, ly + (i - (lines.length - 1) / 2) * fs * 1.1, { size, weight, color: col }));
    }

    // cameras
    for (const cam of arr(site.cameras)) {
      const v = S.views[cam.camera_id];
      const pos = v ? [num(v.x), num(v.y)] : Array.isArray(cam.nominal_position) ? cam.nominal_position : null;
      if (!pos) continue;
      const st = camStatus(cam.camera_id);
      const col = C.cam[st] || C.cam.ok;
      const cx = X(pos[0]), cy = Y(pos[1]);
      c.fillStyle = "#0c1117"; c.strokeStyle = col; c.lineWidth = 2.5;
      c.beginPath(); c.arc(cx, cy, 10, 0, Math.PI * 2); c.fill(); c.stroke();
      c.fillStyle = col; c.beginPath(); c.arc(cx, cy, 4, 0, Math.PI * 2); c.fill();
      const w2 = num(site.width_m, 16);
      const right = pos[0] < w2 / 2;
      text(c, `${cam.camera_id}${st !== "ok" ? ` · ${CAM_LABEL[st] || st}` : ""}`, cx + (right ? 15 : -15), cy - 14,
        { size: 12, align: right ? "left" : "right", color: st === "ok" ? C.text : col, weight: 700 });
    }

    // scale bar
    const bx = X(0), by = Y(0) + 16, len = 5 * sc;
    if (by < H - 2) {
      c.strokeStyle = C.muted; c.lineWidth = 2;
      c.beginPath(); c.moveTo(bx, by - 4); c.lineTo(bx, by); c.lineTo(bx + len, by); c.lineTo(bx + len, by - 4); c.stroke();
      text(c, "5 m", bx + len + 8, by - 1, { size: 11, align: "left", color: C.muted });
    }
  }

  function hexA(hex, a) {
    const h = str(hex).replace("#", "");
    if (h.length !== 6) return `rgba(200,200,200,${a})`;
    return `rgba(${parseInt(h.slice(0, 2), 16)},${parseInt(h.slice(2, 4), 16)},${parseInt(h.slice(4, 6), 16)},${a})`;
  }

  // ------------------------------------------------------------------ map: dynamic layer
  let lastFrame = performance.now();
  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(0.2, (now - lastFrame) / 1000);
    lastFrame = now;
    if (!S.site) return;
    try {
      const key = JSON.stringify([S.coverageOn, S.coverage && S.coverage.version, arr(S.world && S.world.cameras).map((c) => c && c.status)]);
      if (staticDirty || key !== staticKey) { staticKey = key; staticDirty = false; drawStatic(); }
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(layer, 0, 0);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawPendingZones(now);
      const replay = S.mode === "replay";
      const tracks = arr(replay ? S.replay.snap && S.replay.snap.tracks : S.world && S.world.tracks);
      const tNow = replay ? num(S.replay.t) : num(S.world && S.world.t);
      ctx.save();
      // predictions can wander off the plan; keep moving things on the floor (small margin for labels)
      ctx.beginPath();
      ctx.rect(X(0) - 6, Y(num(S.site.depth_m, 10)) - 6, num(S.site.width_m, 16) * sc + 12, num(S.site.depth_m, 10) * sc + 24);
      ctx.clip();
      if (!replay) drawHazards(S.world && S.world.hazards);
      drawTrails(replay ? S.replay.trails : S.trails, tNow);
      drawEvents(now, tNow);
      drawTracks(tracks, dt, replay);
      if (!replay) drawAlertRings(S.world && S.world.hazards);
      ctx.restore();
      drawFocus(now);
    } catch (err) {
      console.warn("Floorsight: map draw failed", err);
    }
  }

  function drawPendingZones(now) {
    const zones = pendingRuleZones();
    if (!zones.size) return;
    const a = 0.5 + 0.5 * Math.sin(now / 300);
    for (const z of arr(S.site.zones)) {
      if (!zones.has(z.zone_id)) continue;
      const poly = arr(z.polygon);
      if (poly.length < 3) continue;
      polyPath(ctx, poly);
      ctx.fillStyle = `rgba(255,209,102,${0.08 + 0.1 * a})`; ctx.fill();
      ctx.strokeStyle = `rgba(255,209,102,${0.55 + 0.45 * a})`; ctx.lineWidth = 4; ctx.stroke();
      const xs = poly.map((p) => p[0]), ys = poly.map((p) => p[1]);
      const msg = "New rule here: waiting for your OK";
      ctx.font = `800 13px system-ui, "Segoe UI", sans-serif`;
      const tw = ctx.measureText(msg).width + 16;
      const w = num(S.site.width_m, 16);
      const below = Y(Math.min(...ys)) + 18;
      const ly = below < Y(0) - 10 ? below : Y(Math.max(...ys)) - 18;
      const lx = clamp(X((Math.min(...xs) + Math.max(...xs)) / 2), X(0) + tw / 2 + 2, X(w) - tw / 2 - 2);
      ctx.fillStyle = "rgba(12,17,23,0.92)";
      roundRect(ctx, lx - tw / 2, ly - 12, tw, 24, 6);
      ctx.fill();
      ctx.strokeStyle = "#ffd166"; ctx.lineWidth = 1.5; ctx.stroke();
      text(ctx, msg, lx, ly + 1, { size: 13, weight: 800, color: "#ffd166" });
    }
  }

  function trackColor(cls) { return C[cls] || C.unknown; }

  // ---- real-camera floor stage: zones, footprints, alert rings (WorldState.hazards)
  const HZ = {
    warn: { fill: "rgba(255,175,0,0.10)", line: "rgba(255,175,0,0.75)", w: 1.5 },
    blindTrailing: { fill: null, line: "rgba(190,200,215,0.55)", w: 1, dash: [4, 5] },
    blind: { fill: "rgba(90,174,255,0.20)", line: "rgba(90,174,255,0.8)", w: 1.5 },
    tail: { fill: "rgba(210,70,210,0.28)", line: "rgba(210,70,210,0.9)", w: 1.5 },
    stop: { fill: "rgba(235,50,50,0.24)", line: "rgba(235,50,50,0.95)", w: 2 },
  };
  function drawHazards(hz) {
    if (!hz) return;
    const zones = obj(hz.zones);
    for (const name of ["warn", "blindTrailing", "blind", "tail", "stop"]) {
      const st = HZ[name];
      for (const poly of arr(zones[name])) {
        if (arr(poly).length < 3) continue;
        polyPath(ctx, poly);
        if (st.fill) { ctx.fillStyle = st.fill; ctx.fill(); }
        ctx.strokeStyle = st.line; ctx.lineWidth = st.w; ctx.setLineDash(st.dash || []); ctx.stroke(); ctx.setLineDash([]);
      }
    }
  }
  function drawAlertRings(hz) {
    if (!hz) return;
    for (const a of arr(hz.alerts)) {
      const p = S.display.get(a.person_id);
      if (!p) continue;
      const cx = X(p.x), cy = Y(p.y);
      const col = a.level === "danger" ? "#ff4d4d" : "#ffaf00";
      ctx.strokeStyle = col; ctx.lineWidth = a.confirmed ? 3 : 2; ctx.setLineDash(a.confirmed ? [] : [5, 4]);
      ctx.beginPath(); ctx.arc(cx, cy, Math.max(12, 0.8 * sc), 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
      const clr = obj(a.clearance_m).median, ttc = obj(a.ttc_s).median;
      let msg;
      if (!a.confirmed) msg = `checking ${Math.min(num(a.streak, 1), 3)}/3`;
      else if (a.level === "danger") msg = arr(a.reasons)[0] || "danger zone";
      else msg = ttc !== undefined && ttc !== null ? (ttc < 0.05 ? "within 0.8 m now" : `within 0.8 m in ${num(ttc).toFixed(1)} s`) : "warning";
      if (clr !== undefined && clr !== null) msg += ` (${num(clr).toFixed(1)} m)`;
      text(ctx, msg, cx, cy - Math.max(12, 0.8 * sc) - 10, { size: 12, weight: 800, color: col });
    }
  }

  function drawTrails(trails, tNow) {
    ctx.lineCap = "round";
    for (const [id, pts] of trails) {
      if (!pts || pts.length < 2) continue;
      const cls = clsOf(id);
      const col = trackColor(cls);
      for (let i = 1; i < pts.length; i++) {
        const age = tNow - pts[i].t;
        const alpha = clamp(1 - age / 5, 0, 1) * 0.6;
        if (alpha <= 0.01) continue;
        ctx.strokeStyle = hexA(col, alpha);
        ctx.lineWidth = 3;
        if (pts[i].state === "unseen") ctx.setLineDash([3, 5]); else ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(X(pts[i - 1].x), Y(pts[i - 1].y)); ctx.lineTo(X(pts[i].x), Y(pts[i].y)); ctx.stroke();
      }
    }
    ctx.setLineDash([]);
  }

  const clsCache = new Map();
  function clsOf(id) { return clsCache.get(id) || "unknown"; }

  function drawEvents(now, tNow) {
    for (const e of S.events.values()) {
      if (!ACTIVE.has(e.status) || !Array.isArray(e.location)) continue;
      if (S.mode === "replay" && num(e.t_detect, num(e.t_start)) > tNow) continue;
      const col = C.status[e.status] || C.status.candidate;
      const cx = X(num(e.location[0])), cy = Y(num(e.location[1]));
      const age = (now - num(S.eventSeenAt.get(e.event_id), now - 99999)) / 1000;
      const simAge = tNow - num(e.t_detect, num(e.t_start));
      // keep rings for anything unresolved or critical; older confirmed warnings settle to a marker
      const pulsing = e.status !== "confirmed" || simAge < 30 || e.severity === "critical";
      const period = 1.8;
      const ph = ((now / 1000) % period) / period;
      const base = Math.max(10, 0.4 * sc);
      if (pulsing) {
        ctx.strokeStyle = hexA(col, (1 - ph) * 0.9);
        ctx.lineWidth = age < 6 ? 4 : 2.5;
        ctx.beginPath(); ctx.arc(cx, cy, base + ph * (age < 6 ? 2.2 : 1.2) * sc, 0, Math.PI * 2); ctx.stroke();
      }
      ctx.fillStyle = hexA(col, 0.9);
      ctx.beginPath(); ctx.moveTo(cx, cy - 8); ctx.lineTo(cx + 8, cy); ctx.lineTo(cx, cy + 8); ctx.lineTo(cx - 8, cy); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = "#0c1117"; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }

  function drawTracks(tracks, dt, replay) {
    const seen = new Set();
    const k = replay ? 1 : 1 - Math.exp(-dt * 12);
    for (const tr of tracks) {
      if (!tr || !tr.track_id || tr.state === "lost") continue;
      const id = tr.track_id;
      seen.add(id);
      clsCache.set(id, str(tr.cls, "unknown"));
      let p = S.display.get(id);
      const tx = num(tr.x), ty = num(tr.y);
      if (!p || replay || Math.hypot(p.x - tx, p.y - ty) > 3) p = { x: tx, y: ty };
      else { p.x += (tx - p.x) * k; p.y += (ty - p.y) * k; }
      S.display.set(id, p);
      const vx = num(tr.vx), vy = num(tr.vy);
      if (Math.hypot(vx, vy) > 0.15) S.headings.set(id, Math.atan2(vy, vx));
      drawTrack(tr, p.x, p.y, S.headings.get(id) || 0);
    }
    if (!replay) for (const id of S.display.keys()) if (!seen.has(id)) S.display.delete(id);
    hitTracks = tracks.filter((t) => t && t.state !== "lost").map((t) => ({ t, p: S.display.get(t.track_id) || { x: num(t.x), y: num(t.y) } }));
  }

  function drawTrack(tr, x, y, heading) {
    const cls = str(tr.cls, "unknown");
    const col = trackColor(cls);
    const unseen = tr.state === "unseen" || tr.state === "estimated";
    const cx = X(x), cy = Y(y);
    const sigma = num(tr.sigma_m, 0.5);
    const ring = clamp(sigma, 0.5, 2.0) * sc; // big uncertainties would swamp the map: cap the drawing, print the number
    if (unseen) {
      const r = ring;
      ctx.setLineDash([6, 5]);
      ctx.strokeStyle = hexA(col, sigma > 2 ? 0.55 : 0.8); ctx.lineWidth = 2;
      ctx.fillStyle = hexA(col, 0.07);
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.lineWidth = 2;
    const fp = cls === "forklift" ? arr(obj(obj(S.world && S.world.hazards).footprints)[tr.track_id]) : [];
    if (fp.length >= 3 && S.mode !== "replay") {
      // measured footprint (truck + carried load) from the real-camera floor stage
      polyPath(ctx, fp);
      if (unseen) { ctx.strokeStyle = col; ctx.setLineDash([5, 4]); ctx.lineWidth = 2; ctx.stroke(); ctx.setLineDash([]); }
      else { ctx.fillStyle = hexA(col, 0.85); ctx.fill(); ctx.strokeStyle = "#1a0d02"; ctx.lineWidth = 1.5; ctx.stroke(); }
      const notes = arr(obj(obj(S.world && S.world.hazards).notes)[tr.track_id]);
      notes.slice(0, 2).forEach((n, i) => text(ctx, n, cx, cy + 0.9 * sc + 14 + 14 * i, { size: 11, weight: 700, color: C.muted }));
    } else if (cls === "forklift") {
      const L = 2.0 * sc, Wd = 1.1 * sc;
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(-heading);
      roundRect(ctx, -L / 2, -Wd / 2, L, Wd, Math.min(6, Wd / 3));
      if (unseen) { ctx.strokeStyle = col; ctx.setLineDash([5, 4]); ctx.stroke(); ctx.setLineDash([]); }
      else { ctx.fillStyle = col; ctx.fill(); ctx.strokeStyle = "#1a0d02"; ctx.lineWidth = 1.5; ctx.stroke(); }
      // heading arrow
      ctx.fillStyle = unseen ? col : "#1a0d02";
      ctx.beginPath();
      ctx.moveTo(L / 2 - 2, 0); ctx.lineTo(L / 2 - Math.min(L * 0.35, 18), -Math.min(Wd * 0.32, 9)); ctx.lineTo(L / 2 - Math.min(L * 0.35, 18), Math.min(Wd * 0.32, 9));
      ctx.closePath(); ctx.fill();
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.moveTo(L / 2 + 14, 0); ctx.lineTo(L / 2 + 4, -6); ctx.lineTo(L / 2 + 4, 6); ctx.closePath(); ctx.fill();
      ctx.restore();
    } else if (cls === "object") {
      const s = Math.max(10, 0.8 * sc);
      if (unseen) { ctx.strokeStyle = col; ctx.setLineDash([4, 3]); ctx.strokeRect(cx - s / 2, cy - s / 2, s, s); ctx.setLineDash([]); }
      else { ctx.fillStyle = col; ctx.fillRect(cx - s / 2, cy - s / 2, s, s); ctx.strokeStyle = "#1c1008"; ctx.lineWidth = 1.5; ctx.strokeRect(cx - s / 2, cy - s / 2, s, s); }
    } else {
      const r = Math.max(7, 0.25 * sc);
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
      if (unseen) { ctx.strokeStyle = col; ctx.lineWidth = 2.5; ctx.stroke(); }
      else { ctx.fillStyle = col; ctx.fill(); ctx.strokeStyle = "#f4f8fc"; ctx.lineWidth = 1.5; ctx.stroke(); }
    }
    if (unseen) {
      text(ctx, `${tr.state} ±${sigma.toFixed(1)} m`, cx, cy + ring + 10, { size: 12, weight: 700, color: C.muted });
    }
  }

  function roundRect(c, x, y, w, h, r) {
    c.beginPath();
    c.moveTo(x + r, y); c.lineTo(x + w - r, y); c.quadraticCurveTo(x + w, y, x + w, y + r);
    c.lineTo(x + w, y + h - r); c.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    c.lineTo(x + r, y + h); c.quadraticCurveTo(x, y + h, x, y + h - r);
    c.lineTo(x, y + r); c.quadraticCurveTo(x, y, x + r, y); c.closePath();
  }

  function drawFocus(now) {
    if (!S.focusEvent || now > S.focusUntil) { if (S.focusUntil !== Infinity) S.focusEvent = null; return; }
    const e = S.events.get(S.focusEvent);
    if (!e || !Array.isArray(e.location)) return;
    const cx = X(num(e.location[0])), cy = Y(num(e.location[1]));
    const r = 22 + 4 * Math.sin(now / 150);
    ctx.strokeStyle = "#ffd166"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx - r - 12, cy); ctx.lineTo(cx - r + 6, cy); ctx.moveTo(cx + r - 6, cy); ctx.lineTo(cx + r + 12, cy);
    ctx.moveTo(cx, cy - r - 12); ctx.lineTo(cx, cy - r + 6); ctx.moveTo(cx, cy + r - 6); ctx.lineTo(cx, cy + r + 12);
    ctx.stroke();
  }

  // ------------------------------------------------------------------ map: hover + summary
  let hitTracks = [];
  function setupMapHover() {
    const tip = $("map-tooltip");
    canvas.addEventListener("mousemove", (ev) => {
      const r = canvas.getBoundingClientRect();
      const mx = ev.clientX - r.left, my = ev.clientY - r.top;
      let best = null, bestD = 18;
      for (const h of hitTracks) {
        const d = Math.hypot(X(h.p.x) - mx, Y(h.p.y) - my);
        if (d < bestD) { bestD = d; best = h.t; }
      }
      if (!best) { tip.hidden = true; return; }
      const speed = Math.hypot(num(best.vx), num(best.vy));
      const cams = arr(best.camera_ids);
      tip.replaceChildren(
        el("strong", { text: `${CLASS_LABEL[best.cls] || "Track"} ${str(best.track_id)}` }),
        el("span", { text: `${speed.toFixed(1)} m/s · ${best.state === "unseen" ? "not in view, position estimated" : cams.length ? `seen by ${cams.join(", ")}` : "seen"}` }));
      tip.hidden = false;
      tip.style.left = `${clamp(mx + 14, 0, r.width - 270)}px`;
      tip.style.top = `${clamp(my + 14, 0, r.height - 60)}px`;
    });
    canvas.addEventListener("mouseleave", () => { tip.hidden = true; });
  }

  function updateMapSummary() {
    S.lastSummaryAt = Date.now();
    const tracks = arr(S.world && S.world.tracks).filter((t) => t && t.state !== "lost");
    const n = (cls) => tracks.filter((t) => t.cls === cls).length;
    const unseen = tracks.filter((t) => t.state === "unseen").length;
    const active = [...S.events.values()].filter((e) => ACTIVE.has(e.status)).length;
    $("map-summary").textContent = `Now on the floor: ${n("person")} people, ${n("forklift")} forklifts, ${n("object")} objects` +
      `${unseen ? `; ${unseen} out of camera view` : ""}. ${active} active event${active === 1 ? "" : "s"}.`;
  }

  // ------------------------------------------------------------------ keyboard + controls
  function setupControls() {
    const covBtn = $("coverage-btn");
    const toggleCoverage = () => {
      S.coverageOn = !S.coverageOn;
      covBtn.setAttribute("aria-pressed", String(S.coverageOn));
      $("coverage-legend").hidden = !S.coverageOn;
      staticDirty = true;
    };
    covBtn.addEventListener("click", toggleCoverage);
    if (new URLSearchParams(location.search).get("coverage") === "1") toggleCoverage();
    $("report-btn").addEventListener("click", openReport);
    document.addEventListener("keydown", (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey || !$("modal-backdrop").hidden) return;
      const tag = (e.target && e.target.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA" || (e.target && e.target.isContentEditable)) return;
      if (e.key === "c" || e.key === "C") { e.preventDefault(); toggleCoverage(); }
      else if (e.key === "l" || e.key === "L") { e.preventDefault(); backToLive(); }
    });
    if (window.ResizeObserver) new ResizeObserver(() => layoutMap()).observe(wrapEl);
    else window.addEventListener("resize", layoutMap);
  }

  // ------------------------------------------------------------------ boot
  function boot() {
    setupRuleForm();
    setupReplay();
    setupModal();
    setupControls();
    setupMapHover();
    layoutMap();
    loadSite().then(() => scheduleRender("cameras"));
    connect();
    requestAnimationFrame(frame);
    setInterval(pollMetrics, 2000);
    setInterval(refreshThumbs, 1000);
    setInterval(pollCameras, 3000);
    setInterval(() => { if (S.approvals.size) scheduleRender("approvals"); }, 5000);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
