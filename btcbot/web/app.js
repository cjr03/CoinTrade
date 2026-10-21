"use strict";
// Phosphor Desk client: live feed -> chart, status poll -> panels/console, command line -> /api/command.

const TOKEN = document.querySelector('meta[name="btcbot-token"]').content;
const $ = (id) => document.getElementById(id);
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

const VIEW_META = {
  "10m": { title: "last ten minutes", sub: "every trade", refresh: 10000, span: 600 },
  "1h": { title: "last hour", sub: "1-minute candles", refresh: 20000, span: 3600 },
  "1d": { title: "last day", sub: "5-minute candles", refresh: 60000, span: 86400 },
  "1w": { title: "last week", sub: "1-hour candles", refresh: 300000, span: 604800 },
};
const TRADE_EVENTS = new Set(["entry", "exit", "scale_up", "manual_buy", "manual_sell", "KILL_SWITCH"]);
const COIN_META = {
  BTC: { name: "bitcoin", ws: "BTC/USD", product: "BTC-USD" },
  ETH: { name: "ethereum", ws: "ETH/USD", product: "ETH-USD" },
  SOL: { name: "solana", ws: "SOL/USD", product: "SOL-USD" },
  XRP: { name: "xrp", ws: "XRP/USD", product: "XRP-USD" },
};

const S = {
  view: "1h", data: null, ticks: [], last: null, prevLast: null, lastTickAt: 0,
  status: null, since: 0, allFills: [], hover: null, dirty: true, pending: null,
  history: [], histIdx: -1, ws: null, wsTickAt: 0, bootId: null, coin: "BTC", coins: [],
  live: {},  // coin -> {last, chg} from the all-coins WebSocket ticker
  tapePx: {},  // coin -> last price painted on the tape (for the tick flash)
  forecast: null,  // the selected coin's forecast curve (/api/forecast)
};
// allocation colours: the trend sleeves in amber shades, the bet pool in the model's cyan
const ALLOC = { BTC: "#ffb547", ETH: "#ffd98c", SOL: "#c7862b", XRP: "#8f6a35", BETS: "#5fe3e8" };
const SYM_TO_COIN = Object.fromEntries(Object.entries(COIN_META).flatMap(([c, m]) => [[m.ws, c], [m.product, c]]));
try { S.coin = localStorage.getItem("btcbot.coin") || "BTC"; } catch (e) { /* private mode */ }
const fillsFor = (c) => S.allFills.filter((f) => f.coin === c);

// --- formatting -------------------------------------------------------------------------------
const num = (n, d = 1) => (n == null || !isFinite(n) ? "—" : n.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }));
const usd = (n) => (n == null || !isFinite(n) ? "—" : (n < 0 ? "−$" : "$") + num(Math.abs(n), 2));
const pct = (n, d = 2) => (n == null || !isFinite(n) ? "—" : (n >= 0 ? "+" : "−") + Math.abs(n * 100).toFixed(d) + "%");
const hhmm = (t) => new Date(t * 1000).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
const dayhh = (t) => new Date(t * 1000).toLocaleString("en-US", { weekday: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false });
const dayLabel = (t) => new Date(t * 1000).toLocaleDateString("en-US", { weekday: "short", day: "numeric" });
// prices span $1 (xrp) to $80,000 (btc): pick decimals by magnitude
const pdec = (n) => (Math.abs(n) >= 1000 ? 1 : Math.abs(n) >= 10 ? 2 : 4);
const px = (n) => num(n, pdec(n));
const pxr = (n) => num(n, Math.abs(n) >= 1000 ? 0 : Math.abs(n) >= 10 ? 2 : 4);

// count a number up/down to its new value instead of jumping (the eye follows the change)
function tween(el, to, fmt, ms = 650) {
  if (to == null || !isFinite(to)) { el.textContent = fmt(to); el._v = null; return; }
  const from = el._v;
  el._v = to;
  if (reduced || from == null || Math.abs(from - to) < 1e-9) { el.textContent = fmt(to); return; }
  const t0 = performance.now();
  cancelAnimationFrame(el._raf);
  const step = (now) => {
    const k = Math.min(1, (now - t0) / ms), e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(from + (to - from) * e);
    if (k < 1) el._raf = requestAnimationFrame(step);
  };
  el._raf = requestAnimationFrame(step);
}
function flash(el, up) {
  if (reduced) return;
  el.classList.remove("tick-up", "tick-down"); void el.offsetWidth; el.classList.add(up ? "tick-up" : "tick-down");
}
const dur = (sec) => { sec = Math.max(0, sec); const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60);
  return h ? `${h}h ${String(m).padStart(2, "0")}m` : `${m}m`; };

// live odds for a 15-minute window: same formula as btcbot.calls.live_odds (constants come with each call)
function erf(x) {  // Abramowitz-Stegun 7.1.26, |error| < 1.5e-7
  const s = Math.sign(x); x = Math.abs(x);
  const k = 1 / (1 + 0.3275911 * x);
  return s * (1 - ((((1.061405429 * k - 1.453152027) * k + 1.421413741) * k - 0.284496736) * k + 0.254829592) * k * Math.exp(-x * x));
}
const Phi = (x) => 0.5 * (1 + erf(x / Math.SQRT2));
const tauEff = (remS) => { const r = remS / 60; return r > 1 ? r - 2 / 3 : r ** 3 / 3; };  // 60-second settlement average
function liveOdds(call, price, now = Date.now() / 1000) {
  if (!call || !call.start || !price || !(call.sigma1m > 0)) return null;
  const t0 = Date.parse(call.from) / 1000, t1 = Date.parse(call.until) / 1000, rem = t1 - now;
  if (rem <= 0) return price >= call.start ? 1 : 0;
  const frac = Math.min(rem / (t1 - t0), 1);
  const z = Math.log(price / call.start) / (call.sigma1m * Math.sqrt(tauEff(rem)));
  return Math.min(0.98, Math.max(0.02, Phi(((call.a0 ?? 1) + (call.a1 ?? 0) * frac) * z + (call.z0 || 0) * Math.sqrt(frac))));
}
// Kalshi's live "yes" (up) price for the same window, in cents
function kalshiCents(call) {
  const k = call && call.kalshi;
  if (!k || k.from !== call.from.replace("+00:00", "Z") && Date.parse(k.from) !== Date.parse(call.from)) return null;
  const mid = k.yes_bid != null && k.yes_ask != null ? (k.yes_bid + k.yes_ask) / 2 : k.last;
  return mid == null ? null : Math.round(mid * 100);
}
function coinPrice(c) {
  const lv = S.live[c], row = ((S.status && S.status.portfolio && S.status.portfolio.coins) || {})[c] || {};
  return lv && Date.now() - lv.at < 15000 ? lv.last : row.last;
}
// a 15-minute chance of "up" in words
const verdict = (p) => p >= 0.8 ? "very likely higher" : p >= 0.6 ? "likely higher" : p > 0.4 ? "a toss-up"
  : p > 0.2 ? "likely lower" : "very likely lower";
const mmss = (sec) => { sec = Math.max(0, Math.floor(sec)); return `${Math.floor(sec / 60)}m ${String(sec % 60).padStart(2, "0")}s`; };

// forecast curve: interpolate the server's horizons (minutes) on log-time; below the first, spread grows as sqrt(t)
const FC_KEYS = ["mu", "q10", "q25", "q50", "q75", "q90"];
function fcAt(f, h) {
  const P = f.points;
  if (!(h > 0)) return Object.fromEntries(FC_KEYS.map((k) => [k, 0]));
  if (h <= P[0].h) {
    const a = P[0], r = h / a.h, s = Math.sqrt(r);
    return Object.fromEntries(FC_KEYS.map((k) => [k, a.mu * r + (a[k] - a.mu) * s]));
  }
  let i = 1;
  while (i < P.length - 1 && P[i].h < h) i++;
  const a = P[i - 1], b = P[i];
  if (h >= b.h) return Object.fromEntries(FC_KEYS.map((k) => [k, b[k]]));
  const w = (Math.log(h) - Math.log(a.h)) / (Math.log(b.h) - Math.log(a.h));
  return Object.fromEntries(FC_KEYS.map((k) => [k, a[k] + (b[k] - a[k]) * w]));
}
const fmtAhead = (sec) => sec < 3600 ? `next ${Math.round(sec / 60)}m` : sec < 172800 ? `next ${(sec / 3600).toFixed(sec < 36000 ? 1 : 0)}h` : `next ${(sec / 86400).toFixed(1)}d`;
async function loadForecast() {
  clearTimeout(loadForecast.t);
  try {
    const coin = S.coin, f = await getJSON(`/api/forecast?coin=${coin}`);
    if (coin === S.coin && f.points) { S.forecast = f; S.dirty = true; }
  } catch (e) { /* keep the last curve */ }
  finally { loadForecast.t = setTimeout(loadForecast, 30000); }
}

async function getJSON(path) {
  const r = await fetch(path, { cache: "no-store" });
  return r.json();
}

// --- boot: the one orchestrated moment ----------------------------------------------------------
function boot() {
  const lines = [
    "phosphor desk  rev 1.0",
    "memory test ................ 640k ok",
    "exchange link .............. ok",
    "model weights .............. loaded",
    "judged on 31 quarters it never trained on",
    "ready",
  ];
  if (reduced) { $("boot").classList.add("done"); return; }
  const pre = $("boot-text");
  let i = 0;
  (function next() {
    if (i < lines.length) { pre.textContent += lines[i++] + "\n"; setTimeout(next, 130); return; }
    setTimeout(() => { $("boot").classList.add("done"); $("crt").classList.add("powering"); }, 260);
  })();
}

// --- live price feed ----------------------------------------------------------------------------
function onPrice(price, t = Date.now() / 1000, fromWs = false) {
  if (!isFinite(price) || !isFinite(t) || price <= 0) return;
  if (fromWs) S.wsTickAt = Date.now();
  S.prevLast = S.last;
  S.last = price;
  S.lastTickAt = Date.now();
  S.ticks.push([t, price]);
  const cutoff = t - 1800;
  while (S.ticks.length && S.ticks[0][0] < cutoff) S.ticks.shift();
  if (S.data && S.data.kind === "candles") {
    const bars = S.data.bars, secs = S.data.secs, lastBar = bars[bars.length - 1];
    if (lastBar && t < lastBar[0]) { /* late trade for an older bar: ignore */ }
    else if (lastBar && t >= lastBar[0] + secs) {
      const start = Math.floor(t / secs) * secs;
      bars.push([start, price, price, price, price, 0]);
      if (bars.length > 400) bars.shift();
    } else if (lastBar) {
      lastBar[4] = price; lastBar[2] = Math.max(lastBar[2], price); lastBar[3] = Math.min(lastBar[3], price);
    }
  }
  paintPrice();
  S.dirty = true;
}

function paintPrice() {
  const el = $("price");
  el.textContent = px(S.last);
  if (S.prevLast != null && S.last !== S.prevLast && !reduced) {
    el.classList.remove("up", "down");
    void el.offsetWidth;
    el.classList.add(S.last > S.prevLast ? "up" : "down");
    clearTimeout(paintPrice.t);
    paintPrice.t = setTimeout(() => el.classList.remove("up", "down"), 120);
  }
}

function connectFeed(exchange) {
  const coin = S.coin;
  if (S.ws) { const old = S.ws; S.ws = null; try { old.close(); } catch (e) { /* already closed */ } }
  const retry = (ws) => () => { if (S.ws === ws) setTimeout(() => { if (S.ws === ws) connectFeed(exchange); }, 4000); };
  try {
    if (exchange === "kraken") {
      const ws = new WebSocket("wss://ws.kraken.com/v2");
      ws.onopen = () => {
        const all = (S.coins.length ? S.coins : Object.keys(COIN_META)).map((c) => COIN_META[c].ws);
        ws.send(JSON.stringify({ method: "subscribe", params: { channel: "trade", symbol: [COIN_META[coin].ws] } }));
        ws.send(JSON.stringify({ method: "subscribe", params: { channel: "ticker", symbol: all } }));
      };
      ws.onmessage = (m) => {
        const d = JSON.parse(m.data);
        if (S.ws !== ws || coin !== S.coin) return;
        if (d.channel === "trade" && d.type === "update") {
          for (const tr of d.data) onPrice(+tr.price, Date.parse(tr.timestamp) / 1000, true);
        } else if (d.channel === "ticker" && d.data) {
          for (const k of d.data) {
            const c = SYM_TO_COIN[k.symbol];
            if (!c) continue;
            S.live[c] = { last: +k.last, chg: k.change_pct / 100, at: Date.now() };
            if (c === S.coin) setStats({ bid: k.bid, ask: k.ask, high24: k.high, low24: k.low, chg: k.change_pct / 100 });
          }
          paintTape();
        }
      };
      ws.onclose = retry(ws);
      S.ws = ws;
    } else {
      const ws = new WebSocket("wss://advanced-trade-ws.coinbase.com");
      ws.onopen = () => ws.send(JSON.stringify({ type: "subscribe", channel: "ticker",
        product_ids: (S.coins.length ? S.coins : Object.keys(COIN_META)).map((c) => COIN_META[c].product) }));
      ws.onmessage = (m) => {
        const d = JSON.parse(m.data);
        if (d.channel !== "ticker" || S.ws !== ws || coin !== S.coin) return;
        for (const ev of d.events || []) for (const k of ev.tickers || []) {
          const c = SYM_TO_COIN[k.product_id];
          if (!c) continue;
          S.live[c] = { last: +k.price, chg: +k.price_percent_chg_24_h / 100, at: Date.now() };
          if (c !== S.coin) continue;
          onPrice(+k.price, Date.now() / 1000, true);
          setStats({ bid: +k.best_bid, ask: +k.best_ask, high24: +k.high_24_h, low24: +k.low_24_h, chg: +k.price_percent_chg_24_h / 100 });
        }
        paintTape();
      };
      ws.onclose = retry(ws);
      S.ws = ws;
    }
  } catch (e) { /* polling fallback below keeps the price moving */ }
}

function setStats({ bid, ask, high24, low24, chg }) {
  const ok = (v) => v != null && v !== "" && isFinite(+v) && +v !== 0;
  if (ok(bid)) $("bid").textContent = px(+bid);
  if (ok(ask)) $("ask").textContent = px(+ask);
  if (ok(high24)) $("hi").textContent = pxr(+high24);
  if (ok(low24)) $("lo").textContent = pxr(+low24);
  if (chg != null && isFinite(chg)) {
    const c = $("chg");
    c.textContent = pct(chg) + " 24h";
    c.classList.toggle("neg", chg < 0);
  }
}

// --- chart data ---------------------------------------------------------------------------------
async function loadView(view) {
  S.view = view;
  document.querySelectorAll(".keys button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.view === view)));
  $("chart-title").textContent = VIEW_META[view].title;
  $("chart-sub").textContent = VIEW_META[view].sub;
  try {
    const coin = S.coin;
    const d = await getJSON(`/api/candles?view=${view}&coin=${coin}`);
    if (d.error) throw new Error(d.error);
    if (view !== S.view || coin !== S.coin) return;
    if (d.kind === "candles") d.bars = d.bars.map((b) => b.slice());
    S.data = d;
    S.dirty = true;
  } catch (e) {
    log(`chart data unavailable: ${e.message}`, "err");
  }
  clearTimeout(loadView.t);
  loadView.t = setTimeout(() => loadView(S.view), VIEW_META[view].refresh);
}

// --- chart drawing ------------------------------------------------------------------------------
const C = {
  amber: "#ffb547", hot: "#ffe2b0", dim: "#5e7a86", ember: "#ff5a36", cyan: "#5fe3e8", grid: "rgba(94,122,134,0.22)",
};

function niceStep(range, target) {
  const raw = range / target, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / mag;
  return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
}

function draw(now) {
  const cv = $("chart");
  const rect = cv.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(rect.width * dpr) || cv.height !== Math.round(rect.height * dpr)) {
    cv.width = Math.round(rect.width * dpr); cv.height = Math.round(rect.height * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);
  if (!S.data) { ctx.fillStyle = C.dim; ctx.font = "20px VT323"; ctx.fillText("tuning in…", 16, 30); return; }

  const padL = 8, padR = 70, padT = 22, padB = 24;
  const futureW = Math.max(70, Math.round((W - padL - padR) * 0.2));
  const xNow = W - padR - futureW;
  const tNow = Date.now() / 1000;
  const meta = VIEW_META[S.view];

  // series
  let bars = null, pts = null, t0, t1 = tNow;
  if (S.data.kind === "candles") {
    bars = S.data.bars;
    t0 = bars.length ? bars[0][0] : tNow - meta.span;
    t1 = Math.max(tNow, bars.length ? bars[bars.length - 1][0] + S.data.secs : tNow);
  } else {
    t0 = tNow - meta.span;
    const lastHist = S.data.points.length ? S.data.points[S.data.points.length - 1][0] : 0;
    pts = S.data.points.concat(S.ticks.filter((p) => p[0] > lastHist)).filter((p) => p[0] >= t0);
  }
  const X = (t) => padL + ((t - t0) / (t1 - t0)) * (xNow - padL);

  // projection inputs
  const st = S.status, pol = st && st.policy, sig = st && st.state.signal;
  const p0 = S.last ?? (bars && bars.length ? bars[bars.length - 1][4] : pts && pts.length ? pts[pts.length - 1][1] : null);
  const fc = S.forecast && S.forecast.coin === S.coin ? S.forecast : null;
  const cone = fc && p0 ? { fc } : null;

  // y range
  let lo = Infinity, hi = -Infinity;
  if (bars) for (const b of bars) { lo = Math.min(lo, b[3]); hi = Math.max(hi, b[2]); }
  if (pts) for (const p of pts) { lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]); }
  if (p0) { lo = Math.min(lo, p0); hi = Math.max(hi, p0); }
  if (!isFinite(lo)) return;
  const span0 = Math.max(hi - lo, p0 * 0.002);
  if (cone) { // let the 80% band show without flattening the history
    const v = fcAt(cone.fc, (meta.span * 0.25) / 60);
    const up = p0 * Math.exp(v.q90), dn = p0 * Math.exp(v.q10);
    hi = Math.min(Math.max(hi, up), hi + span0 * 1.2); lo = Math.max(Math.min(lo, dn), lo - span0 * 1.2);
  }
  const pos = st && st.state.position;
  if (pos) {
    for (const v of [pos.entry_px, pos.stop_px]) if (v > lo - span0 && v < hi + span0) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  }
  if (hi - lo < span0) { const m = (hi + lo) / 2; lo = m - span0 / 2; hi = m + span0 / 2; }
  const pad = (hi - lo) * 0.08;
  lo -= pad; hi += pad;
  const Y = (p) => padT + (1 - (p - lo) / (hi - lo)) * (H - padT - padB);

  // grid + price axis
  ctx.font = "17px VT323";
  ctx.textBaseline = "middle";
  const step = niceStep(hi - lo, 6);
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.setLineDash([1, 4]);
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    const y = Math.round(Y(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = C.dim; ctx.fillText(num(v, step >= 1 ? 0 : step >= 0.01 ? 2 : 4), W - padR + 8, y);
  }
  // time axis
  let tStep = { "10m": 120, "1h": 600, "1d": 4 * 3600, "1w": 86400 }[S.view];
  while (X(t0 + tStep) - X(t0) < 56) tStep *= 2; // keep labels from colliding on narrow screens
  ctx.textBaseline = "alphabetic";
  const tz = new Date().getTimezoneOffset() * 60; // align ticks to local midnight / local hours
  for (let t = Math.ceil((t0 - tz) / tStep) * tStep + tz; t <= tNow; t += tStep) {
    const x = Math.round(X(t)) + 0.5;
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, H - padB); ctx.stroke();
    ctx.fillStyle = C.dim;
    ctx.fillText(S.view === "1w" ? dayLabel(t) : hhmm(t), x - 18, H - 6);
  }
  ctx.setLineDash([]);

  // future strip
  ctx.fillStyle = "rgba(95,227,232,0.035)";
  ctx.fillRect(xNow, padT, futureW, H - padT - padB);
  ctx.save();
  ctx.beginPath(); ctx.rect(xNow, padT, futureW, H - padT - padB); ctx.clip();
  ctx.strokeStyle = "rgba(95,227,232,0.07)";
  for (let x = xNow - H; x < xNow + futureW; x += 9) { ctx.beginPath(); ctx.moveTo(x, H); ctx.lineTo(x + H, 0); ctx.stroke(); }
  ctx.restore();
  ctx.strokeStyle = "rgba(95,227,232,0.45)"; ctx.setLineDash([3, 4]);
  ctx.beginPath(); ctx.moveTo(xNow + 0.5, padT); ctx.lineTo(xNow + 0.5, H - padB); ctx.stroke();
  ctx.setLineDash([]);

  // forecast: median path with 50% and 80% bands, as far ahead as the strip covers at this view's scale
  if (cone) {
    const secsPerPx = (t1 - t0) / (xNow - padL);
    const aheadMax = Math.max(60, (futureW - 6) * secsPerPx);
    cone.secsPerPx = secsPerPx; cone.aheadMax = aheadMax;
    const N = 48, xs = [], b2u = [], b2d = [], b1u = [], b1d = [], mid = [];
    for (let k = 0; k <= N; k++) {
      const sec = (k / N) * aheadMax, v = fcAt(cone.fc, sec / 60);
      xs.push(xNow + sec / secsPerPx);
      mid.push(Y(p0 * Math.exp(v.q50)));
      b1u.push(Y(p0 * Math.exp(v.q75))); b1d.push(Y(p0 * Math.exp(v.q25)));
      b2u.push(Y(p0 * Math.exp(v.q90))); b2d.push(Y(p0 * Math.exp(v.q10)));
    }
    ctx.save();
    ctx.beginPath(); ctx.rect(xNow, padT, futureW, H - padT - padB); ctx.clip();
    const band = (u, d, a) => {
      ctx.beginPath(); ctx.moveTo(xs[0], u[0]);
      for (let k = 1; k <= N; k++) ctx.lineTo(xs[k], u[k]);
      for (let k = N; k >= 0; k--) ctx.lineTo(xs[k], d[k]);
      ctx.closePath(); ctx.fillStyle = `rgba(95,227,232,${a})`; ctx.fill();
    };
    band(b2u, b2d, 0.08); band(b1u, b1d, 0.16);
    ctx.shadowColor = C.cyan; ctx.shadowBlur = 8; ctx.strokeStyle = C.cyan; ctx.lineWidth = 1.6; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(xs[0], mid[0]); for (let k = 1; k <= N; k++) ctx.lineTo(xs[k], mid[k]); ctx.stroke();
    ctx.restore();
    const end = fcAt(cone.fc, aheadMax / 60);
    ctx.fillStyle = C.cyan; ctx.font = "17px VT323";
    ctx.fillText(`${fmtAhead(aheadMax)}: ${pct(Math.exp(end.q50) - 1, 2)}`, xNow + 6, padT + 14);
    ctx.fillStyle = "rgba(95,227,232,0.6)";
    ctx.fillText(`80%: ${pct(Math.exp(end.q10) - 1, 1)} to ${pct(Math.exp(end.q90) - 1, 1)}`, xNow + 6, padT + 30);
  }

  // history
  ctx.save();
  ctx.beginPath(); ctx.rect(padL, padT - 4, xNow - padL, H - padT - padB + 8); ctx.clip();
  if (bars) {
    const w = Math.max(1, ((xNow - padL) / Math.max(bars.length, 1)) * 0.62);
    for (const b of bars) {
      const x = X(b[0] + S.data.secs / 2), up = b[4] >= b[1];
      const col = up ? C.amber : C.ember;
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.shadowColor = col; ctx.shadowBlur = 6; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, Y(b[2])); ctx.lineTo(Math.round(x) + 0.5, Y(b[3])); ctx.stroke();
      const yo = Y(b[1]), yc = Y(b[4]), top = Math.min(yo, yc), h = Math.max(1, Math.abs(yc - yo));
      if (up) { ctx.fillStyle = "rgba(6,20,28,0.9)"; ctx.fillRect(x - w / 2, top, w, h); ctx.strokeRect(Math.round(x - w / 2) + 0.5, Math.round(top) + 0.5, Math.max(1, Math.round(w) - 1), Math.max(1, Math.round(h) - 1)); }
      else ctx.fillRect(x - w / 2, top, w, h);
    }
  } else if (pts && pts.length > 1) {
    // phosphor trace with persistence: older trace is dimmer, the newest minute burns hot
    ctx.lineJoin = "round"; ctx.lineWidth = 1.6;
    ctx.shadowColor = C.amber; ctx.shadowBlur = 10;
    ctx.strokeStyle = "rgba(255,181,71,0.55)";
    ctx.beginPath(); ctx.moveTo(X(pts[0][0]), Y(pts[0][1]));
    for (const p of pts) ctx.lineTo(X(p[0]), Y(p[1]));
    ctx.stroke();
    const hotFrom = tNow - 60;
    const hot = pts.filter((p) => p[0] >= hotFrom);
    if (hot.length > 1) {
      ctx.strokeStyle = C.hot; ctx.lineWidth = 2; ctx.shadowBlur = 16;
      ctx.beginPath(); ctx.moveTo(X(hot[0][0]), Y(hot[0][1]));
      for (const p of hot) ctx.lineTo(X(p[0]), Y(p[1]));
      ctx.stroke();
    }
  }
  ctx.restore();
  ctx.shadowBlur = 0;

  // position lines
  if (pos) {
    hline(ctx, Y(pos.entry_px), padL, W - padR, C.cyan, [6, 4], `entry ${pxr(pos.entry_px)}`);
    if (pos.stop_px > lo && pos.stop_px < hi) hline(ctx, Y(pos.stop_px), padL, W - padR, C.ember, [2, 4], `stop ${pxr(pos.stop_px)}`);
  }
  // fills inside the window
  for (const f of fillsFor(S.coin)) {
    if (f.t < t0 || f.t > tNow) continue;
    const x = X(f.t), y = Y(f.price), buy = f.side === "buy";
    ctx.fillStyle = buy ? C.cyan : C.ember; ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 8;
    ctx.beginPath();
    if (buy) { ctx.moveTo(x, y + 4); ctx.lineTo(x - 6, y + 14); ctx.lineTo(x + 6, y + 14); }
    else { ctx.moveTo(x, y - 4); ctx.lineTo(x - 6, y - 14); ctx.lineTo(x + 6, y - 14); }
    ctx.fill(); ctx.shadowBlur = 0;
  }

  // last price: dashed rail + pulsing dot at "now" + axis tag
  if (p0) {
    const y = Math.round(Y(p0)) + 0.5;
    ctx.strokeStyle = "rgba(255,226,176,0.55)"; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke(); ctx.setLineDash([]);
    const phase = reduced ? 0 : ((now || 0) % 1600) / 1600;
    ctx.fillStyle = `rgba(255,181,71,${0.35 * (1 - phase)})`;
    ctx.beginPath(); ctx.arc(xNow, y, 4 + phase * 10, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = C.hot; ctx.shadowColor = C.amber; ctx.shadowBlur = 12;
    ctx.beginPath(); ctx.arc(xNow, y, 3, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0;
    ctx.fillStyle = C.amber; ctx.fillRect(W - padR + 2, y - 10, padR - 4, 20);
    ctx.fillStyle = "#06141c"; ctx.font = "18px VT323"; ctx.textBaseline = "middle";
    ctx.fillText(pxr(p0), W - padR + 7, y + 1);
  }

  // crosshair
  const hud = $("hud");
  if (S.hover && S.hover.x >= padL && S.hover.x <= W - padR) {
    const { x, y } = S.hover;
    ctx.strokeStyle = "rgba(255,181,71,0.35)"; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(x + 0.5, padT); ctx.lineTo(x + 0.5, H - padB); ctx.moveTo(padL, y + 0.5); ctx.lineTo(W - padR, y + 0.5); ctx.stroke();
    ctx.setLineDash([]);
    const price = lo + (1 - (y - padT) / (H - padT - padB)) * (hi - lo);
    let text;
    if (x > xNow && cone && cone.secsPerPx) {
      const sec = Math.min(cone.aheadMax, (x - xNow) * cone.secsPerPx), v = fcAt(cone.fc, sec / 60);
      const P = (q) => pxr(p0 * Math.exp(v[q]));
      text = `in ${fmtAhead(sec)} (forecast)\nmedian ${P("q50")}\n50%  ${P("q25")} – ${P("q75")}\n80%  ${P("q10")} – ${P("q90")}`;
    } else {
      const t = t0 + ((x - padL) / (xNow - padL)) * (t1 - t0);
      text = `${S.view === "1w" ? dayhh(t) : hhmm(t)}   ${px(price)}`;
      if (bars && bars.length) {
        const b = bars.reduce((a, c) => (Math.abs(c[0] + S.data.secs / 2 - t) < Math.abs(a[0] + S.data.secs / 2 - t) ? c : a));
        text += `\nopen ${px(b[1])}  high ${px(b[2])}\nlow  ${px(b[3])}  close ${px(b[4])}`;
      }
    }
    hud.textContent = text; hud.hidden = false;
  } else hud.hidden = true;
}

function hline(ctx, y, x0, x1, col, dash, label) {
  ctx.strokeStyle = col; ctx.setLineDash(dash); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x0, Math.round(y) + 0.5); ctx.lineTo(x1, Math.round(y) + 0.5); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = col; ctx.font = "16px VT323"; ctx.textBaseline = "bottom"; ctx.fillText(label, x0 + 6, y - 2);
}

function frame(now) {
  if (S.dirty || S.hover || !reduced) { draw(now); S.dirty = false; }
  requestAnimationFrame(frame);
}

// --- status: panels, lamps, console --------------------------------------------------------------
async function pollStatus() {
  clearTimeout(pollStatus.t);
  try {
    const st = await getJSON(`/api/status?since=${S.since}&coin=${S.coin}`);
    if (st.error) throw new Error(st.error);
    const first = !S.status;
    if (st.coin !== S.coin && !first) {  // answered for the coin we just left: keep its events only
      for (const e of st.events) { S.since = Math.max(S.since, e.seq); onEvent(e); }
      throw new Error("stale");
    }
    S.status = st;
    S.coins = Object.keys(st.coins || { BTC: {} });
    if (first) {
      if (!S.coins.includes(S.coin)) { S.coin = S.coins[0]; }
      connectFeed(st.exchange);
      if (st.coin !== S.coin) { setTimeout(pollStatus, 0); }
    }
    paintTape();
    if (S.bootId && st.boot_id !== S.bootId) {
      log("the bot restarted. reload this page to reconnect your session.", "err");
      S.since = 0;
    }
    S.bootId = st.boot_id;
    S.pending = st.pending || null;
    $("prompt").classList.toggle("pending", !!S.pending);
    const tk = st.ticker;
    const wsFresh = Date.now() - S.wsTickAt < 8000;
    if (tk && !wsFresh) { // REST fallback while the WebSocket is quiet
      onPrice(tk.last, tk.t);
      setStats({ bid: tk.bid, ask: tk.ask, high24: tk.high24, low24: tk.low24, chg: tk.open24 ? tk.last / tk.open24 - 1 : NaN });
    }
    for (const e of st.events) { S.since = Math.max(S.since, e.seq); onEvent(e); }
    paintPanels();
  } catch (e) {
    if (e.message !== "stale") $("lamp-feed").classList.remove("on");
  } finally {
    pollStatus.t = setTimeout(pollStatus, 3000);  // always re-arm, whatever happened above
  }
}

function paintPanels() {
  const st = S.status; if (!st || st.coin !== S.coin) return;  // never value one coin's ledger at another's price
  const s = st.state, L = s.ledger || {}, pol = st.policy, sig = s.signal, last = S.last;
  // lamps
  $("lamp-feed").classList.toggle("on", Date.now() - S.wsTickAt < 15000);
  $("lamp-feed").title = Date.now() - S.wsTickAt < 15000 ? "Live WebSocket feed" : "WebSocket quiet: polling every 3s";
  const bot = $("lamp-bot");
  bot.textContent = st.busy ? st.busy : s.halted ? "halted" : s.auto === false ? "bot off" : "bot on";
  bot.classList.toggle("on", s.auto !== false && !s.halted);
  bot.classList.toggle("busy", !!st.busy);
  $("lamp-mode").hidden = st.mode === "live";  // only paper trading gets a label
  // account
  const pv = st.portfolio;
  if (pv) {  // headline: everything the bot manages, P&L against what was actually put in
    tween($("equity"), pv.equity, usd);
    const el = $("pnl");
    el.textContent = `${pv.pnl >= 0 ? "+" : "−"}$${num(Math.abs(pv.pnl), 2)} (${pct(pv.basis ? pv.pnl / pv.basis : 0, 1)})`;
    el.classList.toggle("neg", pv.pnl < 0);
    $("pnl-note").textContent = `of ${usd(pv.basis)} in`;
    $("pnl-note").title = `${usd(pv.basis)} deposited, net of withdrawals; ${usd(pv.fees)} paid in fees so far`;
    paintAlloc(pv);
  }
  $("sleeve-label").textContent = `${COIN_META[S.coin].name} sleeve`;
  if (last) {
    $("btc").textContent = L.qty > 0 ? `${+(+L.qty).toPrecision(6)} ${S.coin.toLowerCase()} · ${usd(L.qty * last)}` : "none";
  }
  $("cash").textContent = usd(L.usd);
  $("fees").textContent = usd(L.fees);
  const pos = s.position;
  const since = pos && last ? ` (${pct(last / pos.entry_px - 1, 1)})` : "";
  $("pos").textContent = pos ? `${pos.manual ? "manual" : "long " + Math.round((pos.frac || 1) * 100) + "%"} from ${pxr(pos.entry_px)}${since}` : "flat";
  $("pos").className = pos && last ? (last < pos.entry_px ? "neg" : "pos") : "";
  $("stop").textContent = pos && pos.stop_px > pos.entry_px * 0.01 ? pxr(pos.stop_px) : "—";
  // model
  $("model-name").textContent = pol.model === "momentum" ? `${pol.sma_days}-day trend rule` : `${pol.model}, ${pol.horizon}h horizon`;
  $("cost").textContent = pct(pol.round_trip, 2).replace("+", "");
  $("gate").textContent = pol.model === "momentum" ? `${pol.sma_days}-day trend rule`
    : pol.gate_pass ? "ML model" : st.mode === "live" ? "ML, watching only" : "ML, paper only";
  $("ml").textContent = pol.model === "momentum" || !pol.gate_pass ? "lost to fees in testing" : "beat fees in testing";
  paintGauge(pol, sig);
  const next = (Math.floor(Date.now() / 3600000) + 1) * 3600000 + 20000;
  const left = Math.max(0, next - Date.now());
  $("countdown").textContent = `in ${Math.floor(left / 60000)}m ${String(Math.floor((left % 60000) / 1000)).padStart(2, "0")}s`;
  paintBets();
  paintCalls();
  $("hint").textContent = S.pending ? `confirm ${S.pending}: ${st.pending_desc || "pending action"}` : st.busy ? `bot is ${st.busy}` : "↑ ↓ history";
}

function paintGauge(pol, sig) {
  const v = $("verdict");
  if (!sig) { v.textContent = "waiting for the first hourly signal"; return; }
  let val, entry, exit, R, mid = "0";
  if (pol.model === "momentum") {
    val = sig.sma ? sig.daily_close / sig.sma - 1 : 0;
    entry = 0; exit = 0; R = Math.max(Math.abs(val) * 1.4, 0.05);
    v.textContent = `Trend ${sig.pred > 0 ? "up" : "down"}: yesterday closed ${pct(val, 1)} ${val >= 0 ? "above" : "below"} the ${pol.sma_days}-day average. I want to be ${sig.pred > 0 ? "long" : "in cash"}.`;
  } else {
    val = sig.pred; entry = pol.entry_thr ?? 0; exit = pol.exit_thr;
    R = Math.max(Math.abs(entry) * 1.3, sig.sigma_h * 1.5, Math.abs(val) * 1.2);
    v.textContent = `Expecting ${pct(val)} over ${sig.horizon}h (±${pct(sig.sigma_h, 1).replace("+", "")}). ` +
      (pol.entry_thr == null ? "Entries are off." : val > entry ? "That clears the entry bar." : `Entry needs ${pct(entry)}, so I wait.`);
  }
  const at = (x) => `${Math.min(100, Math.max(0, ((x + R) / (2 * R)) * 100))}%`;
  $("g-needle").style.left = at(val);
  $("g-entry").style.left = at(entry);
  $("g-exit").style.left = at(exit);
  $("g-lo").textContent = pct(-R, 1); $("g-mid").textContent = mid; $("g-hi").textContent = pct(R, 1);
}

function onEvent(e) {
  const t = Date.parse(e.ts) / 1000;
  const f = e.fill;
  switch (e.event) {
    case "tick": case "signal": {
      const pol = S.status && S.status.policy;
      const closeT = Date.parse(e.bar) / 1000 + 3600;
      if (pol && pol.model === "momentum") log(`${(e.coin || "btc").toLowerCase()}: hour to ${hhmm(closeT)} closed ${pxr(e.close)}. trend ${e.pred > 0 ? "up, want long" : "down, want cash"}.${e.note ? " " + e.note : ""}`, "model", t);
      else log(`${(e.coin || "btc").toLowerCase()}: hour to ${hhmm(closeT)} closed ${pxr(e.close)}. projection ${pct(e.pred)} over ${e.horizon}h.${e.note ? " " + e.note : ""}`, "model", t);
      break;
    }
    case "order":
      if (e.stage === "placing") log(`${(e.coin || "").toLowerCase()} order ${e.side} ${e.size ?? ""} ${e.price ? "@ " + px(e.price) : ""}`.trim(), "", t);
      else if (e.stage === "done") log(`order ${e.order_id} ${String(e.status).toLowerCase()}`, "", t);
      else if (e.stage === "UNSETTLED") log(`order ${e.order_id} could not be confirmed cancelled; it will be cancelled on restart`, "err", t);
      break;
    case "startup": log(`${(e.coin || "btc").toLowerCase()} sleeve online: ${usd(e.ledger.usd)} cash, ${+(e.ledger.qty ?? e.ledger.btc ?? 0)} ${e.coin || "BTC"}.`, "model", t); break;
    case "error": log(`step failed: ${e.note}`, "err", t); break;
    case "halted": case "KILL_SWITCH": log(e.note || "kill switch fired: flattened and halted", "err", t); break;
    case "skip": log(e.note, "model", t); break;
    case "deposit": case "withdraw": case "auto": case "resume": case "move":
    case "pool_fund": case "pool_defund": case "pool_auto": case "pool_resume": case "pool_autofund": break;
    case "sell_for_bet":
      log(`sold ${+(+f.size).toPrecision(6)} ${(e.coin || "btc").toLowerCase()} at ${px(f.price)} (fee ${usd(f.fee)}) to fund a ${e.bet.toLowerCase()} bet`, "ok", t);
      S.allFills.push({ t, price: f.price, size: f.size, side: "sell", kind: e.event, coin: e.coin || "BTC" });
      paintFills(); S.dirty = true; break; // the command reply already said it
    case "bet_open": {
      log(`bet ${e.coin.toLowerCase()}: ${e.side} ${+(+e.fill.size).toPrecision(6)} at ${px(e.fill.price)} for ${e.H}h (P(up) ${Math.round(e.p_up * 100)}%)`, "model", t);
      S.allFills.push({ t, price: e.fill.price, size: e.fill.size, side: e.side === "long" ? "buy" : "sell", kind: "bet", coin: e.coin });
      paintFills(); S.dirty = true; break;
    }
    case "bet_close": {
      log(`bet ${e.coin.toLowerCase()} closed (${e.reason}): ${e.pnl >= 0 ? "+" : "−"}$${num(Math.abs(e.pnl), 2)}`, e.pnl >= 0 ? "ok" : "err", t);
      S.allFills.push({ t, price: e.fill.price, size: e.fill.size, side: e.side === "long" ? "sell" : "buy", kind: "bet", coin: e.coin });
      paintFills(); S.dirty = true; break;
    }
    case "bet_unfilled": case "bet_close_unfilled": log(`bet ${e.coin.toLowerCase()}: order did not fill`, "model", t); break;
    case "pool_halted": case "pool_shorts_off": case "pool_error": log(`bet pool: ${e.note}`, "err", t); break;
    case "pool_skip": log(`bet pool: ${e.note}`, "model", t); break;
    case "pool_startup": log(`bet pool online: ${usd(e.ledger.usd)} cash, betting on ${e.coins.join(", ").toLowerCase()}${e.shorts ? " (long and short)" : " (long only)"}`, "model", t); break;
    case "entry_unfilled": case "scale_up_unfilled": log("entry order expired unfilled; staying flat", "model", t); break;
    case "exit_unfilled": log("exit did not fill; will retry next hour", "err", t); break;
    default:
      if (TRADE_EVENTS.has(e.event) && f) {
        const side = /exit|sell|KILL/.test(e.event) ? "sell" : "buy";
        if (!e.event.startsWith("manual_")) log(`${e.event.replace("_", " ")}: ${side} ${+f.size} ${e.coin || "BTC"} at ${px(f.price)}, fee ${usd(f.fee)}${e.pnl_pct != null ? `, ${pct(e.pnl_pct)} on the trade` : ""}`, side === "sell" ? "err" : "ok", t);
        S.allFills.push({ t, price: f.price, size: f.size, side, kind: e.event, coin: e.coin || "BTC" });
        paintFills();
        S.dirty = true;
      }
  }
}

function cycleCoin(step) {
  const i = S.coins.indexOf(S.coin);
  if (i >= 0 && S.coins.length > 1) switchCoin(S.coins[(i + step + S.coins.length) % S.coins.length]);
}

function switchCoin(c) {
  if (c === S.coin) return;
  S.coin = c;
  try { localStorage.setItem("btcbot.coin", c); } catch (e) { /* private mode */ }
  S.ticks = []; S.last = null; S.prevLast = null; S.data = null; S.wsTickAt = 0; S.dirty = true;
  $("price").textContent = "—";
  $("pair").textContent = `${c.toLowerCase()}/usd`;
  $("coin-name").textContent = COIN_META[c].name;
  if (!reduced) { const cv = $("chart"); cv.classList.remove("swap"); void cv.offsetWidth; cv.classList.add("swap"); }
  paintTape(); paintFills();
  pollStatus();
  loadView(S.view);
  S.forecast = null; loadForecast();
  pollNotes.key = null; pollNotes();
  if (S.status) connectFeed(S.status.exchange);
}

// coin tape: every coin's live price, 24h move and what you hold of it, always on screen
function paintTape() {
  const nav = $("tape");
  $("pair").textContent = `${S.coin.toLowerCase()}/usd`;
  $("coin-name").textContent = COIN_META[S.coin].name;
  const pv = S.status && S.status.portfolio;
  const coins = S.coins.length ? S.coins : Object.keys(COIN_META);
  if (nav.querySelectorAll("button").length !== coins.length) {  // build once; then only update text
    nav.querySelectorAll("button").forEach((b) => b.remove());
    for (const c of coins) {
      const b = document.createElement("button");
      b.dataset.coin = c;
      b.setAttribute("role", "tab");
      b.title = `Show ${COIN_META[c].name}`;
      b.innerHTML = '<span class="t-sym"></span><span class="t-name"></span><span class="t-px"></span>' +
        '<span class="t-chg"></span><span class="t-held"></span><span class="t-tags"></span><span class="t-call"></span>';
      b.querySelector(".t-sym").textContent = c.toLowerCase();
      b.querySelector(".t-name").textContent = COIN_META[c].name;
      nav.append(b);
    }
  }
  for (const b of nav.querySelectorAll("button")) {
    const c = b.dataset.coin, row = (pv && pv.coins && pv.coins[c]) || {}, live = S.live[c] || {};
    const fresh = live.at && Date.now() - live.at < 15000;
    const last = fresh ? live.last : row.last, chg = fresh ? live.chg : row.chg24;
    b.setAttribute("aria-selected", String(c === S.coin));
    const pxEl = b.querySelector(".t-px"), prev = S.tapePx[c];
    if (last != null && prev !== last) {
      pxEl.textContent = px(last);
      if (prev != null) flash(pxEl, last > prev);
      S.tapePx[c] = last;
    }
    const ch = b.querySelector(".t-chg");
    ch.textContent = chg == null ? "" : pct(chg, 2);
    ch.classList.toggle("neg", chg < 0);
    // what you hold of this coin (trend sleeve + any bet + sleeve cash), revalued at the live price
    const value = row.qty != null ? (row.qty || 0) * (last || 0) + ((row.value || 0) - (row.qty || 0) * (row.last || 0)) : null;
    const held = b.querySelector(".t-held");
    if (value != null && value >= 0.005) tween(held, value, (v) => `held ${usd(v)}`, 450);
    else { held.textContent = "held —"; held._v = null; }
    held.classList.toggle("none", !(value >= 0.005));
    const tags = [];
    if (row.trend) tags.push(`<i class="${row.trend}">${row.trend === "up" ? "▲ trend up" : "▼ trend down"}</i>`);
    if (row.bet) tags.push(`<i class="betting">● bet ${pct(row.bet.pnl_pct, 1)}</i>`);
    const tg = b.querySelector(".t-tags"), html = tags.join("");
    if (tg.innerHTML !== html) tg.innerHTML = html;
    const call = ((S.status && S.status.calls) || {})[c], ce = b.querySelector(".t-call");
    const odds = liveOdds(call, coinPrice(c)), left = call && call.until ? Date.parse(call.until) / 1000 - Date.now() / 1000 : 0;
    ce.className = `t-call ${odds == null || left <= 0 ? "none" : odds >= 0.5 ? "up" : "down"}`;
    const kc = kalshiCents(call);
    ce.textContent = !call ? "" : call.error ? "15-minute odds unavailable" : odds == null ? "" : left <= 0 ? "15-minute window closed; next one starting"
      : `By ${hhmm(Date.parse(call.until) / 1000)}: ${Math.round(odds * 100)}% chance it's higher${kc != null ? ` · Kalshi ${kc}%` : ""}`;
    ce.title = call && call.start ? `Chance the price at ${hhmm(Date.parse(call.until) / 1000)} is above ${px(call.start)}, where this 15-minute window started` : "";
  }
  const sel = nav.querySelector('button[aria-selected="true"]'), glow = $("tape-glow");
  if (sel) { glow.style.width = `${sel.offsetWidth}px`; glow.style.transform = `translateX(${sel.offsetLeft}px)`; glow.style.opacity = 1; }
}

// where the money is: one bar, one segment per sleeve and the bet pool
function paintAlloc(pv) {
  const parts = Object.entries(pv.parts).filter(([, v]) => v >= 0.01);
  const total = parts.reduce((a, [, v]) => a + v, 0) || 1;
  const bar = $("alloc"), leg = $("alloc-legend");
  const key = parts.map(([k]) => k).join();
  if (bar.dataset.key !== key) {
    bar.dataset.key = key; bar.textContent = ""; leg.textContent = "";
    for (const [k] of parts) {
      const seg = document.createElement("span"); seg.dataset.k = k; seg.style.background = ALLOC[k] || "#3a5562"; seg.style.width = "0%";
      bar.append(seg);
      const li = document.createElement("li"); li.dataset.k = k;
      li.innerHTML = '<i></i><span class="k"></span><span class="v"></span><span class="p"></span>';
      li.querySelector("i").style.background = ALLOC[k] || "#3a5562";  // CSSOM, not a style attribute (CSP)
      li.querySelector(".k").textContent = k === "BETS" ? "bets" : k.toLowerCase();
      if (k !== "BETS") { li.classList.add("link"); li.title = `Show ${COIN_META[k].name}`; li.addEventListener("click", () => switchCoin(k)); }
      leg.append(li);
    }
  }
  for (const [k, v] of parts) {
    bar.querySelector(`[data-k="${k}"]`).style.width = `${(v / total) * 100}%`;
    const li = leg.querySelector(`[data-k="${k}"]`);
    li.querySelector(".v").textContent = usd(v);
    li.querySelector(".p").textContent = `${Math.round((v / total) * 100)}%`;
    li.classList.toggle("sel", k === S.coin);
  }
}

// could a bet on `c` raise its stake? idle sleeve cash, or another coin worth selling (never c itself)
function canFund(c) {
  const rows = (S.status && S.status.portfolio && S.status.portfolio.coins) || {};
  return Object.entries(rows).some(([k, r]) => (r.value || 0) - (r.qty || 0) * (r.last || 0) >= 6
    || (k !== c && (r.qty_trend || 0) * (r.last || 0) >= 6));
}

// the short-term bet pool: one card per open bet, and how close each other coin is to a bet
// 15-minute up/down calls: recommendations with their tested and live record (the bot never trades them)
function paintCalls() {
  const calls = S.status && S.status.calls, win = $("calls-win");
  win.hidden = !calls || !Object.keys(calls).length;
  if (win.hidden) return;
  const ol = $("calls");
  const next = (Math.floor(Date.now() / 900000) + 1) * 900000 + 30000;
  const any = Object.values(calls).find((k) => k.until);
  $("calls-sub").textContent = any ? `${hhmm(Date.parse(any.from) / 1000)}–${hhmm(Date.parse(any.until) / 1000)} · ${mmss(Date.parse(any.until) / 1000 - Date.now() / 1000)} left`
    : `next in ${dur((next - Date.now()) / 1000)}`;
  let n = 0, hits = 0, net = 0, tested = [];
  for (const c of S.coins) {
    const k = calls[c]; if (!k) continue;
    let li = ol.querySelector(`[data-c="${c}"]`);
    if (!li) {
      li = document.createElement("li"); li.dataset.c = c;
      li.innerHTML = '<b></b><span class="say"></span><span class="sub"></span>';
      li.querySelector("b").textContent = c.toLowerCase();
      li.addEventListener("click", () => switchCoin(c));
      ol.append(li);
    }
    if (li.dataset.bar !== k.bar && li.dataset.bar && !reduced) { li.classList.remove("fresh"); void li.offsetWidth; li.classList.add("fresh"); }
    li.dataset.bar = k.bar || "";
    const price = coinPrice(c), odds = liveOdds(k, price), say = li.querySelector(".say");
    li.className = `${li.classList.contains("fresh") ? "fresh " : ""}${odds == null ? "none" : odds >= 0.6 ? "up" : odds <= 0.4 ? "down" : "even"}`;
    if (k.error || odds == null) {
      say.textContent = k.error ? "odds unavailable right now" : "waiting for the window to start";
      li.querySelector(".sub").textContent = "";
    } else {
      const pctTxt = `${Math.round(odds * 100)}%`;
      if (li._odds != null && Math.round(li._odds * 100) !== Math.round(odds * 100)) flash(say, odds > li._odds);
      li._odds = odds;
      say.textContent = `${verdict(odds)[0].toUpperCase() + verdict(odds).slice(1)}: ${pctTxt} chance it ends above ${px(k.start)}`;
      const diff = price - k.start, kc = kalshiCents(k);
      const parts = [`now ${px(price)}, ${diff >= 0 ? "$" + num(Math.abs(diff), pdec(price)) + " above" : "$" + num(Math.abs(diff), pdec(price)) + " below"}`];
      if (kc != null) parts.push(`Kalshi says ${kc}%`);
      if (k.dir) parts.push(`model called ${k.dir === "up" ? "higher" : "lower"} at the start`);
      li.querySelector(".sub").textContent = parts.join(" · ");
    }
    li.title = `Show ${COIN_META[c].name}`;
    const r = k.record || {};
    n += r.n || 0; hits += r.hits || 0; net += r.net_bp || 0;
    if (k.tested && k.tested.hit_rate != null) tested.push(k.tested);
  }
  const avg = (key) => tested.reduce((a, x) => a + x[key], 0) / (tested.length || 1);
  $("calls-note").textContent = "Each line is the chance a coin finishes this 15-minute window higher than it started, "
    + "judged on the average price in the last minute, the way Kalshi settles. It updates as the price moves. "
    + (n ? `The model's opening calls were right ${hits} of ${n} time${n === 1 ? "" : "s"} in the last day. ` : "")
    + (tested.length ? `In testing they were right ${Math.round(avg("hit_rate") * 100)}% of the time but still lost money once trading fees were paid. ` : "")
    + "For information only; the bot never trades on these.";
}

function paintBets() {
  const st = S.status, pool = st && st.pool, box = $("bets"), head = $("bets-head");
  const win = box.closest(".bets-win");
  win.hidden = !pool;
  if (!pool) return;
  const L = pool.ledger || {}, bets = pool.bets || {}, open = Object.keys(bets);
  const eq = st.portfolio && st.portfolio.parts ? st.portfolio.parts.BETS : null;
  const state = pool.halted ? "halted" : pool.needs_review ? "needs review" : pool.auto === false ? "paused" : "betting";
  $("bets-sub").textContent = `${state}${pool.shorts ? ", long + short" : ", long only"}${pool.autofund !== false ? ", sells coins to fund" : ""}`;
  win.classList.toggle("alert", state === "halted" || state === "needs review");
  head.innerHTML = "";
  for (const [k, v, cls] of [["pool", usd(eq), ""], ["free cash", usd(L.usd), ""], ["open", String(open.length), ""],
    ["realized", `${(L.realized || 0) >= 0 ? "+" : "−"}$${num(Math.abs(L.realized || 0), 2)}`, (L.realized || 0) < 0 ? "neg" : ""]]) {
    const d = document.createElement("div");
    d.innerHTML = `<span class="k"></span><span class="v ${cls}"></span>`;
    d.querySelector(".k").textContent = k; d.querySelector(".v").textContent = v;
    head.append(d);
  }
  const now = Date.now() / 1000;
  const want = new Set();
  for (const c of open) {
    want.add(`bet-${c}`);
    const b = bets[c], last = (S.live[c] && S.live[c].last) || ((st.portfolio && st.portfolio.coins && st.portfolio.coins[c]) || {}).last || b.entry_px;
    const side = b.side > 0 ? 1 : -1, ret = (last / b.entry_px - 1) * side, pnl = b.qty * (last - b.entry_px) * side;
    const t0 = Date.parse(b.bar_close || b.opened) / 1000, t1 = t0 + b.H * 3600;
    const done = Math.min(1, Math.max(0, (now - t0) / (t1 - t0)));
    const toStop = side > 0 ? b.stop_px / last - 1 : last / b.stop_px - 1;
    let card = box.querySelector(`[data-id="bet-${c}"]`);
    if (!card) {
      card = document.createElement("article");
      card.className = "bet" + (reduced ? "" : " new"); card.dataset.id = `bet-${c}`;
      card.innerHTML = '<header><b class="c"></b><span class="side"></span><span class="ret"></span><span class="pnl"></span></header>' +
        '<div class="time"><span class="fill"></span></div><footer><span class="left"></span><span class="stop"></span></footer>';
      card.addEventListener("click", () => switchCoin(c));
      card.title = `Show ${COIN_META[c].name}`;
      box.prepend(card);
    }
    card.querySelector(".c").textContent = c.toLowerCase();
    card.querySelector(".side").textContent = `${side > 0 ? "long" : "short"} ${usd(b.stake)} · ${Math.round(b.p_up * 100)}% up`;
    const r = card.querySelector(".ret"); r.textContent = pct(ret, 2); r.classList.toggle("neg", ret < 0);
    const pl = card.querySelector(".pnl"); pl.textContent = `${pnl >= 0 ? "+" : "−"}$${num(Math.abs(pnl), 2)}`; pl.classList.toggle("neg", pnl < 0);
    card.classList.toggle("losing", ret < 0);
    card.querySelector(".fill").style.width = `${done * 100}%`;
    card.querySelector(".left").textContent = `${pxr(b.entry_px)} → ${pxr(last)} · ${dur(t1 - now)} left`;
    card.querySelector(".stop").textContent = `stop ${pxr(b.stop_px)} ${pct(toStop, 1)}`;
    card.querySelector(".time").title = `closes around ${dayhh(t1)}`;
  }
  const sigs = pool.signals || {}, models = pool.models || {};
  for (const c of S.coins) {
    if (bets[c] || !models[c]) continue;
    want.add(`watch-${c}`);
    const sg = sigs[c], thr = models[c].threshold;
    let row = box.querySelector(`[data-id="watch-${c}"]`);
    if (!row) {
      row = document.createElement("div"); row.className = "watch"; row.dataset.id = `watch-${c}`;
      row.innerHTML = '<b class="c"></b><span class="meter"><span class="lvl"></span><span class="thr"></span></span><span class="txt"></span>';
      row.querySelector(".c").textContent = c.toLowerCase();
      box.append(row);
    }
    const p = sg ? sg.p_up : null;
    row.querySelector(".lvl").style.width = p == null ? "0%" : `${Math.min(100, Math.max(0, (p - 0.3) / 0.5 * 100))}%`;
    row.querySelector(".thr").style.left = `${Math.min(100, Math.max(0, (thr - 0.3) / 0.5 * 100))}%`;
    row.classList.toggle("ready", p != null && p >= thr);
    const broke = p != null && p >= thr && (L.usd || 0) < 1;  // the model wants a bet the pool can't pay for
    row.querySelector(".txt").textContent = p == null ? `waiting for the hour · bets at ${Math.round(thr * 100)}%`
      : broke ? `${Math.round(p * 100)}% up · ${pool.autofund === false ? "no free cash" : canFund(c) ? "raises stake next hour" : `nothing but ${c.toLowerCase()} to sell`}` : `${Math.round(p * 100)}% up · bets at ${Math.round(thr * 100)}%`;
    row.classList.toggle("broke", broke);
    row.title = `${COIN_META[c].name}: the bet model's chance the price is higher in ${models[c].H}h`;
  }
  for (const el of [...box.children]) if (!want.has(el.dataset.id)) el.remove();
  if (!box.children.length) box.innerHTML = '<p class="empty">no bets open</p>';
}

function paintFills() {
  const ol = $("fills");
  ol.textContent = "";
  for (const f of S.allFills.slice(-30).reverse()) {
    const li = document.createElement("li");
    const a = document.createElement("span"); a.className = f.side === "sell" ? "side-sell" : ""; a.textContent = `${f.side} ${f.coin.toLowerCase()}`;
    const b = document.createElement("span"); b.textContent = `${+(+f.size).toPrecision(6)} @ ${pxr(f.price)}`;
    const c = document.createElement("span"); c.className = "when"; c.textContent = hhmm(f.t);
    li.append(a, b, c); ol.append(li);
  }
  if (!S.allFills.length) { const li = document.createElement("li"); li.className = "empty"; li.textContent = "no trades yet"; ol.append(li); }
}

function log(text, cls = "", t = Date.now() / 1000) {
  const box = $("console");
  const p = document.createElement("p");
  const ts = document.createElement("span"); ts.className = "t"; ts.textContent = hhmm(t);
  const body = document.createElement("span"); body.className = cls; body.textContent = text;
  p.append(ts, body);
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 30;
  box.append(p);
  while (box.childElementCount > 500) box.firstChild.remove();
  if (atBottom) box.scrollTop = box.scrollHeight;
}

// --- notes crawl ----------------------------------------------------------------------------------
async function pollNotes() {
  clearTimeout(pollNotes.t);
  try {
    const coin = S.coin;
    const notes = await getJSON(`/api/notes?coin=${coin}`);
    if (coin !== S.coin) return;
    const key = JSON.stringify(notes);
    if (key !== pollNotes.key) {
      pollNotes.key = key;
      const track = $("crawl");
      track.textContent = "";
      for (const n of notes) { const s = document.createElement("span"); s.className = `note ${n.kind}`; s.textContent = n.text; track.append(s); }
      const chars = notes.reduce((a, n) => a + n.text.length, 0);
      track.style.setProperty("--crawl-s", `${Math.max(40, chars / 9)}s`);
    }
  } catch (e) { /* keep the last notes */ }
  finally { pollNotes.t = setTimeout(pollNotes, 20000); }
}

// --- command line ---------------------------------------------------------------------------------
function loadHistory() { try { S.history = JSON.parse(localStorage.getItem("btcbot.history") || "[]"); } catch (e) { S.history = []; } }
function saveHistory() { try { localStorage.setItem("btcbot.history", JSON.stringify(S.history.slice(-100))); } catch (e) { /* private mode */ } }

async function submit(line) {
  line = line.trim();
  if (!line) return;
  S.history.push(line); S.histIdx = -1; saveHistory();
  log(`> ${line}`, "you");
  const [cmd, arg] = [line.split(/\s+/)[0].toLowerCase(), line.split(/\s+/).slice(1).join(" ").toLowerCase()];
  if (cmd === "clear") { $("console").textContent = ""; return; }
  if (cmd === "coin") {
    const c = { BITCOIN: "BTC", XBT: "BTC", ETHEREUM: "ETH", ETHER: "ETH", ETHERIUM: "ETH", SOLANA: "SOL", RIPPLE: "XRP" }[arg.toUpperCase()] || arg.toUpperCase();
    if (S.coins.includes(c)) switchCoin(c); else log(`coins here: ${S.coins.join(", ").toLowerCase()}`, "err");
    return;
  }
  if (cmd === "view") {
    if (VIEW_META[arg]) { loadView(arg); log(`chart: ${VIEW_META[arg].title}`, "ok"); }
    else log("views are 10m, 1h, 1d and 1w", "err");
    return;
  }
  const form = $("prompt");
  try {
    const r = await fetch("/api/command", {
      method: "POST", headers: { "Content-Type": "application/json", "X-Btcbot-Token": TOKEN }, body: JSON.stringify({ line, coin: S.coin }),
    });
    if (r.status === 403) { log("this page's session expired (the bot restarted). reload the page.", "err"); return; }
    const d = await r.json();
    for (const l of d.lines || [d.error || "no response"]) log(l, d.error ? "err" : d.ok ? "ok" : "");
    S.pending = d.confirm || null;
    form.classList.toggle("pending", !!S.pending);
    if (/^(confirm|cancel)$/.test(cmd)) { S.pending = null; form.classList.remove("pending"); }
  } catch (e) {
    log(`could not reach the bot: ${e.message}`, "err");
  }
  paintPanels();
}

function wire() {
  document.querySelectorAll(".keys button").forEach((b) => b.addEventListener("click", () => loadView(b.dataset.view)));
  $("prompt").addEventListener("submit", (ev) => { ev.preventDefault(); const i = $("cmd"); submit(i.value); i.value = ""; });
  $("cmd").addEventListener("keydown", (ev) => {
    if ((ev.key === "ArrowLeft" || ev.key === "ArrowRight") && !$("cmd").value) {  // empty prompt: step through coins
      ev.preventDefault(); cycleCoin(ev.key === "ArrowLeft" ? -1 : 1); return;
    }
    if (ev.key !== "ArrowUp" && ev.key !== "ArrowDown") return;
    ev.preventDefault();
    if (!S.history.length) return;
    S.histIdx = S.histIdx < 0 ? S.history.length : S.histIdx;
    S.histIdx = Math.min(S.history.length, Math.max(0, S.histIdx + (ev.key === "ArrowUp" ? -1 : 1)));
    $("cmd").value = S.history[S.histIdx] || "";
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.target.tagName === "INPUT") return;
    if (ev.key === "ArrowLeft" || ev.key === "ArrowRight") { ev.preventDefault(); cycleCoin(ev.key === "ArrowLeft" ? -1 : 1); return; }
    if (ev.key.length === 1 && !ev.metaKey && !ev.ctrlKey) $("cmd").focus();
  });
  $("tape").addEventListener("click", (ev) => { const b = ev.target.closest("button[data-coin]"); if (b) switchCoin(b.dataset.coin); });
  window.addEventListener("resize", () => paintTape());
  const cv = $("chart");
  cv.addEventListener("pointermove", (ev) => { const r = cv.getBoundingClientRect(); S.hover = { x: ev.clientX - r.left, y: ev.clientY - r.top }; S.dirty = true; });
  cv.addEventListener("pointerleave", () => { S.hover = null; S.dirty = true; });
  window.addEventListener("resize", () => { S.dirty = true; });
  setInterval(paintPanels, 1000);
  setInterval(paintTape, 1000);  // the 15-minute countdown and odds keep moving between price ticks
}

loadHistory();
wire();
boot();
log("type help for commands, or click a chart range or a coin. ← → step through the coins.", "model");
loadView("1h");
loadForecast();
pollStatus();
pollNotes();
requestAnimationFrame(frame);
$("cmd").focus();
