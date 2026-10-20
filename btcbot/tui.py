"""Phosphor Desk in the terminal.

    python -m btcbot.tui                 # attach to a running bot (./run_live.sh or btcbot.app)
    python -m btcbot.tui --start         # no bot running? start one in this process (env as for btcbot.app)

The TUI is a client of the bot's local server (127.0.0.1:BTCBOT_PORT), so it never runs a second
trader against the same account.  Click a chart range, a coin, a market row or a command chip; up/down
walk the command history, ctrl+l clears the console, ctrl+q quits.  Type `help` for commands.
"""
import argparse
import asyncio
import json
import logging
import math
import os
import re
import threading
import time
from datetime import datetime

import requests
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Input, RichLog, Static

PORT = int(os.environ.get("BTCBOT_PORT", 8787))
BASE = f"http://127.0.0.1:{PORT}"

AMBER, HOT, DIM_AMBER = "#ffb547", "#ffe2b0", "#a8762f"
EMBER, CYAN, DIM, GLASS = "#ff5a36", "#5fe3e8", "#5e7a86", "#06141c"

VIEWS = {  # view -> (title, subtitle, refresh seconds, span seconds)
    "10m": ("last ten minutes", "every trade", 10, 600),
    "1h": ("last hour", "1-minute candles", 20, 3600),
    "1d": ("last day", "5-minute candles", 60, 86400),
    "1w": ("last week", "1-hour candles", 300, 604800),
}
# clickable chips above the prompt: label -> what clicking does (fill the prompt, or run read-only commands)
CHIPS = (("buy", "fill('buy ')"), ("sell", "fill('sell ')"), ("deposit", "fill('deposit ')"),
         ("withdraw", "fill('withdraw ')"), ("move", "fill('move ')"), ("status", "run('status')"),
         ("help", "run('help')"))
TRADE_EVENTS = {"entry", "exit", "scale_up", "manual_buy", "manual_sell", "KILL_SWITCH"}


# --- formatting --------------------------------------------------------------------------------
def num(n, d=1):
    return "—" if n is None or not math.isfinite(n) else f"{n:,.{d}f}"


def usd(n):
    return "—" if n is None or not math.isfinite(n) else ("−$" if n < 0 else "$") + f"{abs(n):,.2f}"


def pct(n, d=2):
    return "—" if n is None or not math.isfinite(n) else ("+" if n >= 0 else "−") + f"{abs(n) * 100:.{d}f}%"


def px(n):
    """Prices span $1 (xrp) to $80,000 (btc): decimals by magnitude."""
    return num(n, 1 if abs(n) >= 1000 else 2 if abs(n) >= 10 else 4) if n is not None and math.isfinite(n) else "—"


def pxr(n):
    return num(n, 0 if abs(n) >= 1000 else 2 if abs(n) >= 10 else 4) if n is not None and math.isfinite(n) else "—"


COIN_NAMES = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "xrp"}
COIN_WS = {"BTC": ("BTC/USD", "BTC-USD"), "ETH": ("ETH/USD", "ETH-USD"), "SOL": ("SOL/USD", "SOL-USD"),
           "XRP": ("XRP/USD", "XRP-USD")}


SYM_COIN = {v: c for c, pair in COIN_WS.items() for v in pair}


def link(style, action):
    """A style whose text runs a Desk action when clicked (see Desk.on_click). Not Textual's @click, which
    repaints every link in one link colour and would hide the selected chip's text."""
    return Style.parse(style) + Style.from_meta({"go": action})


def hhmm(t):
    return datetime.fromtimestamp(t).strftime("%H:%M")


# --- big LED digits -----------------------------------------------------------------------------
DIGITS = {
    "0": ["█▀█", "█ █", "▀▀▀"], "1": ["▀█ ", " █ ", "▀▀▀"], "2": ["▀▀█", "█▀▀", "▀▀▀"],
    "3": ["▀▀█", " ▀█", "▀▀▀"], "4": ["█ █", "▀▀█", "  ▀"], "5": ["█▀▀", "▀▀█", "▀▀▀"],
    "6": ["█▀▀", "█▀█", "▀▀▀"], "7": ["▀▀█", "  █", "  ▀"], "8": ["█▀█", "█▀█", "▀▀▀"],
    "9": ["█▀█", "▀▀█", "▀▀▀"], ",": ["  ", "  ", "▄▀"], ".": [" ", " ", "▄"], "—": ["   ", "▀▀▀", "   "],
}


def big_number(s, style):
    rows = ["", "", ""]
    for ch in s:
        glyph = DIGITS.get(ch, ["   "] * 3)
        for i in range(3):
            rows[i] += glyph[i] + " "
    return Text("\n".join(r.rstrip() for r in rows), style=style)


# --- chart rendering (pure: data -> rich Text) ----------------------------------------------------
BRAILLE = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]]


class Grid:
    """A character canvas with per-cell style, plus a 2x4 braille sub-grid for line plots."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.ch = [[" "] * w for _ in range(h)]
        self.st = [[""] * w for _ in range(h)]
        self.br = {}

    def put(self, r, c, ch, style, force=True):
        if 0 <= r < self.h and 0 <= c < self.w and (force or self.ch[r][c] == " "):
            self.ch[r][c], self.st[r][c] = ch, style

    def text(self, r, c, s, style, force=True):
        for i, ch in enumerate(s):
            self.put(r, c + i, ch, style, force)

    def dot(self, px, py, style):
        c, r = px // 2, py // 4
        if 0 <= r < self.h and 0 <= c < self.w:
            bits = self.br.get((r, c), 0) | BRAILLE[py % 4][px % 2]
            self.br[(r, c)] = bits
            self.ch[r][c], self.st[r][c] = chr(0x2800 + bits), style

    def line(self, x0, y0, x1, y1, style):
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while True:
            self.dot(x0, y0, style)
            if x0 == x1 and y0 == y1:
                return
            e2 = 2 * err
            if e2 >= dy:
                err += dy; x0 += sx
            if e2 <= dx:
                err += dx; y0 += sy

    def render(self):
        t = Text(no_wrap=True, overflow="crop")
        for r in range(self.h):
            run, style = "", None
            for c in range(self.w):
                s = self.st[r][c]
                if s != style and run:
                    t.append(run, style=style or None); run = ""
                style = s
                run += self.ch[r][c]
            if run:
                t.append(run, style=style or None)
            if r < self.h - 1:
                t.append("\n")
        return t


def nice_step(rng, target):
    raw = rng / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    n = raw / mag
    return (1 if n < 1.5 else 2 if n < 3 else 5 if n < 7 else 10) * mag


def bucket_bars(bars, cols):
    """Merge candles so each one gets at least one column."""
    if len(bars) <= cols:
        return bars
    k = math.ceil(len(bars) / cols)
    out = []
    for i in range(0, len(bars), k):
        grp = bars[i:i + k]
        out.append([grp[0][0], grp[0][1], max(b[2] for b in grp), min(b[3] for b in grp), grp[-1][4],
                    sum(b[5] for b in grp)])
    return out


def render_chart(width, height, view, data, ticks, last, status, fills):
    g = Grid(max(width, 20), max(height, 6))
    W, H = g.w, g.h
    axis = 9
    future = max(8, int((W - axis) * 0.18))
    plot_w = W - axis - future
    plot_h = H - 1  # bottom row holds the time labels
    now = time.time()
    if not data or plot_w < 10 or plot_h < 4:
        g.text(0, 1, "tuning in…", DIM)
        return g.render()
    span = VIEWS[view][3]
    pol = status.get("policy") if status else None
    st = status.get("state", {}) if status else {}
    sig = st.get("signal")

    bars = pts = None
    if data["kind"] == "candles":
        secs = data["secs"]
        bars = bucket_bars(data["bars"], plot_w)
        bsecs = secs * max(1, math.ceil(len(data["bars"]) / plot_w))
        t0 = bars[0][0] if bars else now - span
        t1 = max(now, bars[-1][0] + bsecs) if bars else now
    else:
        t0, t1 = now - span, now
        hist = data["points"]
        last_hist = hist[-1][0] if hist else 0
        pts = [p for p in hist + [p for p in ticks if p[0] > last_hist] if p[0] >= t0]

    p0 = last or (bars[-1][4] if bars else pts[-1][1] if pts else None)
    lo, hi = math.inf, -math.inf
    for b in bars or []:
        lo, hi = min(lo, b[3]), max(hi, b[2])
    for p in pts or []:
        lo, hi = min(lo, p[1]), max(hi, p[1])
    if p0:
        lo, hi = min(lo, p0), max(hi, p0)
    if not math.isfinite(lo):
        return g.render()
    span0 = max(hi - lo, (p0 or hi) * 0.002)
    cone = None
    if sig and p0 and pol:
        drift = 0.0 if pol["model"] == "momentum" else sig["pred"]
        horizon = sig.get("horizon") or pol["horizon"]
        secs_per_col = (t1 - t0) / plot_w
        f_max = min(1.0, future * secs_per_col / (horizon * 3600))
        cone = (drift, sig["sigma_h"], horizon, f_max, secs_per_col)
        fv = min(1.0, span * 0.25 / (horizon * 3600))
        hi = min(max(hi, p0 * math.exp(drift * fv + sig["sigma_h"] * math.sqrt(fv))), hi + span0 * 1.2)
        lo = max(min(lo, p0 * math.exp(drift * fv - sig["sigma_h"] * math.sqrt(fv))), lo - span0 * 1.2)
    pos = st.get("position")
    if pos:
        for v in (pos["entry_px"], pos.get("stop_px", 0)):
            if lo - span0 < v < hi + span0:
                lo, hi = min(lo, v), max(hi, v)
    if hi - lo < span0:
        m = (hi + lo) / 2
        lo, hi = m - span0 / 2, m + span0 / 2
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad

    def Y(p, sub=1):  # row (or sub-row) for a price, 0 at the top
        return (1 - (p - lo) / (hi - lo)) * (plot_h * sub - 1)

    def X(t):
        return (t - t0) / (t1 - t0) * plot_w

    # price grid + axis
    step = nice_step(hi - lo, max(3, plot_h // 4))
    v = math.ceil(lo / step) * step
    while v <= hi:
        r = round(Y(v))
        for c in range(0, plot_w, 2):
            g.put(r, c, "·", "#1d3a47", force=False)
        g.text(r, W - axis + 1, num(v, 0 if step >= 1 else 2 if step >= 0.01 else 4).rjust(axis - 1), DIM)
        v += step
    # time labels, aligned to local time
    tstep = {"10m": 120, "1h": 600, "1d": 4 * 3600, "1w": 86400}[view]
    while X(t0 + tstep) - X(t0) < 8:
        tstep *= 2
    tz = -time.localtime().tm_gmtoff
    t = math.ceil((t0 - tz) / tstep) * tstep + tz
    while t <= now:
        c = int(X(t))
        label = datetime.fromtimestamp(t).strftime("%a %d") if view == "1w" else hhmm(t)
        if c + len(label) < plot_w:
            g.text(H - 1, c, label, DIM)
        t += tstep

    # future strip + projection cone
    for r in range(plot_h):
        for c in range(plot_w, plot_w + future):
            g.put(r, c, "╱" if (r + c) % 6 == 0 else " ", "#12313b")
        g.put(r, plot_w, "┊", "#2f6f75")
    if cone:
        drift, sigma, horizon, f_max, secs_per_col = cone
        for c in range(1, future):
            f_raw = c * secs_per_col / (horizon * 3600)
            if f_raw > f_max + 1e-9:
                break
            s = sigma * math.sqrt(f_raw)
            r2a, r2b = Y(p0 * math.exp(drift * f_raw + 2 * s)), Y(p0 * math.exp(drift * f_raw - 2 * s))
            r1a, r1b = Y(p0 * math.exp(drift * f_raw + s)), Y(p0 * math.exp(drift * f_raw - s))
            for r in range(max(0, math.floor(r2a)), min(plot_h, math.ceil(r2b) + 1)):
                inner = r1a - 0.5 <= r <= r1b + 0.5
                g.put(r, plot_w + c, "▒" if inner else "░", "#3f9da2" if inner else "#245a5e")
            g.put(round(Y(p0 * math.exp(drift * f_raw))), plot_w + c, "━", f"bold {CYAN}")
        ahead = f_max * horizon
        if pol["model"] == "momentum":
            label = ("trend up, long" if sig["pred"] > 0 else "trend down, cash") if future >= 18 else \
                    ("long" if sig["pred"] > 0 else "cash")
        else:
            label = f"{pct(drift)} in {horizon}h"
        g.text(0, plot_w + 2, label[:future - 2], f"bold {CYAN}")
        shown = f"{round(ahead * 60)}m" if ahead < 1 else f"{ahead:.0f}h"
        g.text(1, plot_w + 2, f"next {shown}"[:future - 2], CYAN)

    # history
    if bars:
        for b in bars:
            c = min(plot_w - 1, int(X(b[0] + bsecs / 2)))
            up = b[4] >= b[1]
            col = AMBER if up else EMBER
            for r in range(round(Y(b[2])), round(Y(b[3])) + 1):
                g.put(r, c, "│", col)
            # body drawn with half-cell precision
            bt, bb = sorted((Y(b[1], 2), Y(b[4], 2)))
            bt, bb = math.floor(bt), max(math.ceil(bb), math.floor(bt) + 1)
            for hr in range(bt, bb):
                r = hr // 2
                cur = g.ch[r][c] if 0 <= r < g.h else " "
                half = "▀" if hr % 2 == 0 else "▄"
                g.put(r, c, "█" if cur in "▀▄█" and cur != half else half, f"bold {col}" if up else col)
    elif pts and len(pts) > 1:
        hot_from = now - 60  # the newest minute burns hot, like phosphor persistence
        prev = None
        for p in pts:
            x, y = min(int(X(p[0]) * 2), plot_w * 2 - 1), int(Y(p[1], 4))
            style = f"bold {HOT}" if p[0] >= hot_from else AMBER
            if prev:
                g.line(prev[0], prev[1], x, y, style)
            prev = (x, y)

    # position lines + fills
    if pos:
        r = round(Y(pos["entry_px"]))
        for c in range(0, plot_w, 2):
            g.put(r, c, "╌", CYAN, force=False)
        g.text(r, 1, f"\u00a0entry\u00a0{pxr(pos['entry_px'])}\u00a0", f"{GLASS} on {CYAN}")  # nbsp: rails skip it
        if pos.get("stop_px", 0) > max(lo, pos["entry_px"] * 0.01):
            r = round(Y(pos["stop_px"]))
            for c in range(0, plot_w, 2):
                g.put(r, c, "╌", EMBER, force=False)
            g.text(r, 1, f"\u00a0stop\u00a0{pxr(pos['stop_px'])}\u00a0", f"{GLASS} on {EMBER}")
    for f in fills:
        if t0 <= f["t"] <= now:
            buy = f["side"] == "buy"
            g.put(round(Y(f["price"])) + (1 if buy else -1), min(plot_w - 1, int(X(f["t"]))),
                  "▲" if buy else "▼", f"bold {CYAN if buy else EMBER}")

    # last price rail + axis tag
    if p0:
        r = round(Y(p0))
        for c in range(plot_w):
            g.put(r, c, "┄", "#8c6a3a", force=False)
        g.put(r, plot_w, "◆", f"bold {HOT}")
        g.text(r, W - axis, f" {pxr(p0)} ".rjust(axis)[-axis:], f"bold {GLASS} on {AMBER}")
    return g.render()


# --- widgets ---------------------------------------------------------------------------------------
class Chart(Widget):
    def __init__(self, desk, **kw):
        super().__init__(**kw)
        self.desk = desk

    def render(self):
        d = self.desk
        fills = [f for f in d.fills if f["coin"] == d.coin]
        return render_chart(self.size.width, self.size.height, d.view, d.data, d.ticks, d.last, d.status, fills)


class Crawl(Static):
    """Model notes on a ticker tape."""

    def on_mount(self):
        self.i = 0
        self.notes = Text("connecting to the model…", style=CYAN)
        self.set_interval(0.12, self.advance)

    def set_notes(self, notes):
        t = Text()
        colors = {"trade": AMBER, "account": AMBER, "alert": f"bold {EMBER}"}
        for n in notes:
            t.append("  ◆ ", style=DIM)
            t.append(n["text"], style=colors.get(n["kind"], CYAN))
        self.notes = t

    def advance(self):
        w = max(self.size.width - 1, 10)
        loop = self.notes + Text("      ") + self.notes
        n = len(self.notes.plain) + 6
        i = self.i % n
        self.update(loop[i:i + w])
        self.i += 1


class Prompt(Input):
    """Command line; up/down walk the history."""

    def on_key(self, event):
        desk = self.app
        if event.key in ("up", "down") and desk.history:
            desk.hist_i = len(desk.history) if desk.hist_i < 0 else desk.hist_i
            desk.hist_i = max(0, min(len(desk.history), desk.hist_i + (-1 if event.key == "up" else 1)))
            self.value = desk.history[desk.hist_i] if desk.hist_i < len(desk.history) else ""
            self.cursor_position = len(self.value)
        else:
            return
        event.prevent_default()
        event.stop()


# --- the app ---------------------------------------------------------------------------------------
CSS = f"""
Screen {{ background: {GLASS}; color: {AMBER}; }}
#crawl-row {{ height: 1; background: #0a1f29; }}
#crawl-tag {{ width: 7; background: {CYAN}; color: {GLASS}; text-style: bold; content-align: center middle; }}
#crawl {{ width: 1fr; }}
#quote {{ height: 4; padding: 0 1; }}
#pair {{ width: 16; }}
#price {{ width: auto; min-width: 30; padding: 0 1 0 0; }}
#chg {{ width: 10; padding: 1 0 0 1; }}
#stats {{ width: 1fr; min-width: 42; padding: 1 1 0 2; }}
#keys {{ width: auto; padding: 0 1; }}
#coins {{ height: 1; padding: 0 1; }}
#actions {{ height: 1; padding: 0 1; }}
#lamps {{ width: auto; }}
#desk {{ height: 1fr; }}
#left {{ width: 1fr; }}
#side {{ width: 38; }}
.win {{ border: round {DIM_AMBER}; border-title-color: {AMBER}; border-title-style: bold; padding: 0 1; background: {GLASS}; }}
.model-win {{ border: round #2f6f75; border-title-color: {CYAN}; }}
#chart-win {{ height: 1fr; padding: 0; }}
#chart {{ height: 1fr; }}
#console-win {{ height: 10; }}
#console {{ height: 1fr; background: {GLASS}; scrollbar-size-vertical: 1; scrollbar-color: {DIM_AMBER};
            scrollbar-background: {GLASS}; scrollbar-background-hover: {GLASS}; scrollbar-background-active: {GLASS}; }}
#account {{ height: auto; }}
#model {{ height: auto; }}
#fills-win {{ height: 1fr; }}
#prompt {{ height: 3; border: round {AMBER}; background: #030b10; color: {HOT}; padding: 0 1; }}
#prompt:focus {{ border: round {HOT}; }}
#prompt.pending {{ border: round {EMBER}; }}
Screen.narrow #side {{ display: none; }}
Screen.narrow #stats {{ display: none; }}
Screen.narrow #chg {{ display: none; }}
Screen.narrow #pair {{ width: 13; }}
"""


class Desk(App):
    CSS = CSS
    TITLE = "Phosphor Desk"
    BINDINGS = [("ctrl+q", "quit", "quit"), ("ctrl+l", "clear", "clear console")]

    def __init__(self):
        super().__init__()
        self.view = "1h"
        self.data = None
        self.ticks = []
        self.last = self.prev = None
        self.status = None
        self.since = 0
        self.fills = []
        self.token = None
        self.boot_id = None
        self.ws_at = 0.0
        self.history, self.hist_i = [], -1
        self.http = requests.Session()
        self.feeding = False
        self.coin = "BTC"
        self.ws = None
        self.coins_cache = ["BTC"]
        self.pool_cache = None
        self.live = {}  # coin -> {last, chg, at} from the all-coins ticker
        self.coin_states = {}
        self.flash_up, self.flash_until = True, 0.0

    def compose(self) -> ComposeResult:
        with Horizontal(id="crawl-row"):
            yield Static("model", id="crawl-tag")
            yield Crawl(id="crawl")
        with Horizontal(id="quote"):
            yield Static(id="pair")
            yield Static(id="price")
            yield Static(id="chg")
            yield Static(id="stats")
            yield Static(id="keys")
            yield Static(id="lamps")
        yield Static(id="coins")
        with Horizontal(id="desk"):
            with Vertical(id="left"):
                with Vertical(id="chart-win", classes="win"):
                    yield Chart(self, id="chart")
                with Vertical(id="console-win", classes="win"):
                    yield RichLog(id="console", wrap=True, markup=False, max_lines=500)
            with Vertical(id="side"):
                yield Static(id="account", classes="win")
                yield Static(id="model", classes="win model-win")
                with Vertical(id="fills-win", classes="win"):
                    yield Static(id="fills")
        yield Static(id="actions")
        yield Prompt(placeholder="buy 25 eth · sell all · move 20 btc sol · help", id="prompt")

    def on_mount(self):
        for wid, title in (("#console-win", "console"), ("#account", "portfolio"), ("#model", "model"),
                           ("#fills-win", "fills")):
            self.query_one(wid).border_title = title
        self.paint_keys()
        self.paint_actions()
        self.query_one("#prompt").focus()
        self.log_line("type help for commands, or click a chart range, a coin or a command. ctrl+q quits.", CYAN)
        self.run_worker(self.boot())
        self.set_interval(2.0, self.poll_status)
        self.set_interval(20.0, self.poll_notes)
        self.set_interval(1.0, self.repaint)
        self.on_resize(None)

    def on_resize(self, event):
        self.screen.set_class(self.size.width < 110, "narrow")

    # --- data -------------------------------------------------------------------------------
    async def boot(self):
        try:
            html = await asyncio.to_thread(lambda: self.http.get(BASE + "/", timeout=5).text)
            self.token = re.search(r'name="btcbot-token" content="([^"]+)"', html).group(1)
        except Exception as e:
            self.log_line(f"can't reach the bot at {BASE} ({e}). start it with ./run_live.sh", EMBER)
            return
        await self.poll_status()
        await self.load_view(self.view)
        await self.poll_notes()
        if self.status and not self.feeding:
            self.feeding = True
            self.run_worker(self.feed(self.status["exchange"]))

    async def get(self, path):
        return await asyncio.to_thread(lambda: self.http.get(BASE + path, timeout=15).json())

    async def load_view(self, view):
        self.view = view
        self.paint_keys()
        title, sub, _, _ = VIEWS[view]
        self.query_one("#chart-win").border_title = f"{title} [{DIM}]{sub}[/]"
        try:
            coin = self.coin
            d = await self.get(f"/api/candles?view={view}&coin={coin}")
            if "error" in d:
                raise RuntimeError(d["error"])
            if view == self.view and coin == self.coin:
                self.data = d
                self.query_one("#chart").refresh()
        except Exception as e:
            self.log_line(f"chart data unavailable: {e}", EMBER)

    async def poll_status(self):
        if not self.token:
            return
        try:
            st = await self.get(f"/api/status?since={self.since}&coin={self.coin}")
        except Exception:
            self.paint_lamps(offline=True)
            return
        if self.boot_id and st.get("boot_id") != self.boot_id:
            self.log_line("the bot restarted; reconnecting", EMBER)
            self.since, self.token, self.boot_id = 0, None, None
            self.run_worker(self.boot())
            return
        self.boot_id = st.get("boot_id")
        if st.get("coin", self.coin) != self.coin:  # answered for the coin we just left
            for e in st.get("events", []):
                self.since = max(self.since, e["seq"])
                self.handle_bot_event(e)
            return
        self.status = st
        self.coin_states = st.get("coins") or {}
        self.coins_cache = list(self.coin_states) or ["BTC"]
        tk = st.get("ticker") or {}
        if tk and time.time() - self.ws_at > 8:  # REST fallback while the WebSocket is quiet
            self.on_price(tk["last"], tk["t"])
        for e in st.get("events", []):
            self.since = max(self.since, e["seq"])
            self.handle_bot_event(e)
        self.query_one("#prompt").set_class(bool(st.get("pending")), "pending")
        self.paint_panels()
        self.paint_coins()

    async def poll_notes(self):
        if self.token:
            try:
                self.query_one(Crawl).set_notes(await self.get(f"/api/notes?coin={self.coin}"))
            except Exception:
                pass

    async def feed(self, exchange):
        """Live trades over the exchange WebSocket, for whichever coin is on screen."""
        import ssl

        import certifi
        import websockets
        url = "wss://ws.kraken.com/v2" if exchange == "kraken" else "wss://advanced-trade-ws.coinbase.com"
        # python.org builds of Python ship without system CA roots; use certifi's bundle like requests does
        tls = ssl.create_default_context(cafile=certifi.where())
        while True:
            coin = self.coin
            try:
                async with websockets.connect(url, ping_interval=20, ssl=tls) as ws:
                    self.ws = ws
                    sym, product = COIN_WS[coin]
                    every = self.coin_list()
                    if exchange == "kraken":
                        await ws.send(json.dumps({"method": "subscribe",
                                                  "params": {"channel": "trade", "symbol": [sym]}}))
                        await ws.send(json.dumps({"method": "subscribe", "params": {
                            "channel": "ticker", "symbol": [COIN_WS[c][0] for c in every]}}))
                    else:
                        await ws.send(json.dumps({"type": "subscribe", "channel": "ticker",
                                                  "product_ids": [COIN_WS[c][1] for c in every]}))
                    async for msg in ws:
                        if coin != self.coin:
                            break  # coin switched: reconnect for the new one
                        d = json.loads(msg)
                        if exchange == "kraken" and d.get("channel") == "trade" and d.get("type") == "update":
                            for tr in d["data"]:
                                ts = datetime.fromisoformat(tr["timestamp"].replace("Z", "+00:00")).timestamp()
                                self.on_price(float(tr["price"]), ts, ws=True)
                        elif exchange == "kraken" and d.get("channel") == "ticker":
                            for k in d.get("data", []):
                                c = SYM_COIN.get(k.get("symbol"))
                                if c:
                                    self.live[c] = {"last": float(k["last"]), "chg": float(k["change_pct"]) / 100,
                                                    "at": time.time()}
                        elif exchange != "kraken" and d.get("channel") == "ticker":
                            for ev in d.get("events", []):
                                for k in ev.get("tickers", []):
                                    c = SYM_COIN.get(k.get("product_id"))
                                    if not c:
                                        continue
                                    self.live[c] = {"last": float(k["price"]),
                                                    "chg": float(k["price_percent_chg_24_h"]) / 100, "at": time.time()}
                                    if c == coin:
                                        self.on_price(float(k["price"]), time.time(), ws=True)
            except Exception:
                await asyncio.sleep(4)
            if coin != self.coin:
                continue

    def coin_list(self):
        return self.coins_cache

    def switch_coin(self, coin):
        if coin == self.coin or coin not in self.coin_list():
            return
        self.coin = coin
        self.status = None  # the old coin's status must never be painted against the new coin's price
        self.ticks, self.last, self.prev, self.data, self.ws_at = [], None, None, None, 0.0
        if self.ws is not None:
            self.run_worker(self.ws.close())
        self.log_line(f"showing {COIN_NAMES[coin]}", HOT)
        self.paint_coins()
        self.run_worker(self.load_view(self.view))
        self.run_worker(self.poll_status())
        self.run_worker(self.poll_notes())

    def paint_coins(self):
        coins = self.coin_states
        t = Text()
        for c in self.coin_list():
            info = coins.get(c, {})
            L = info.get("ledger") or {}
            funded = (L.get("usd") or 0) + (L.get("qty") or 0) > 0
            state = ("halted" if info.get("halted") else "review" if info.get("needs_review") else
                     "unfunded" if not funded else ("held" if (info.get("position") or {}).get("manual") else "long")
                     if info.get("position") else
                     "manual" if info.get("auto") is False else "cash")
            on, go = c == self.coin, f"coin('{c}')"
            t.append(f" {c.lower()} ", style=link(f"bold {GLASS} on {AMBER}" if on else f"{AMBER} on #12303f", go))
            t.append(f" {state:<9}", style=link(CYAN if state == "long" else EMBER if state in ("halted", "review")
                                                else DIM, go))
        pool = (self.status or {}).get("pool") or self.pool_cache
        if pool:
            self.pool_cache = pool
            L, n = pool.get("ledger") or {}, len(pool.get("bets") or {})
            state = ("halted" if pool.get("halted") else "review" if pool.get("needs_review") else
                     "unfunded" if not (L.get("usd") or n) else f"${L.get('usd', 0):,.2f} cash, {n} open")
            t.append("  bets ", style=link(f"bold {CYAN}", "run('bets')"))
            t.append(state, style=link(EMBER if state in ("halted", "review") else CYAN, "run('bets')"))
        self.query_one("#coins").update(t)

    def paint_actions(self):
        t = Text()
        for label, action in CHIPS:
            t.append(f" {label} ", style=link(f"{AMBER} on #12303f", action))
            t.append(" ")
        t.append("click to fill the prompt, then edit and press enter", style=DIM)
        self.query_one("#actions").update(t)

    def on_price(self, price, t, ws=False):
        if not (price and math.isfinite(price) and math.isfinite(t)):
            return
        if ws:
            self.ws_at = time.time()
        if self.last is not None and price != self.last:
            self.flash_up, self.flash_until = price > self.last, time.time() + 0.4
        self.prev, self.last = self.last, price
        self.ticks.append([t, price])
        cutoff = t - 1800
        while self.ticks and self.ticks[0][0] < cutoff:
            self.ticks.pop(0)
        if self.data and self.data["kind"] == "candles" and self.data["bars"]:
            bars, secs = self.data["bars"], self.data["secs"]
            lb = bars[-1]
            if t >= lb[0] + secs:
                bars.append([int(t // secs * secs), price, price, price, price, 0])
                del bars[:-400]
            elif t >= lb[0]:
                lb[4] = price; lb[2] = max(lb[2], price); lb[3] = min(lb[3], price)

    # --- painting ---------------------------------------------------------------------------
    def repaint(self):
        self.paint_price()
        self.paint_lamps()
        self.query_one("#chart").refresh()
        if self.status:
            self.paint_panels()
        if self.token and int(time.time()) % VIEWS[self.view][2] == 0:
            self.run_worker(self.load_view(self.view))

    def paint_price(self):
        style = f"bold {HOT}"
        if time.time() < self.flash_until:
            style = f"bold {'#fff3dc' if self.flash_up else EMBER}"
        self.query_one("#price").update(big_number(px(self.last) if self.last else "—", style))
        st = self.status or {}
        tk = st.get("ticker") or {}
        pair = Text(f"{self.coin.lower()}/usd\n", style=f"bold {AMBER}")
        ex = {"kraken": "kraken pro", "coinbase": "coinbase adv."}.get(st.get("exchange"), "connecting")
        pair.append(f"{ex}\n{st.get('mode', '')} money", style=DIM)
        self.query_one("#pair").update(pair)
        if tk.get("open24") and self.last:
            chg = self.last / tk["open24"] - 1
            self.query_one("#chg").update(Text(f"{pct(chg)}\n24h", style=f"bold {EMBER if chg < 0 else AMBER}"))
        s = Text()
        for i, (label, key, d) in enumerate((("bid", "bid", 1), ("ask", "ask", 1), ("24h high", "high24", 0),
                                              ("24h low", "low24", 0))):
            s.append(f"{label:<9}", style=DIM)
            s.append(f"{num(tk.get(key), d) if tk.get(key) else '—':<11}", style=AMBER)
            if i % 2:
                s.append("\n")
        self.query_one("#stats").update(s)

    def paint_keys(self):
        t = Text("\n")
        for v in VIEWS:
            on = v == self.view
            t.append(f" {v} ", style=link(f"bold {GLASS} on {AMBER}" if on else f"{AMBER} on #12303f", f"view('{v}')"))
            t.append(" ")
        self.query_one("#keys").update(t)

    def paint_lamps(self, offline=False):
        st = self.status or {}
        s = st.get("state", {})
        t = Text("\n")
        feed_on = time.time() - self.ws_at < 15
        t.append(" feed ", style=f"bold {GLASS} on {AMBER}" if feed_on else f"{DIM} on #0c2230")
        t.append(" ")
        bot = "offline" if offline else st.get("busy") or ("halted" if s.get("halted") else
                                                            "bot off" if s.get("auto") is False else "bot on")
        if offline or s.get("halted"):
            bot_style = f"bold {GLASS} on {EMBER}"
        elif s.get("auto", True):
            bot_style = f"bold {GLASS} on {AMBER}"
        else:
            bot_style = f"{DIM} on #0c2230"
        flip = "bot on" if s.get("auto") is False else "bot off"
        t.append(f" {bot} ", style=link(bot_style, f"fill('{flip}')") if st else bot_style)
        t.append(" ")
        live = st.get("mode") == "live"
        t.append(" live money " if live else " paper ", style=f"bold {GLASS} on {EMBER}" if live else f"{DIM} on #0c2230")
        self.query_one("#lamps").update(t)

    def paint_panels(self):
        st = self.status
        s, pol = st["state"], st["policy"]
        L = s.get("ledger", {})
        last = self.last
        a = Text()
        pv = st.get("portfolio")
        if pv:  # headline: the whole portfolio (all coins + bet pool), P&L vs money actually put in
            pnl, basis = pv["pnl"], pv["basis"]
            a.append_text(big_number(f"{pv['equity']:,.2f}", f"bold {HOT}"))
            a.append(f"\n{'+' if pnl >= 0 else '−'}${abs(pnl):,.2f} ({pct(pnl / basis if basis else 0, 1)}) all coins\n",
                     style=f"bold {EMBER if pnl < 0 else AMBER}")
            parts = "  ".join(f"{'bets' if k == 'BETS' else k.lower()} {usd(v)}" for k, v in pv["parts"].items() if v >= 0.01)
            a.append(parts + "\n", style=DIM)
            a.append(f"{'coin':<4}{'price':>10}{'24h':>8}{'held':>9}\n", style=DIM)  # rows end in " ↑•": 34 cells, the panel width
            for c, row in (pv.get("coins") or {}).items():  # live market row per coin
                lv = self.live.get(c) or {}
                fresh = lv and time.time() - lv["at"] < 15
                row_px = lv["last"] if fresh else row.get("last")  # not `last`: that is the selected coin's price
                chg = lv["chg"] if fresh else row.get("chg24")
                held = row.get("qty") or 0
                value = held * (row_px or 0)
                arrow = "↑" if row.get("trend") == "up" else "↓" if row.get("trend") == "down" else " "
                sel, go = c == self.coin, f"coin('{c}')"  # the whole row switches to that coin
                a.append(f"{c.lower():<4}", style=link(f"bold {HOT}" if sel else AMBER, go))
                a.append(f"{px(row_px):>10}", style=link(f"bold {HOT}" if sel else AMBER, go))
                a.append(f"{pct(chg, 2):>8}", style=link(EMBER if (chg or 0) < 0 else AMBER, go))
                a.append(f"{(usd(value) if held else '—'):>9}", style=link(AMBER if held else DIM, go))
                a.append(f" {arrow}", style=link(CYAN if arrow == "↑" else EMBER, go))
                if row.get("bet"):
                    a.append("•", style=link(f"bold {CYAN}", go))
                a.append("\n")
        if last:
            a.append(f"{self.coin.lower() + ' sleeve':<13}", style=DIM)
            a.append(f"{usd(L['usd'] + L['qty'] * last):>21}\n", style=AMBER)
        pos = s.get("position")
        rows = [("sleeve cash", usd(L.get("usd"))),
                (COIN_NAMES.get(self.coin, self.coin.lower()), f"{L.get('qty', 0):.6g} ≈ {usd(L.get('qty', 0) * (last or 0))}"),
                ("position", (f"{'manual' if pos.get('manual') else 'long'} @ {pxr(pos['entry_px'])}")
                 if pos else "flat"),
                ("stop", pxr(pos["stop_px"]) if pos and pos.get("stop_px", 0) > pos["entry_px"] * 0.01 else "—"),
                ("paid in fees", usd(L.get("fees")))]
        for k, v in rows:
            a.append(f"{k:<13}", style=DIM)
            a.append(f"{v:>21}\n", style=AMBER)
        if s.get("halted") or s.get("needs_review"):
            a.append(f"\n{s.get('needs_review') or 'kill switch engaged'}. type resume once checked.\n",
                     style=f"bold {EMBER}")
        self.query_one("#account").update(a)

        m = Text()
        sig = s.get("signal")
        name = f"{pol['sma_days']}-day trend rule" if pol["model"] == "momentum" else f"{pol['model']}, {pol['horizon']}h"
        m.append(f"{name}\n", style=DIM)
        if sig:
            if pol["model"] == "momentum":
                val = sig["daily_close"] / sig["sma"] - 1 if sig.get("sma") else 0
                m.append(f"Trend {'up' if sig['pred'] > 0 else 'down'}: yesterday closed {pct(val, 1)} "
                         f"{'above' if val >= 0 else 'below'} the average. I want to be "
                         f"{'long' if sig['pred'] > 0 else 'in cash'}.\n", style=CYAN)
                entry, R = 0.0, max(abs(val) * 1.4, 0.05)
            else:
                val, entry = sig["pred"], pol["entry_thr"] or 0
                m.append(f"Expecting {pct(val)} over {sig['horizon']}h. Entry needs {pct(entry)}.\n", style=CYAN)
                R = max(abs(entry) * 1.3, sig["sigma_h"] * 1.5, abs(val) * 1.2)
            m.append_text(self.gauge(val, entry, R, 32))
        else:
            m.append("waiting for the first hourly signal\n", style=CYAN)
        nxt = (time.time() // 3600 + 1) * 3600 + 20 - time.time()
        gate = ("ml failed; rule" if pol["model"] == "momentum" else "passed" if pol.get("gate_pass")
                else "failed; watching" if st["mode"] == "live" else "failed; paper only")
        for k, v in (("next decision", f"in {int(nxt // 60)}m {int(nxt % 60):02d}s"),
                     ("round trip cost", f"{pol['round_trip'] * 100:.2f}%"), ("go-live gate", gate)):
            m.append(f"{k:<16}", style=DIM)
            m.append(f"{v:>18}\n", style=AMBER)
        self.query_one("#model").update(m)

        f = Text()
        for x in reversed(self.fills[-12:]):
            f.append(f"{x['side']:<5}", style=f"bold {EMBER}" if x["side"] == "sell" else f"bold {CYAN}")
            f.append(f"{x['coin'].lower():<4}{x['size']:<10.4g}@ {pxr(x['price']):<8}", style=AMBER)
            f.append(f" {hhmm(x['t'])}\n", style=DIM)
        self.query_one("#fills").update(f if self.fills else Text("no trades yet", style=DIM))

    @staticmethod
    def gauge(val, entry, R, w):
        def col(x):
            return max(0, min(w - 1, round((x + R) / (2 * R) * (w - 1))))
        cells, styles = ["─"] * w, [DIM] * w
        for i in range(0, w, max(1, w // 8)):
            cells[i] = "┼"
        cells[col(0)] = "┃"
        cells[col(entry)], styles[col(entry)] = "▲", f"bold {AMBER}"
        cells[col(val)], styles[col(val)] = "◆", f"bold {CYAN}"
        t = Text()
        for ch, stl in zip(cells, styles):
            t.append(ch, style=stl)
        t.append(f"\n{pct(-R, 1):<{w // 2}}{pct(R, 1):>{w - w // 2}}\n", style=DIM)
        return t

    # --- console + events -------------------------------------------------------------------
    def log_line(self, text, style=AMBER, t=None):
        line = Text(hhmm(t or time.time()) + "  ", style=DIM)
        line.append(text, style=style)
        self.query_one("#console", RichLog).write(line)

    def handle_bot_event(self, e):
        t = datetime.fromisoformat(e["ts"]).timestamp()
        ev, f = e.get("event"), e.get("fill")
        pol = (self.status or {}).get("policy") or {}
        if ev in ("tick", "signal"):
            close_t = datetime.fromisoformat(e["bar"]).timestamp() + 3600
            what = (f"trend {'up, want long' if e['pred'] > 0 else 'down, want cash'}" if pol.get("model") == "momentum"
                    else f"projection {pct(e['pred'])} over {e.get('horizon')}h")
            self.log_line(f"{e.get('coin', 'BTC').lower()}: hour to {hhmm(close_t)} closed {pxr(e['close'])}. {what}."
                          + (f" {e['note']}" if e.get("note") else ""), CYAN, t)
        elif ev == "order":
            stage = e.get("stage")
            if stage == "placing":
                at = px(e["price"]) if e.get("price") else "market"
                self.log_line(f"{e.get('coin', '').lower()} order {e.get('side')} {e.get('size', '')} @ {at}", DIM_AMBER, t)
            elif stage == "done":
                why = f" ({e['reason']})" if e.get("reason") else ""
                self.log_line(f"order {e.get('order_id')} {str(e.get('status')).lower()}{why}", DIM_AMBER, t)
            elif stage == "UNSETTLED":
                self.log_line(f"order {e.get('order_id') or e.get('userref')} unconfirmed; bot halted for review", EMBER, t)
        elif ev == "startup":
            q = e["ledger"].get("qty", e["ledger"].get("btc", 0))
            self.log_line(f"{e.get('coin', 'BTC').lower()} sleeve online: {usd(e['ledger']['usd'])} cash, {float(q):.6g} {e.get('coin', 'BTC')}.", CYAN, t)
        elif ev == "error":
            self.log_line(f"step failed: {e.get('note')}", EMBER, t)
        elif ev in ("halted", "KILL_SWITCH") and not f:
            self.log_line(e.get("note") or "kill switch fired", EMBER, t)
        elif ev == "skip":
            self.log_line(e.get("note", ""), CYAN, t)
        elif ev == "bet_open":
            f = e["fill"]
            self.log_line(f"bet {e['coin'].lower()}: {e['side']} {float(f['size']):.6g} at {px(f['price'])} for {e['H']}h "
                          f"(P(up) {e['p_up']:.0%})", CYAN, t)
            self.fills.append({"t": t, "price": f["price"], "size": float(f["size"]),
                               "side": "buy" if e["side"] == "long" else "sell", "coin": e["coin"]})
        elif ev == "bet_close":
            f = e["fill"]
            self.log_line(f"bet {e['coin'].lower()} closed ({e['reason']}): {'+' if e['pnl'] >= 0 else '−'}${abs(e['pnl']):.2f}",
                          HOT if e["pnl"] >= 0 else EMBER, t)
            self.fills.append({"t": t, "price": f["price"], "size": float(f["size"]),
                               "side": "sell" if e["side"] == "long" else "buy", "coin": e["coin"]})
        elif ev in ("pool_halted", "pool_shorts_off", "pool_error"):
            self.log_line(f"bet pool: {e.get('note')}", EMBER, t)
        elif ev == "pool_skip":
            self.log_line(f"bet pool: {e.get('note')}", CYAN, t)
        elif ev == "pool_startup":
            self.log_line(f"bet pool online: {usd(e['ledger']['usd'])} cash, betting on {', '.join(e['coins']).lower()}"
                          f"{' (long and short)' if e.get('shorts') else ' (long only)'}", CYAN, t)
        elif ev in ("entry_unfilled", "scale_up_unfilled"):
            self.log_line("entry order expired unfilled; staying flat", CYAN, t)
        elif ev == "exit_unfilled":
            self.log_line("exit did not fill; will retry next hour", EMBER, t)
        if ev in TRADE_EVENTS and f:
            side = "sell" if ev in ("exit", "manual_sell", "KILL_SWITCH") else "buy"
            if not ev.startswith("manual_"):  # manual trades were already echoed by the command reply
                pnl = f", {pct(e['pnl_pct'])} on the trade" if e.get("pnl_pct") is not None else ""
                self.log_line(f"{ev.replace('_', ' ')}: {side} {float(f['size']):.6g} {e.get('coin', 'BTC')} at {px(f['price'])}, "
                              f"fee {usd(f['fee'])}{pnl}", EMBER if side == "sell" else HOT, t)
            self.fills.append({"t": t, "price": f["price"], "size": float(f["size"]), "side": side,
                               "coin": e.get("coin", "BTC")})

    # --- input ------------------------------------------------------------------------------
    async def on_input_submitted(self, event):
        line = event.value.strip()
        event.input.value = ""
        await self.run_line(line)

    # click targets (see link()): chart range, coin, prompt prefill, read-only commands
    async def on_click(self, event):
        go = (event.style.meta or {}).get("go")
        if go:
            await self.run_action(go)

    async def action_view(self, view):
        if view in VIEWS:
            await self.load_view(view)

    def action_coin(self, coin):
        self.switch_coin(coin)

    def action_fill(self, text):
        p = self.query_one("#prompt", Prompt)
        p.value = text
        p.cursor_position = len(text)
        p.focus()

    async def action_run(self, line):
        if line in ("status", "help", "bets"):  # never a money-moving command from a single click
            await self.run_line(line)

    async def run_line(self, line):
        if not line:
            return
        self.history.append(line)
        self.hist_i = -1
        self.log_line(f"> {line}", f"bold {HOT}")
        cmd, _, arg = line.lower().partition(" ")
        if cmd in ("quit", "exit"):
            self.exit()
            return
        if cmd == "clear":
            self.action_clear()
            return
        if cmd == "coin":
            want = {"bitcoin": "BTC", "xbt": "BTC", "ethereum": "ETH", "ether": "ETH", "etherium": "ETH",
                    "solana": "SOL", "ripple": "XRP"}.get(arg.strip(), arg.strip().upper())
            if want in self.coin_list():
                self.switch_coin(want)
            else:
                self.log_line(f"coins here: {', '.join(self.coin_list()).lower()}", EMBER)
            return
        if cmd == "view":
            if arg.strip() in VIEWS:
                await self.load_view(arg.strip())
            else:
                self.log_line("views are 10m, 1h, 1d and 1w", EMBER)
            return
        if not self.token:
            self.log_line("not connected to the bot", EMBER)
            return
        try:
            r = await asyncio.to_thread(lambda: self.http.post(
                BASE + "/api/command", json={"line": line, "coin": self.coin}, timeout=120,
                headers={"X-Btcbot-Token": self.token, "Origin": BASE}))
            if r.status_code == 403:
                self.log_line("session expired (the bot restarted); reconnecting", EMBER)
                self.token = None
                self.run_worker(self.boot())
                return
            d = r.json()
        except Exception as e:
            self.log_line(f"could not reach the bot: {e}", EMBER)
            return
        for out in d.get("lines", []):
            self.log_line(out, EMBER if d.get("error") else HOT if d.get("ok") else AMBER)
        self.query_one("#prompt").set_class(bool(d.get("confirm")), "pending")
        await self.poll_status()

    def action_clear(self):
        self.query_one("#console", RichLog).clear()


def start_bot_in_process():
    """Run every coin's trader + the local server inside this process (only when no bot is running)."""
    from . import app as web
    from .trader import configure
    pf, _ = configure([])
    # the traders log to state/*.log; drop console handlers so they can't paint over the TUI
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    web.serve(pf, PORT)


def main():
    ap = argparse.ArgumentParser(description="Phosphor Desk terminal UI")
    ap.add_argument("--start", action="store_true", help="start the bot in this process if none is running")
    a = ap.parse_args()
    try:
        requests.get(BASE + "/api/status", timeout=3)
        running = True
    except requests.RequestException:
        running = False
    if not running:
        if not a.start:
            raise SystemExit(f"no bot is listening on {BASE}. Start it with ./run_live.sh "
                             "(paper: python -m btcbot.app), or add --start to host it in this terminal.")
        start_bot_in_process()
        time.sleep(1.5)
    Desk().run()


if __name__ == "__main__":
    main()
