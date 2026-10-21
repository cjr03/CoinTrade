"""Terminal UI for the trading bot: live chart, model notes, command line.

    python -m btcbot.app            # http://127.0.0.1:8787  (env vars as for btcbot.trader)

Runs one hourly trader per coin (BTCBOT_COINS) in background threads and serves a local-only UI.  Commands
that move money in live mode require a typed confirmation code.  The server only listens on
127.0.0.1 and every POST must carry the per-session token embedded in the page.
"""
import json
import logging
import math
import os
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

from .coins import COINS, coin_symbol
from .kraken import public as kraken_public
from .trader import ART, STATE, Busy, configure

WEB = Path(__file__).resolve().parent / "web"
PORT = int(os.environ.get("BTCBOT_PORT", 8787))
TOKEN = secrets.token_urlsafe(24)
BOOT_ID = secrets.token_hex(4)
log = logging.getLogger("btcbot")

VIEWS = {  # view -> (kraken interval minutes, coinbase granularity, seconds per bar, bars)
    "1h": (1, "ONE_MINUTE", 60, 60),
    "1d": (5, "FIVE_MINUTE", 300, 288),
    "1w": (60, "ONE_HOUR", 3600, 168),
}


# --- market data ------------------------------------------------------------------------------
_cache = {}


def cached(key, ttl, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = fn()
    _cache[key] = (time.time(), val)
    return val


def candles(exchange, view, coin="BTC"):
    c = COINS[coin]
    if view == "10m":
        return {"kind": "ticks", "points": ticks_10m(exchange, coin)}
    k_int, cb_gran, secs, n = VIEWS[view]
    if exchange == "kraken":
        res = kraken_public("OHLC", pair=c["kraken_pair"], interval=k_int)
        rows = res[c["kraken_key"]][-n:]
        bars = [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6])] for r in rows]
    else:
        end = int(time.time()) // secs * secs
        r = requests.get(f"https://api.coinbase.com/api/v3/brokerage/market/products/{c['product']}/candles",
                         params={"start": end - secs * (n - 1), "end": end, "granularity": cb_gran, "limit": 350},
                         timeout=10)
        r.raise_for_status()
        bars = sorted([[int(x["start"]), float(x["open"]), float(x["high"]), float(x["low"]), float(x["close"]),
                        float(x["volume"])] for x in r.json()["candles"]])
    return {"kind": "candles", "secs": secs, "bars": bars}


def ticks_10m(exchange, coin="BTC"):
    c = COINS[coin]
    since = time.time() - 600
    pts = []
    if exchange == "kraken":
        cursor = str(int(since * 1e9))
        for _ in range(12):  # Kraken returns <= 1000 trades per page; page until we reach now
            res = kraken_public("Trades", pair=c["kraken_pair"], since=cursor)
            rows = res[c["kraken_key"]]
            pts += [[float(t[2]), float(t[0])] for t in rows]
            if len(rows) < 1000 or (pts and pts[-1][0] > time.time() - 5):
                break
            cursor = res["last"]
    else:
        r = requests.get(f"https://api.coinbase.com/api/v3/brokerage/market/products/{c['product']}/candles",
                         params={"start": int(since), "end": int(time.time()), "granularity": "ONE_MINUTE"}, timeout=10)
        r.raise_for_status()
        pts = sorted([[int(x["start"]) + 30, float(x["close"])] for x in r.json()["candles"]])
    pts = [p for p in pts if p[0] >= since]
    if len(pts) > 1500:  # thin out bursts; keep the shape
        step = len(pts) / 1500
        pts = [pts[int(i * step)] for i in range(1500)]
    return pts


def ticker(exchange, coin="BTC"):
    c = COINS[coin]
    if exchange == "kraken":
        t = kraken_public("Ticker", pair=c["kraken_pair"])[c["kraken_key"]]
        return {"bid": float(t["b"][0]), "ask": float(t["a"][0]), "last": float(t["c"][0]),
                "open24": float(t["o"]), "high24": float(t["h"][1]), "low24": float(t["l"][1]),
                "vol24": float(t["v"][1]), "t": time.time()}
    r = requests.get(f"https://api.coinbase.com/api/v3/brokerage/market/products/{c['product']}", timeout=10)
    r.raise_for_status()
    p = r.json()
    last = float(p["price"])
    chg = float(p.get("price_percentage_change_24h") or 0) / 100
    return {"bid": last, "ask": last, "last": last, "open24": last / (1 + chg), "high24": None, "low24": None,
            "vol24": float(p.get("volume_24h") or 0), "t": time.time()}


def markets(exchange, coins):
    """Live price, 24h change/range and volume for every coin (Kraken: one Ticker call for all)."""
    out = {}
    if exchange == "kraken":
        res = kraken_public("Ticker", pair=",".join(COINS[c]["kraken_pair"] for c in coins))
        for c in coins:
            t = res[COINS[c]["kraken_key"]]
            last, open24 = float(t["c"][0]), float(t["o"])
            out[c] = {"last": last, "bid": float(t["b"][0]), "ask": float(t["a"][0]), "open24": open24,
                      "chg24": last / open24 - 1 if open24 else None, "high24": float(t["h"][1]),
                      "low24": float(t["l"][1]), "vol24_usd": float(t["v"][1]) * float(t["p"][1])}
    else:
        for c in coins:
            t = ticker(exchange, c)
            out[c] = {**t, "chg24": t["last"] / t["open24"] - 1 if t.get("open24") else None, "vol24_usd": None}
    return out


def fp(x):
    """Prices from $1 (xrp) to $80,000 (btc), never in scientific notation."""
    return f"{x:,.1f}" if x >= 1000 else f"{x:,.2f}" if x >= 10 else f"{x:.4f}"


def tick(exchange, coin):
    return cached(f"tick:{exchange}:{coin}", 2, lambda: ticker(exchange, coin))


def portfolio_value(pf):
    """Everything the bot manages, valued at live bids. Moves between sleeves/pool cancel out, so
    `basis` is what the owner actually put in (net of withdrawals) and pnl is the real gain/loss."""
    parts, equity, basis, fees = {}, 0.0, 0.0, 0.0
    try:
        mk = cached(f"markets:{pf.exchange}", 2, lambda: markets(pf.exchange, pf.coins))
    except Exception as e:
        log.warning("markets: %s", e)
        mk = {}
    prices = {c: (mk.get(c) or {}).get("bid") for c in pf.coins}
    for c in pf.coins:
        L = pf[c].ledger
        px = prices[c] if prices[c] is not None else ((pf[c].s.get("signal") or {}).get("close") or 0.0)
        eq = L["usd"] + L["qty"] * px
        parts[c] = round(eq, 2)
        equity += eq
        basis += L["deposited"] - L["withdrawn"]
        fees += L["fees"]
    bets = {}
    if pf.pool:
        L = pf.pool.ledger
        eq = pf.pool.equity({c: p for c, p in prices.items() if p})
        parts["BETS"] = round(eq, 2)
        equity += eq
        basis += L["deposited"] - L["withdrawn"]
        fees += L["fees"]
        bets = dict(pf.pool.s.get("bets") or {})
    coins = {}
    for c in pf.coins:  # one live row per coin: market data + everything the bot holds of it
        m = mk.get(c) or {}
        px = prices[c] or 0.0
        t = pf[c]
        bet = bets.get(c)
        qty_trend = t.ledger["qty"]
        qty_bet = bet["qty"] if bet and bet["side"] > 0 else 0.0
        sig = t.s.get("signal") or {}
        coins[c] = {"last": m.get("last"), "chg24": m.get("chg24"), "high24": m.get("high24"),
                    "low24": m.get("low24"), "vol24_usd": m.get("vol24_usd"),
                    "qty": qty_trend + qty_bet, "qty_trend": qty_trend, "qty_bet": qty_bet,
                    "value": (qty_trend + qty_bet) * px + t.ledger["usd"],
                    "trend": None if not sig else ("up" if sig.get("pred", 0) > 0 else "down"),
                    "bet": None if not bet else {"side": "long" if bet["side"] > 0 else "short",
                                                 "entry": bet["entry_px"],
                                                 "pnl_pct": (px / bet["entry_px"] - 1) * bet["side"] if px else None},
                    "position": bool(t.s.get("position"))}
    return {"equity": equity, "basis": basis, "pnl": equity - basis, "fees": fees, "parts": parts,
            "coins": coins, "priced": all(p is not None for p in prices.values())}


# --- the model's notes ------------------------------------------------------------------------
def research_numbers(exchange):
    """Walk-forward results at this exchange's fees (Kraken file first, then the Coinbase base case)."""
    fee_key = "kraken" if exchange == "kraken" else "base"
    out = []
    for f in sorted(ART.glob("report_H*.json")):
        out += [r for r in json.loads(f.read_text()) if r["fees"].startswith(fee_key)]
    return out


def coin_research():
    path = ART / "coins_report.json"
    return json.loads(path.read_text()) if path.exists() else {}


def notes(pf, coin):
    st = pf[coin].status()
    pol, s = st["policy"], st["state"]
    name = COINS[coin]["name"]
    rep = cached("research:" + st["exchange"], 600, lambda: research_numbers(st["exchange"]))
    cr = cached("coin-research", 600, coin_research).get(coin)
    sig = s.get("signal") or {}
    L = s.get("ledger", {})
    out = [
        {"kind": "philosophy", "text": "I learned from eleven years of hourly candles and was judged only on "
                                       "31 quarters I never saw while training. Anything that only worked in-sample was thrown out."},
        {"kind": "philosophy", "text": f"Fees are the opponent. A round trip here costs {pol['round_trip']:.2%}, "
                                       f"more than {name} moves on a typical day, so I trade rarely and post-only."},
    ]
    if rep and coin == "BTC":
        b = max(rep, key=lambda r: r["strategy"]["sharpe"])
        out.append({"kind": "research", "text": f"Best ML result after fees: {b['model']} on a {b['H']}h horizon, "
                    f"{b['strategy']['cagr']:+.1%} a year against buy-and-hold's {b['buy_hold']['cagr']:+.1%}. "
                    f"Out-of-sample IC {b['ic_mean']:.3f}, barely above coin-flipping and far too faint to pay the fees."})
    if cr:
        out.append({"kind": "research", "text": f"{coin} since {cr['start'][:4]}: the {pol.get('sma_days', 50)}-day trend rule made "
                    f"{cr['trend']['cagr']:+.0%} a year (worst drop {cr['trend']['max_dd']:.0%}); holding made "
                    f"{cr['hold']['cagr']:+.0%} (worst drop {cr['hold']['max_dd']:.0%})."})
    if pol["model"] == "momentum":
        out.append({"kind": "philosophy", "text": f"I hold {name} while its daily close is above its "
                    f"{pol['sma_days']}-day average and sit in cash otherwise. It is a drawdown rule, not a crystal ball."})
    else:
        out.append({"kind": "philosophy", "text": f"I only enter when the expected {pol['horizon']}h return clears "
                    f"{(pol['entry_thr'] or 0):.2%}. Most hours it doesn't, and sitting out is the trade."})
    if sig:
        if pol["model"] == "momentum":
            gap = sig["daily_close"] / sig["sma"] - 1 if sig.get("sma") else 0
            out.append({"kind": "projection", "text": f"{coin} trend is {'up' if sig['pred'] > 0 else 'down'}: yesterday's "
                        f"close {fp(sig.get('daily_close', 0))} sits {gap:+.1%} from the {pol['sma_days']}-day "
                        f"average {fp(sig.get('sma', 0))}, so I want to be {'long' if sig['pred'] > 0 else 'in cash'}."})
        else:
            mid = sig["close"] * math.exp(sig["pred"])
            act = "would buy" if pol["entry_thr"] is not None and sig["pred"] > pol["entry_thr"] else "wait"
            out.append({"kind": "projection", "text": f"Next {sig['horizon']}h: {sig['pred']:+.2%} to about {mid:,.0f}, "
                        f"one-sigma band ±{sig['sigma_h']:.1%}. Entry needs {(pol['entry_thr'] or 0):+.2%}, so I {act}."})
    pos = s.get("position")
    if pos:
        stop = f", stop at {fp(pos['stop_px'])}" if pos.get("stop_px", 0) > 1e-9 and pos["stop_px"] > pos["entry_px"] * 0.01 else ""
        who = " (your manual buy: I only guard it with the kill switch)" if pos.get("manual") else ""
        out.append({"kind": "position", "text": f"Holding {pos['qty']:.6g} {coin} from {fp(pos['entry_px'])}{stop}{who}."})
    shown = ("entry", "exit", "scale_up", "manual_buy", "manual_sell", "KILL_SWITCH", "deposit", "withdraw", "move")
    for e in [e for e in list(pf.events) if e.get("event") in shown and e.get("coin") == coin][-4:]:
        f = e.get("fill") or {}
        if f:
            pnl = f", {e['pnl_pct']:+.2%} on the trade" if e.get("pnl_pct") is not None else ""
            out.append({"kind": "trade", "text": f"{e['event'].replace('_', ' ')}: {f.get('size', 0):.6g} {coin} at "
                        f"{fp(f.get('price', 0))}, fee ${f.get('fee', 0):.2f}{pnl}"})
        elif e["event"] == "KILL_SWITCH":
            out.append({"kind": "alert", "text": f"The {coin} kill switch fired but found nothing to sell."})
        else:
            out.append({"kind": "trade", "text": f"{e['event']} ${e.get('amount', 0):,.2f}"})
    if L:
        out.append({"kind": "account", "text": f"{coin} sleeve: ${L.get('usd', 0):,.2f} cash, {L.get('qty', 0):.6g} {coin}, "
                    f"${L.get('fees', 0):,.2f} paid in fees so far."})
    if pf.pool and coin in pf.pool.models:
        m = pf.pool.models[coin]
        v = m["validation"]
        ps = pf.pool.s
        bet = ps["bets"].get(coin)
        sig = (ps.get("signals") or {}).get(coin)
        out.append({"kind": "research", "text": f"Short-term {coin} bets: {m['H']}h, placed when I'm at least "
                    f"{m['threshold']:.0%} sure of the direction. Over the last 6 months that would have meant "
                    f"{v['bets_per_year']:.0f} bets a year turning $100 into ${v['final_from_100']:.0f}; the pool is small for a reason."})
        if sig:
            out.append({"kind": "projection", "text": f"{coin} next {m['H']}h: {sig['p_up']:.0%} chance up"
                        + (f"; holding a {'long' if bet['side'] > 0 else 'short'} bet from {fp(bet['entry_px'])}." if bet else ".")})
    if s.get("halted"):
        out.append({"kind": "alert", "text": f"The {coin} kill switch is engaged. Type 'resume {coin.lower()}' once you know why."})
    if s.get("needs_review"):
        out.append({"kind": "alert", "text": f"{coin} needs review: {s['needs_review']}"})
    return out


# --- commands ---------------------------------------------------------------------------------
HELP = [
    "coins: btc eth sol xrp. a command acts on the coin on screen unless you name one.",
    "buy 25 eth      buy $25 of ethereum now (market order, taker fee)",
    "buy all sol     put all of the solana sleeve's cash into solana",
    "sell 25 xrp     sell $25 worth      sell 50%      sell all",
    "deposit 50 eth  paper: add $50      live: hand $50 of exchange cash to that coin's sleeve",
    "withdraw 20     paper: remove $20   live: send $20 to your bank (needs opt-in)",
    "move 20 btc eth move $20 of cash from the btc sleeve to the eth sleeve (or to/from 'bets')",
    "convert 20 btc eth   sell $20 of btc and buy eth with it     convert all sol xrp",
    "rebalance btc 50 eth 30 sol 20   redistribute everything to those percentages",
    "bets            short-term bet pool status    bets on | off | resume | shorts on | shorts off",
    "bets fund on    let a bet sell another coin for its stake (capped)    bets fund off: pool cash only",
    "coin sol        show solana         (or click the coin or its market row)",
    "bot off | on    pause or resume automatic trading (add 'all' for every coin)",
    "resume          re-arm after the kill switch or a review flag",
    "view 1d         switch the chart: 10m, 1h, 1d, 1w",
    "status   clear   help",
]
PENDING = {}


def parse_amount(arg):
    m = re.fullmatch(r"\$?([0-9]*\.?[0-9]+)\s*(usd|\$|%)?", arg.strip().lower())
    if not m:
        raise ValueError(f"can't read the amount '{arg}'; try 25, $25, 50% or all")
    return float(m.group(1)), (m.group(2) or "usd").replace("$", "usd")


def split_coins(arg):
    """Pull coin words out of an argument string: ('25 eth', ...) -> ('25', ['ETH'])."""
    words, coins = [], []
    for w in arg.split():
        c = coin_symbol(w)
        (coins if c else words).append(c or w)
    return " ".join(words), coins


def run_command(pf, line, coin="BTC"):
    parts = line.strip().split(maxsplit=1)
    if not parts:
        return {"lines": []}
    cmd, arg = parts[0].lower(), (parts[1].strip().lower() if len(parts) > 1 else "")
    rest, named = split_coins(arg)
    if cmd == "move":
        return move_command(pf, arg)
    if cmd == "convert":
        return convert_command(pf, rest, named)
    if cmd == "rebalance":
        return rebalance_command(pf, arg)
    if cmd in ("bets", "pool"):
        return bets_command(pf, arg)
    if len(named) > 1:
        return {"lines": ["name one coin, for example 'buy 25 eth'"], "error": True}
    coin = named[0] if named else coin
    if coin not in pf.coins:
        return {"lines": [f"{coin} isn't enabled; start the bot with BTCBOT_COINS including {coin}"], "error": True}
    t = pf[coin]
    st = t.status()
    live = st["mode"] == "live"
    L = st["state"]["ledger"]

    def price():
        try:
            return tick(pf.exchange, coin)["bid"]
        except Exception as e:
            raise ValueError(f"no {coin} price from the exchange right now ({e}); try again in a moment")

    if cmd == "help":
        return {"lines": HELP}
    if cmd == "status":
        lines, total = [], 0.0
        for c in pf.coins:
            Lc = pf[c].ledger
            try:
                px = tick(pf.exchange, c)["bid"]
            except Exception:
                px = float("nan")
            eq = Lc["usd"] + Lc["qty"] * px
            total += eq if math.isfinite(eq) else 0.0
            s = pf[c].s
            flag = " halted" if s.get("halted") else " needs review" if s.get("needs_review") else ""
            lines.append(f"{c:<4} equity ${eq:,.2f}   cash ${Lc['usd']:,.2f}   {Lc['qty']:.6g} {c}   "
                         f"bot {'on' if s.get('auto', True) else 'off'}{flag}")
        pv = portfolio_value(pf)
        if pf.pool:
            lines.append(f"bets equity ${pv['parts'].get('BETS', 0):,.2f} (see 'bets')")
        lines.append(f"total ${pv['equity']:,.2f}, put in ${pv['basis']:,.2f}, {'+' if pv['pnl'] >= 0 else '−'}"
                     f"${abs(pv['pnl']):,.2f} ({pv['pnl'] / pv['basis']:+.1%}) after ${pv['fees']:.2f} fees, "
                     f"on {pf.exchange} ({st['mode']})" if pv["basis"] else f"total ${pv['equity']:,.2f}")
        return {"lines": lines}
    if cmd == "bot" and rest in ("on", "off", "on all", "off all"):
        on = rest.startswith("on")
        targets = pf.coins if rest.endswith("all") else [coin]
        for c in targets:
            pf[c].set_auto(on)
        return {"lines": [f"automatic trading {'resumed' if on else 'paused'} for {', '.join(targets)}"], "ok": True}
    if cmd == "resume":
        cleared = t.resume()
        return {"lines": [f"{coin}: cleared {', '.join(cleared)}" if cleared else f"{coin}: nothing to resume"],
                "ok": bool(cleared)}
    if cmd == "cancel":
        PENDING.clear()
        return {"lines": ["pending action cancelled"]}
    if cmd == "confirm":
        action = PENDING.pop(arg.upper(), None)
        if not action or time.time() - action["t"] > 120:
            return {"lines": ["no pending action with that code (codes expire after 2 minutes)"], "error": True}
        return execute(pf, action)

    if cmd in ("buy", "sell", "deposit", "withdraw"):
        if not rest:
            return {"lines": [f"how much? for example '{cmd} 25 {coin.lower()}'"], "error": True}
        try:
            amt, unit = (100.0, "%") if rest == "all" else parse_amount(rest)
            if not (0 < amt < 1e7) or (unit == "%" and amt > 100):
                raise ValueError("amount out of range")
            px = price() if cmd in ("buy", "sell") else None
        except ValueError as e:
            return {"lines": [str(e)], "error": True}
        if cmd == "buy":
            usd = L["usd"] * amt / 100 if unit == "%" else amt
            if usd > L["usd"] + 1e-9:
                return {"lines": [f"the {coin} sleeve has ${L['usd']:,.2f} in cash"], "error": True}
            action = {"cmd": "buy", "coin": coin, "usd": usd,
                      "desc": f"buy ${usd:,.2f} of {COINS[coin]['name']} at market (about {fp(px)}); a manual buy is "
                              "yours: the bot won't sell it on a signal, only on the kill switch"}
        elif cmd == "sell":
            if L["qty"] <= 0:
                return {"lines": [f"the {coin} sleeve holds no {coin} to sell"], "error": True}
            qty = L["qty"] * amt / 100 if unit == "%" else amt / px
            if qty > L["qty"] * 1.0001 + 1e-12:
                return {"lines": [f"the {coin} sleeve holds {L['qty']:.6g} {coin} (${L['qty'] * px:,.2f})"], "error": True}
            qty = min(qty, L["qty"])
            action = {"cmd": "sell", "coin": coin, "qty": qty,
                      "desc": f"sell {qty:.6g} {coin} (about ${qty * px:,.2f}) at market"}
        else:
            if unit != "usd":
                return {"lines": [f"{cmd} takes a dollar amount, for example '{cmd} 50 {coin.lower()}'"], "error": True}
            dest = " to your bank account" if cmd == "withdraw" and live else ""
            action = {"cmd": cmd, "coin": coin, "usd": amt, "desc": f"{cmd} ${amt:,.2f} ({coin} sleeve){dest}"}
        return stage(action, live, pf)
    return {"lines": [f"unknown command '{cmd}'. type 'help' for the list."], "error": True}


def move_command(pf, arg):
    """'move 20 btc eth', 'move 20 from btc to eth' or 'move 20 to eth from btc'."""
    src = dst = None
    plain, amount = [], []
    words = arg.split()
    for i, w in enumerate(words):
        c = "BETS" if w in ("bets", "pool", "bet") else coin_symbol(w)
        prev = words[i - 1] if i else ""
        if c and prev == "to":
            dst = c
        elif c and prev == "from":
            src = c
        elif c:
            plain.append(c)
        elif w not in ("to", "from"):
            amount.append(w)
    for c in plain:  # coins without from/to fill the remaining slots in order: source, then destination
        if src is None:
            src = c
        elif dst is None:
            dst = c
        else:
            src = None
    if not src or not dst or src == dst:
        return {"lines": ["move needs two different coins, for example 'move 20 btc eth' or "
                          "'move 20 from btc to eth'"], "error": True}
    known = pf.coins + (["BETS"] if pf.pool else [])
    if src not in known or dst not in known:
        return {"lines": [f"you can move cash between {', '.join(c.lower() for c in known)}"], "error": True}
    try:
        amt, unit = parse_amount(" ".join(amount))
        if unit != "usd" or not 0 < amt < 1e7:
            raise ValueError("move takes a dollar amount, for example 'move 20 btc eth'")
    except ValueError as e:
        return {"lines": [str(e)], "error": True}
    action = {"cmd": "move", "usd": amt, "src": src, "dst": dst,
              "desc": f"move ${amt:,.2f} of cash from the {src} sleeve to the {dst} sleeve"}
    return stage(action, pf.live, pf)


def convert_command(pf, rest, named):
    if len(named) != 2 or named[0] == named[1] or any(c not in pf.coins for c in named):
        return {"lines": ["convert needs two enabled coins, for example 'convert 20 btc eth' or 'convert all sol xrp'"],
                "error": True}
    src, dst = named
    usd = fraction = None
    try:
        if rest.strip() != "all":
            amt, unit = parse_amount(rest)
            if unit == "%":
                if not 0 < amt <= 100:
                    raise ValueError("a percentage must be between 0 and 100")
                fraction = amt / 100
            elif not 0 < amt < 1e7:
                raise ValueError("amount out of range")
            else:
                usd = amt
    except ValueError as e:
        return {"lines": [str(e)], "error": True}
    what = (f"{fraction:.0%} of the {src}" if fraction is not None else f"all {src}" if usd is None
            else f"${usd:,.2f} of {src}")
    action = {"cmd": "convert", "src": src, "dst": dst, "usd": usd, "fraction": fraction,
              "desc": f"sell {what} and buy {dst} with the proceeds (two market orders, about 0.8% fee each; "
                      f"the {dst} becomes your manual holding)"}
    return stage(action, pf.live, pf)


def rebalance_command(pf, arg):
    words, weights, pending = arg.replace("%", " ").split(), {}, None
    bad = {"lines": ["give each coin a percentage, for example 'rebalance btc 50 eth 30 sol 20'"], "error": True}
    for w in words:
        c = coin_symbol(w)
        if c:
            if pending or c in weights:
                return bad  # a coin without a number, or named twice
            pending = c
        elif pending:
            try:
                v = float(w)
            except ValueError:
                return {"lines": [f"can't read '{w}' as a percentage"], "error": True}
            if not 0 <= v <= 100:
                return {"lines": [f"{v:g}% is not a percentage between 0 and 100"], "error": True}
            weights[pending] = v
            pending = None
        else:
            return {"lines": [f"'{w}' is not a coin I trade here"], "error": True}
    if not weights or pending or any(c not in pf.coins for c in weights):
        return {"lines": ["give each coin a percentage, for example 'rebalance btc 50 eth 30 sol 20'"], "error": True}
    if sum(weights.values()) > 100.0001:
        return {"lines": [f"those add up to {sum(weights.values()):.0f}%; keep it at 100% or less"], "error": True}
    plan = ", ".join(f"{c.lower()} {w:g}%" for c, w in weights.items())
    others = [c for c in pf.coins if c not in weights]
    tail = f"; {', '.join(o.lower() for o in others)} go to 0%" if others else ""
    action = {"cmd": "rebalance", "weights": weights,
              "desc": f"rebalance the sleeves to {plan}{tail} (market orders, about 0.8% fee per trade; "
                      "coins bought become manual holdings the trend bot won't sell on a signal)"}
    return stage(action, pf.live, pf)


def bets_command(pf, arg):
    if not pf.pool:
        return {"lines": ["the bet pool is not running; start the bot with BTCBOT_BETS=1"], "error": True}
    pool = pf.pool
    if arg in ("on", "off"):
        pool.set_auto(arg == "on")
        return {"lines": [f"short-term bets {'resumed' if arg == 'on' else 'paused'}"], "ok": True}
    if arg == "resume":
        cleared = pool.resume()
        return {"lines": [f"bet pool: cleared {', '.join(cleared)}" if cleared else "bet pool: nothing to resume"]}
    if arg in ("fund on", "fund off"):
        with pool.lock:
            pool.s["autofund"] = arg.endswith("on")
            pool.journal(event="pool_autofund", on=pool.s["autofund"])
            pool.save()
        return {"lines": ["bets may now sell another coin (or use idle sleeve cash) for their stake, up to the "
                          "pool's funding cap" if arg.endswith("on") else
                          "bets will only use the pool's own cash"], "ok": True}
    if arg in ("shorts on", "shorts off"):
        with pool.lock:
            pool.s["shorts"] = arg.endswith("on")
            pool.save()
        return {"lines": [f"short bets {'enabled' if arg.endswith('on') else 'disabled'}"], "ok": True}
    st = pool.status()
    L = st["ledger"]
    prices = {}
    for c in st["bets"]:
        try:
            prices[c] = tick(pf.exchange, c)["bid"]
        except Exception:
            pass
    eq = pool.equity(prices)
    basis = L["deposited"] - L["withdrawn"]
    lines = [f"bet pool ${eq:,.2f} (put in ${basis:,.2f}, realized {L['realized']:+.2f}, fees ${L['fees']:.2f})"
             f"{' HALTED' if st.get('halted') else ''}{' needs review' if st.get('needs_review') else ''}"
             f", bets {'on' if st.get('auto', True) else 'off'}, shorts {'on' if st.get('shorts') else 'off'}"
             f", funding from coins {'on' if st.get('autofund', True) else 'off'}"]
    for c, m in st["models"].items():
        b = st["bets"].get(c)
        v = m["validation"]
        now = (f"open {'long' if b['side'] > 0 else 'short'} {b['qty']:.6g} @ {fp(b['entry_px'])}" if b else "no bet")
        sig = (st.get("signals") or {}).get(c)
        p = f"P(up) {sig['p_up']:.0%}" if sig else "P(up) —"
        lines.append(f"{c:<4} {m['H']}h bets at P>={m['threshold']:.0%}   {p:<12} {now}   "
                     f"(last 6 months: {v['bets_per_year']:.0f}/yr, $100 -> ${v['final_from_100']:.0f})")
    return {"lines": lines}


def stage(action, live, pf):
    action["t"] = time.time()
    if live:
        code = secrets.token_hex(2).upper()
        PENDING.clear()
        PENDING[code] = action
        return {"lines": [f"live account: {action['desc']}.", f"type 'confirm {code}' within 2 minutes, or 'cancel'."],
                "confirm": code}
    return execute(pf, action)


def execute(pf, a):
    try:
        if a["cmd"] == "move":
            pf.move(a["usd"], a["src"], a["dst"])
            return {"lines": [f"moved ${a['usd']:,.2f} from {a['src'].lower()} to {a['dst'].lower()}"], "ok": True}
        if a["cmd"] == "convert":
            sold, bought = pf.convert(a["src"], a["dst"], a["usd"], a.get("fraction"))
            return {"lines": [f"sold {sold['size']:.6g} {a['src']} at {fp(sold['price'])}, bought {bought['size']:.6g} "
                              f"{a['dst']} at {fp(bought['price'])} (fees ${sold['fee'] + bought['fee']:.2f})"], "ok": True}
        if a["cmd"] == "rebalance":
            done = pf.rebalance(a["weights"])
            lines = [f"{side} {f['size']:.6g} {c} at {fp(f['price'])}" for side, c, f in done] or ["already balanced"]
            return {"lines": lines, "ok": True}
        coin, t = a["coin"], pf[a["coin"]]
        if a["cmd"] == "buy":
            f = t.manual_buy(a["usd"])
            return {"lines": [f"bought {f['size']:.6g} {coin} at {fp(f['price'])}, fee ${f['fee']:.2f}"], "ok": True}
        if a["cmd"] == "sell":
            f = t.manual_sell(a["qty"])
            return {"lines": [f"sold {f['size']:.6g} {coin} at {fp(f['price'])}, fee ${f['fee']:.2f}"], "ok": True}
        if a["cmd"] == "deposit":
            note = t.deposit(a["usd"])
            return {"lines": [f"deposited ${a['usd']:,.2f} to {coin} ({note})"], "ok": True}
        if a["cmd"] == "withdraw":
            res = t.withdraw(a["usd"])
            return {"lines": [f"withdrew ${a['usd']:,.2f} from {coin}, reference {res.get('refid')}"], "ok": True}
    except (Busy, ValueError, PermissionError, NotImplementedError, RuntimeError, OSError) as e:
        return {"lines": [str(e)], "error": True}
    return {"lines": ["nothing done"], "error": True}


# --- HTTP -------------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    portfolio = None
    caller = None  # 15-minute calls (btcbot.calls), when their models exist
    forecaster = None  # chart forecast curves (btcbot.forecast), when their fits exist
    server_version = "btcbot"

    def log_message(self, fmt, *args):
        pass

    def _send(self, body, ctype, code=200, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj, default=str).encode(), "application/json", code)

    def _local(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in ("127.0.0.1", "localhost")  # blocks DNS rebinding

    def _coin(self, q):
        c = coin_symbol(q.get("coin", "")) or self.portfolio.coins[0]
        return c if c in self.portfolio.coins else self.portfolio.coins[0]

    def do_GET(self):
        if not self._local():
            return self._json({"error": "forbidden"}, 403)
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        pf = self.portfolio
        ex = pf.exchange
        try:
            if u.path in ("/", "/index.html"):
                csp = ("default-src 'self'; style-src 'self' https://fonts.googleapis.com; "
                       "font-src https://fonts.gstatic.com; connect-src 'self' wss://ws.kraken.com "
                       "wss://advanced-trade-ws.coinbase.com; frame-ancestors 'none'")
                html = (WEB / "index.html").read_text().replace("__TOKEN__", TOKEN)
                self._send(html.encode(), "text/html; charset=utf-8", extra={"Content-Security-Policy": csp})
            elif u.path in ("/app.js", "/style.css", "/favicon.svg"):
                ctype = {"js": "text/javascript", "css": "text/css", "svg": "image/svg+xml"}[u.path.rsplit(".", 1)[1]]
                self._send((WEB / u.path.lstrip("/")).read_bytes(), ctype)
            elif u.path == "/api/status":
                coin = self._coin(q)
                st = pf[coin].status()
                st["coin"] = coin
                st["coins"] = {}
                for c in pf.coins:
                    s = pf[c].s
                    st["coins"][c] = {"ledger": dict(s.get("ledger", {})), "position": s.get("position"),
                                      "auto": s.get("auto", True), "halted": s.get("halted"),
                                      "needs_review": s.get("needs_review"),
                                      "pred": (s.get("signal") or {}).get("pred"), "busy": pf[c].busy}
                try:
                    st["ticker"] = tick(ex, coin)
                except Exception as e:
                    st["ticker"] = None
                    log.warning("ticker %s: %s", coin, e)
                st["events"] = [e for e in list(pf.events) if e["seq"] > int(q.get("since", 0))][-80:]
                if pf.pool:
                    ps = pf.pool.status()
                    st["pool"] = {"ledger": ps["ledger"], "bets": ps["bets"], "auto": ps.get("auto", True),
                                  "halted": ps.get("halted"), "needs_review": ps.get("needs_review"),
                                  "shorts": ps.get("shorts"), "signal": (ps.get("signals") or {}).get(coin),
                                  "model": ps["models"].get(coin), "signals": ps.get("signals") or {},
                                  "models": {c: {"H": m["H"], "threshold": m["threshold"]}
                                             for c, m in ps["models"].items()},
                                  "autofund": ps.get("autofund", True)}
                st["portfolio"] = portfolio_value(pf)
                st["calls"] = self.caller.snapshot() if self.caller else None
                st["boot_id"] = BOOT_ID
                pend = [(k, a) for k, a in PENDING.items() if time.time() - a["t"] <= 120]
                st["pending"] = pend[0][0] if pend else None
                st["pending_desc"] = pend[0][1]["desc"] if pend else None
                self._json(st)
            elif u.path == "/api/candles":
                coin, view = self._coin(q), q.get("view", "1h")
                if view not in ("10m", *VIEWS):
                    return self._json({"error": "unknown view"}, 400)
                ttl = {"10m": 5, "1h": 15, "1d": 60, "1w": 300}[view]
                self._json(cached(f"c:{ex}:{coin}:{view}", ttl, lambda: candles(ex, view, coin)))
            elif u.path == "/api/forecast":
                coin = self._coin(q)
                if not self.forecaster or coin not in self.forecaster.fits:
                    return self._json({"error": "no forecast fit for this coin (python -m btcbot.forecast --fit)"}, 404)
                self._json(cached(f"forecast:{coin}", 20, lambda: self.forecaster.curve(coin, model_leans(pf, coin, self.caller))))
            elif u.path == "/api/notes":
                self._json(notes(pf, self._coin(q)))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            log.warning("GET %s failed: %s", u.path, e)
            self._json({"error": str(e)}, 502)

    def do_POST(self):
        if not self._local() or self.headers.get("X-Btcbot-Token") != TOKEN:
            return self._json({"error": "forbidden"}, 403)
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost"):
            return self._json({"error": "forbidden"}, 403)
        if urlparse(self.path).path != "/api/command":
            return self._json({"error": "not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 <= n <= 4096:
                return self._json({"error": "request too large"}, 413)
            body = json.loads(self.rfile.read(n) or b"{}")
            line = str(body.get("line", ""))[:200]
            raw = str(body.get("coin", "") or self.portfolio.coins[0])
            coin = coin_symbol(raw)
            if coin is None:
                return self._json({"lines": [f"unknown coin '{raw[:12]}'"], "error": True})
            self._json(run_command(self.portfolio, line, coin))
        except Exception as e:
            log.exception("command failed")
            self._json({"lines": [f"error: {e}"], "error": True})


def model_leans(pf, coin, caller):
    """What each model currently leans towards for `coin`, for the chart's forecast curve."""
    leans = []
    if caller:
        call = caller.snapshot().get(coin) or {}
        if call.get("p_up") is not None:
            leans.append(("15m model", call["p_up"], 15))
    # the bet model is left out: its validation lost money on every coin, so its P(up) isn't a forecast
    sig = pf[coin].s.get("signal") or {}
    if pf.policy.get("model") == "momentum" and sig.get("pred") is not None:
        leans.append(("trend", "up" if sig["pred"] > 0 else "down", None))
    return leans


def serve(pf, port=PORT):
    """Start every coin's trader thread and the HTTP server thread; returns the stop event."""
    Handler.portfolio = pf
    stop = threading.Event()
    threading.Thread(target=pf.run, kwargs={"stop": stop}, daemon=True, name="portfolio").start()
    try:  # 15-minute up/down calls: recommendations only, they never place an order
        from .calls import Caller
        caller = Caller(pf.coins, STATE)
        if caller.models:
            Handler.caller = caller
            threading.Thread(target=caller.run, kwargs={"stop": stop}, daemon=True, name="calls").start()
    except Exception:
        log.exception("15-minute calls unavailable")
    try:
        from .forecast import Forecaster
        Handler.forecaster = Forecaster(pf.coins)
    except Exception:
        log.exception("forecast curves unavailable")
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True, name="http").start()
    return stop, srv


def main():
    pf, _ = configure()
    stop, srv = serve(pf)
    print(f"btcbot: http://127.0.0.1:{PORT}  ({pf.exchange}, {'live' if pf.live else 'paper'}, coins {', '.join(pf.coins)})")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        stop.set()
        srv.shutdown()


if __name__ == "__main__":
    main()
