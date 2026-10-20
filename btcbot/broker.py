"""Broker adapters: PaperBroker (local simulation on live public data) and CoinbaseBroker (real orders).

Both expose the same small interface used by the trader:
    quote() -> (bid, ask)            balances() -> (usd, btc)          fees() -> (maker, taker)
    buy_maker(usd, deadline_s, ref_px)        sell_maker(btc, deadline_s, ref_px, fallback_taker)
    market_sell(btc)                 product_ok() -> bool              cancel_open_orders()

Maker orders mirror the backtest: post-only at the decision-time price (never chasing past it);
a resting buy only counts as filled when price trades through it.  Order loops never raise once
something has filled — they return the partial fill so the trader can always book it.
"""
import calendar
import logging
import math
import time
import uuid

import requests

PRODUCT = "BTC-USD"
PUBLIC = "https://api.coinbase.com/api/v3/brokerage/market"
TERMINAL = ("FILLED", "CANCELLED", "EXPIRED", "FAILED")
log = logging.getLogger("btcbot")


def floor_to(x, inc):
    return math.floor(x / inc + 1e-9) * inc


def public_quote(product=PRODUCT):
    r = requests.get(f"{PUBLIC}/product_book", params={"product_id": product, "limit": 1}, timeout=10)
    r.raise_for_status()
    book = r.json()["pricebook"]
    return float(book["bids"][0]["price"]), float(book["asks"][0]["price"])


def public_trades_since(ts, product=PRODUCT):
    """Recent public trade prices printed after unix time ts — used for paper fill checks."""
    r = requests.get(f"{PUBLIC}/products/{product}/ticker", params={"limit": 100}, timeout=10)
    r.raise_for_status()
    out = []
    for tr in r.json().get("trades", []):
        t = calendar.timegm(time.strptime(tr["time"][:19], "%Y-%m-%dT%H:%M:%S"))
        if t > ts:
            out.append(float(tr["price"]))
    return out


def _merge(a, b):
    """Combine two partial fills into one record."""
    if not a or not b:
        return a or b
    size = a["size"] + b["size"]
    return {"price": (a["price"] * a["size"] + b["price"] * b["size"]) / size, "size": size,
            "fee": a["fee"] + b["fee"], "liquidity": f"{a['liquidity']}+{b['liquidity']}"}


class PaperBroker:
    """Simulates post-only limit orders against the live public tape. A resting buy at P fills
    only when a public trade prints strictly below P (mirrors the backtest fill rule).
    `self.s["paper"]` plays the role of the exchange account balances."""

    min_order_usd = 1.0

    def __init__(self, state, maker=0.006, taker=0.012, poll=5, capital=100.0, max_notional=110.0, coin="BTC"):
        from .coins import COINS
        self.s = state
        self.s.setdefault("paper", {"usd": capital, "btc": 0.0})  # "btc" = units of this sleeve's coin
        self.maker, self.taker, self.poll, self.max_notional = maker, taker, poll, max_notional
        self.coin, self.product = coin, COINS[coin]["product"]

    def quote(self):
        return public_quote(self.product)

    def balances(self):
        return self.s["paper"]["usd"], self.s["paper"]["btc"]

    def total_usd(self):
        return self.s["paper"]["usd"]

    def fees(self):
        return self.maker, self.taker

    def product_ok(self):
        return True

    def cancel_open_orders(self):
        return []

    def deposit(self, usd):
        self.s["paper"]["usd"] += usd

    def withdraw(self, usd):
        if usd > self.s["paper"]["usd"] + 1e-9:
            raise ValueError("insufficient paper USD")
        self.s["paper"]["usd"] -= usd
        return {"refid": f"paper-{uuid.uuid4().hex[:8]}", "amount": usd}

    def _trades_since(self, ts):
        return public_trades_since(ts, self.product)

    def _rests(self, side, px, deadline_s):
        start = time.time()
        while time.time() - start < deadline_s:
            try:
                for price in self._trades_since(start):
                    if (side == "buy" and price < px) or (side == "sell" and price > px):
                        return True
            except Exception as e:  # one bad poll must not abort a multi-hour resting order
                log.warning("paper tape poll failed: %s", e)
            time.sleep(self.poll)
        return False

    def buy_maker(self, usd, deadline_s=7200, ref_px=None):
        bid, _ = self.quote()
        px = min(bid, ref_px) if ref_px else bid  # post-only: never at or through the ask
        usd = min(usd, self.s["paper"]["usd"], self.max_notional)
        if usd < self.min_order_usd or not self._rests("buy", px, deadline_s):
            return None
        fee = usd * self.maker
        qty = (usd - fee) / px
        self.s["paper"]["usd"] -= usd
        self.s["paper"]["btc"] += qty
        return {"price": px, "size": qty, "fee": fee, "liquidity": "maker"}

    def sell_maker(self, btc, deadline_s=7200, ref_px=None, fallback_taker=True):
        _, ask = self.quote()
        px = max(ask, ref_px) if ref_px else ask
        btc = min(btc, self.s["paper"]["btc"])
        if self._rests("sell", px, deadline_s):
            return self._settle_sell(btc, px, self.maker, "maker")
        return self.market_sell(btc) if fallback_taker else None

    def market_buy(self, usd):
        _, ask = self.quote()
        usd = min(usd, self.s["paper"]["usd"], self.max_notional)
        if usd < self.min_order_usd:
            return None
        px = ask * 1.0002
        fee = usd * self.taker
        qty = (usd - fee) / px
        self.s["paper"]["usd"] -= usd
        self.s["paper"]["btc"] += qty
        return {"price": px, "size": qty, "fee": fee, "liquidity": "taker"}

    def market_sell(self, btc):
        bid, _ = self.quote()
        return self._settle_sell(min(btc, self.s["paper"]["btc"]), bid * 0.9998, self.taker, "taker")

    # paper shorts: fills only; the bet pool's ledger books the P&L
    def short_maker(self, qty, deadline_s=1800, ref_px=None):
        _, ask = self.quote()
        px = max(ask, ref_px) if ref_px else ask
        if qty * px < self.min_order_usd or not self._rests("sell", px, deadline_s):
            return None
        return {"price": px, "size": qty, "fee": qty * px * self.maker, "liquidity": "maker"}

    def cover_maker(self, qty, deadline_s=1800, ref_px=None, fallback_taker=True):
        bid, _ = self.quote()
        px = min(bid, ref_px) if ref_px else bid
        if self._rests("buy", px, deadline_s):
            return {"price": px, "size": qty, "fee": qty * px * self.maker, "liquidity": "maker"}
        return self.market_cover(qty) if fallback_taker else None

    def market_cover(self, qty):
        _, ask = self.quote()
        px = ask * 1.0002
        return {"price": px, "size": qty, "fee": qty * px * self.taker, "liquidity": "taker"}

    def _settle_sell(self, btc, px, fee_rate, liq):
        if btc <= 0 or btc * px < self.min_order_usd:
            return None  # the exchange would refuse an order this small
        gross = btc * px
        fee = gross * fee_rate
        self.s["paper"]["btc"] -= btc
        self.s["paper"]["usd"] += gross - fee
        return {"price": px, "size": btc, "fee": fee, "liquidity": liq}


class CoinbaseBroker:
    """Real orders via the official coinbase-advanced-py SDK (responses support ['key'] access,
    returning None for absent fields; they have no .get())."""

    def __init__(self, api_key, api_secret, max_notional=110.0, requote_s=60, on_order=None, coin="BTC"):
        from coinbase.rest import RESTClient

        from .coins import COINS
        self.c = RESTClient(api_key=api_key, api_secret=api_secret, timeout=15)
        self.max_notional, self.requote_s = max_notional, requote_s
        self.on_order = on_order or (lambda **kw: None)  # persistence hook: called before/after placing
        self.coin, self.product = coin, COINS[coin]["product"]
        p = self.c.get_product(self.product)
        self.base_inc = float(p["base_increment"])
        self.quote_inc = float(p["quote_increment"])
        self.min_quote = float(p["quote_min_size"])

    # --- read-only ---------------------------------------------------------------------------
    def quote(self):
        b = self.c.get_best_bid_ask(product_ids=[self.product])["pricebooks"][0]
        return float(b["bids"][0]["price"]), float(b["asks"][0]["price"])

    def balances(self):
        usd = btc = 0.0
        for a in self.c.get_accounts(limit=250)["accounts"]:
            if a["currency"] == "USD":
                usd = float(a["available_balance"]["value"])
            elif a["currency"] == self.coin:
                btc = float(a["available_balance"]["value"])
        return usd, btc

    def total_usd(self):
        total = 0.0
        for a in self.c.get_accounts(limit=250)["accounts"]:
            if a["currency"] == "USD":
                total = float(a["available_balance"]["value"]) + float((a["hold"] or {}).get("value", 0) if isinstance(a["hold"], dict) else 0)
        return total

    def fees(self):
        t = self.c.get_transaction_summary(product_type="SPOT")["fee_tier"]
        return float(t["maker_fee_rate"]), float(t["taker_fee_rate"])

    def product_ok(self):
        p = self.c.get_product(self.product)
        return not (p["trading_disabled"] or p["cancel_only"] or p["is_disabled"] or p["limit_only"]
                    or (p["status"] or "online").lower() != "online")

    def cancel_open_orders(self):
        """Startup reconciliation: cancel any resting BTC-USD orders (e.g. left by a crash)."""
        orders = self.c.list_orders(product_ids=[self.product], order_status=["OPEN"])["orders"] or []
        ids = [o["order_id"] for o in orders]
        if ids:
            log.warning("cancelling %d open order(s) found at startup: %s", len(ids), ids)
            self.c.cancel_orders(order_ids=ids)
        return ids

    def deposit(self, usd):
        raise NotImplementedError("Coinbase Advanced Trade API cannot pull bank deposits; deposit in the Coinbase app")

    def withdraw(self, usd):
        raise NotImplementedError("bank withdrawals are not supported via the Advanced Trade API; use the Coinbase app")

    # --- orders ------------------------------------------------------------------------------
    def _settle(self, oid, timeout=120):
        """Cancel if still resting (retrying) until the order is terminal; None if it never is."""
        t0, delay = time.time(), 2
        while time.time() - t0 < timeout:
            try:
                o = self.c.get_order(oid)["order"]
                if o["status"] in TERMINAL:
                    return o
                self.c.cancel_orders(order_ids=[oid])
            except Exception as e:
                log.warning("settle %s: %s", oid, e)
            time.sleep(delay); delay = min(delay * 2, 15)
        return None

    def _work(self, side, target_size, deadline_s, ref_px):
        """Rest post-only orders at the touch, never worse than ref_px, until filled or deadline."""
        start = time.time()
        filled, notional, fees = 0.0, 0.0, 0.0
        while time.time() - start < deadline_s:
            oid = None
            try:
                bid, ask = self.quote()
                px = min(bid, ref_px) if side == "BUY" else max(ask, ref_px)
                size = floor_to(target_size - filled, self.base_inc)
                if size * px < self.min_quote:
                    break
                cid = str(uuid.uuid4())
                self.on_order(stage="placing", side=side, client_order_id=cid, size=size, price=px)
                fn = self.c.limit_order_gtc_buy if side == "BUY" else self.c.limit_order_gtc_sell
                r = fn(client_order_id=cid, product_id=self.product, base_size=f"{size:.8f}",
                       limit_price=f"{floor_to(px, self.quote_inc):.2f}", post_only=True)
                if not r["success"]:
                    log.warning("order rejected: %s", r["error_response"] or r["failure_reason"])
                    time.sleep(5)
                    continue
                oid = r["success_response"]["order_id"]
                self.on_order(stage="placed", side=side, order_id=oid, client_order_id=cid)
                waited = 0
                while waited < self.requote_s and time.time() - start < deadline_s:
                    time.sleep(5); waited += 5
                    try:
                        if self.c.get_order(oid)["order"]["status"] in TERMINAL:
                            break
                    except Exception as e:
                        log.warning("poll %s: %s", oid, e)
            except Exception as e:
                log.warning("order loop error: %s", e)
                time.sleep(5)
            if oid is None:
                continue
            o = self._settle(oid)  # always leaves no resting order behind
            if o is None:
                self.on_order(stage="UNSETTLED", order_id=oid)
                break  # book what we have; startup reconciliation cancels it
            self.on_order(stage="done", order_id=oid, status=o["status"])
            fsz = float(o["filled_size"] or 0)
            if fsz > 0:
                filled += fsz
                notional += fsz * float(o["average_filled_price"])
                fees += float(o["total_fees"] or 0)
            if o["status"] == "FILLED":
                break
        if filled == 0:
            return None
        return {"price": notional / filled, "size": filled, "fee": fees, "liquidity": "maker"}

    def buy_maker(self, usd, deadline_s=7200, ref_px=None):
        maker, _ = self.fees()
        ref_px = ref_px or self.quote()[0]
        usd = min(usd, self.max_notional)  # hard cap on entry exposure
        return self._work("BUY", usd / (ref_px * (1 + maker)), deadline_s, ref_px)

    def sell_maker(self, btc, deadline_s=7200, ref_px=None, fallback_taker=True):
        ref_px = ref_px or self.quote()[1]
        btc = min(btc, self.balances()[1])
        res = self._work("SELL", btc, deadline_s, ref_px)
        rem = btc - (res["size"] if res else 0.0)
        try:
            if fallback_taker and rem * self.quote()[0] >= self.min_quote:
                log.warning("maker exit incomplete; crossing the spread for the remaining %.8f BTC", rem)
                res = _merge(res, self.market_sell(rem))
        except Exception:
            if res is None:
                raise
            log.exception("taker fallback failed; returning the maker part that did fill")
        return res

    def market_buy(self, usd):
        usd = floor_to(min(usd, self.max_notional), self.quote_inc)
        if usd < self.min_quote:
            return None
        cid = str(uuid.uuid4())
        self.on_order(stage="placing", side="MARKET_BUY", client_order_id=cid, quote_size=usd)
        r = self.c.market_order_buy(client_order_id=cid, product_id=self.product, quote_size=f"{usd:.2f}")
        if not r["success"]:
            raise RuntimeError(f"market buy failed: {r['error_response'] or r['failure_reason']}")
        return self._await_market(r["success_response"]["order_id"])

    def market_sell(self, btc):
        """Taker sell of at most `btc` (the bot's own position) — used by stops and the kill switch."""
        _, avail = self.balances()
        size = floor_to(min(btc, avail), self.base_inc)
        if size * self.quote()[0] < self.min_quote:
            return None
        cid = str(uuid.uuid4())
        self.on_order(stage="placing", side="MARKET_SELL", client_order_id=cid, size=size)
        r = self.c.market_order_sell(client_order_id=cid, product_id=self.product, base_size=f"{size:.8f}")
        if not r["success"]:
            raise RuntimeError(f"market sell failed: {r['error_response'] or r['failure_reason']}")
        return self._await_market(r["success_response"]["order_id"])

    def _await_market(self, oid):
        for _ in range(30):
            o = self.c.get_order(oid)["order"]
            if o["status"] in TERMINAL and float(o["filled_size"] or 0) > 0:
                return {"price": float(o["average_filled_price"]), "size": float(o["filled_size"]),
                        "fee": float(o["total_fees"] or 0), "liquidity": "taker"}
            if o["status"] in TERMINAL:
                return None
            time.sleep(1)
        raise RuntimeError(f"market order {oid} did not complete in 30s")
