"""Kraken Pro spot adapters with the same interface as broker.CoinbaseBroker / PaperBroker.

Live orders use Kraken's REST API directly (no SDK): post-only limit orders (oflags=post) at the
decision price; fees are always taken in USD (oflags=fciq; sells otherwise default to BTC fees),
never chasing past the decision price; taker market orders only for stops / the kill switch /
explicit manual commands.  Every order carries a unique `userref` so an order whose AddOrder
response was lost can still be found and cancelled.

API key permissions needed: Query Funds, Query Open Orders & Trades, Create & Modify Orders,
Cancel/Close Orders.  Withdraw Funds only if you explicitly enable bank withdrawals
(BTCBOT_ALLOW_WITHDRAW=1 and KRAKEN_WITHDRAW_KEY = the name of a bank account you verified in
Kraken's UI); Kraken only ever withdraws to pre-registered destinations.
"""
import base64
import hashlib
import hmac
import logging
import math
import threading
import time
import urllib.parse

import requests

from .broker import PaperBroker, _merge
from .coins import COINS

API = "https://api.kraken.com"
PRIVATE_GAP_S = 3.2  # every coin shares one API key: stay under the Starter tier's 0.33/s counter decay
_private_lock = threading.Lock()
_private_state = {"nonce": 0, "last_call": 0.0}
_ref_lock = threading.Lock()
_ref_state = {"last": 0}  # userrefs must be unique across every coin's broker, not just per broker
_fee_cache = {}
TERMINAL = ("closed", "canceled", "expired")
FATAL = ("EOrder:Insufficient funds", "EGeneral:Permission denied", "EAPI:Invalid key",
         "EAPI:Invalid signature", "EOrder:Orders limit exceeded",
         "EGeneral:Invalid arguments", "EAPI:Feature disabled", "EFunding")
BACKOFF = ("EAPI:Rate limit exceeded", "EOrder:Rate limit exceeded", "EGeneral:Temporary lockout",
           "EService:Unavailable", "EService:Busy", "EService:Market in cancel_only mode",
           "EService:Market in post_only mode")
AMBIGUOUS = ("EService:Unavailable", "EService:Busy", "EGeneral:Internal error", "EService:Deadline elapsed")
log = logging.getLogger("btcbot")


class Unrecoverable(RuntimeError):
    """An order's existence could not be determined; nothing more may be placed until a human checks."""


class KrakenError(RuntimeError):
    def __init__(self, method, errors):
        super().__init__(f"kraken {method}: {errors}")
        self.errors = errors
        text = " ".join(errors)
        self.fatal = any(text.startswith(f) or f in text for f in FATAL)
        self.backoff = any(b in text for b in BACKOFF)
        self.ambiguous = any(a in text for a in AMBIGUOUS)


def floor_to(x, inc):
    return math.floor(x / inc + 1e-9) * inc


def public(path, **params):
    r = requests.get(f"{API}/0/public/{path}", params=params, timeout=10)
    r.raise_for_status()
    d = r.json()
    if d["error"]:
        raise KrakenError(path, d["error"])
    return d["result"]


def kraken_quote(coin="BTC"):
    c = COINS[coin]
    t = public("Ticker", pair=c["kraken_pair"])[c["kraken_key"]]
    return float(t["b"][0]), float(t["a"][0])


def sign(path, data, secret):
    """API-Sign = base64(HMAC-SHA512(b64decode(secret), path + SHA256(nonce + postdata)))."""
    post = urllib.parse.urlencode(data)
    digest = hashlib.sha256((str(data["nonce"]) + post).encode()).digest()
    mac = hmac.new(base64.b64decode(secret), path.encode() + digest, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()


class KrakenPaperBroker(PaperBroker):
    """Paper trading against Kraken's live book and tape at Kraken's fees and order minimums."""

    min_order_usd = 5.5  # every pair's ordermin is roughly $5

    def __init__(self, *args, coin="BTC", **kw):
        super().__init__(*args, **kw)
        self.coin = coin

    def quote(self):
        return kraken_quote(self.coin)

    def _rests(self, side, px, deadline_s):
        """Page forward through the public tape with Kraken's `last` cursor (max 1000 per call)."""
        start = time.time()
        cursor = str(int(start * 1e9))
        while time.time() - start < deadline_s:
            try:
                c = COINS[self.coin]
                res = public("Trades", pair=c["kraken_pair"], since=cursor)
                for tr in res[c["kraken_key"]]:
                    price, t = float(tr[0]), float(tr[2])
                    if t > start and ((side == "buy" and price < px) or (side == "sell" and price > px)):
                        return True
                cursor = res["last"]
            except Exception as e:
                log.warning("paper tape poll failed: %s", e)
            time.sleep(self.poll)
        return False


class KrakenBroker:
    def __init__(self, api_key, api_secret, max_notional=110.0, requote_s=60, on_order=None,
                 withdraw_key=None, allow_withdraw=False, coin="BTC"):
        self.key, self.secret = api_key, api_secret
        self.max_notional, self.requote_s = max_notional, requote_s
        self.on_order = on_order or (lambda **kw: None)
        self.withdraw_key, self.allow_withdraw = withdraw_key, allow_withdraw
        self.coin = coin
        c = COINS[coin]
        self.pair, self.pair_key, self.asset = c["kraken_pair"], c["kraken_key"], c["kraken_asset"]
        p = public("AssetPairs", pair=self.pair)[self.pair_key]
        self.tick = float(p["tick_size"])
        self.price_decimals = max(0, -math.floor(math.log10(self.tick) + 1e-9))
        self.lot = 10 ** -int(p["lot_decimals"])
        self.ordermin = float(p["ordermin"])
        self.costmin = float(p["costmin"])
        self.min_order_usd = max(self.ordermin * kraken_quote(coin)[0], self.costmin) * 1.05

    # --- transport ---------------------------------------------------------------------------
    def _private(self, method, **data):
        path = f"/0/private/{method}"
        with _private_lock:  # one nonce sequence and one pace for every coin sharing this key
            wait = _private_state["last_call"] + PRIVATE_GAP_S - time.time()
            if wait > 0:
                time.sleep(wait)
            nonce = max(int(time.time() * 1000), _private_state["nonce"] + 1)  # strictly increasing
            _private_state["nonce"] = nonce
            data["nonce"] = nonce
            headers = {"API-Key": self.key, "API-Sign": sign(path, data, self.secret)}
            try:
                r = requests.post(API + path, data=data, headers=headers, timeout=15)
            finally:
                _private_state["last_call"] = time.time()
        r.raise_for_status()
        d = r.json()
        if d["error"]:
            err = KrakenError(method, d["error"])
            if err.backoff:  # slow every coin down, not just this one
                with _private_lock:
                    _private_state["last_call"] = time.time() + 15
            raise err
        return d["result"]

    def _new_ref(self):
        with _ref_lock:
            ref = max(int(time.time() * 10) % 2_000_000_000, _ref_state["last"] + 1)
            _ref_state["last"] = ref
        return ref

    # --- read-only ---------------------------------------------------------------------------
    def quote(self):
        return kraken_quote(self.coin)

    def balances(self):
        """Available (not on hold) USD and this coin."""
        b = self._private("BalanceEx")
        avail = lambda k: float(b[k]["balance"]) - float(b[k].get("hold_trade", 0)) if k in b else 0.0
        return avail("ZUSD"), avail(self.asset)

    def total_usd(self):
        """All USD in the account, including cash on hold for resting buys (shared by every sleeve)."""
        b = self._private("BalanceEx")
        return float(b["ZUSD"]["balance"]) if "ZUSD" in b else 0.0

    def fees(self):
        hit = _fee_cache.get(self.pair)
        if hit and time.time() - hit[0] < 600:
            return hit[1]
        tv = self._private("TradeVolume", pair=self.pair)
        maker = float(tv["fees_maker"][self.pair_key]["fee"]) / 100  # Kraken reports percent
        taker = float(tv["fees"][self.pair_key]["fee"]) / 100
        _fee_cache[self.pair] = (time.time(), (maker, taker))
        return maker, taker

    def product_ok(self):
        return public("AssetPairs", pair=self.pair)[self.pair_key].get("status", "online") == "online"

    def cancel_open_orders(self):
        opened = self._private("OpenOrders")["open"]
        ids = [txid for txid, o in opened.items() if o["descr"]["pair"] == self.pair]
        for txid in ids:
            log.warning("cancelling open Kraken order found at startup: %s", txid)
            self._cancel(txid)
        return ids

    # --- funding -----------------------------------------------------------------------------
    def deposit(self, usd):
        """Kraken's API cannot pull money from a bank; return the account's USD deposit methods."""
        methods = self._private("DepositMethods", asset="ZUSD")
        raise NotImplementedError("bank deposits must be started in Kraken (app/web). Available methods: "
                                  + ", ".join(m.get("method", "?") for m in methods))

    def withdraw(self, usd):
        if not (self.allow_withdraw and self.withdraw_key):
            raise PermissionError("bank withdrawals are disabled: set BTCBOT_ALLOW_WITHDRAW=1 and "
                                  "KRAKEN_WITHDRAW_KEY=<name of your verified bank account in Kraken>, "
                                  "and give the API key 'Withdraw Funds' permission")
        info = self._private("WithdrawInfo", asset="ZUSD", key=self.withdraw_key, amount=f"{usd:.2f}")
        if float(info.get("limit", usd)) < usd:
            raise ValueError(f"Kraken's withdrawal limit right now is ${float(info['limit']):.2f}")
        res = self._private("Withdraw", asset="ZUSD", key=self.withdraw_key, amount=f"{usd:.2f}",
                            max_fee=str(info.get("fee", "0")))
        return {"refid": res["refid"], "amount": usd, "fee": float(info.get("fee", 0)),
                "method": info.get("method")}

    # --- orders ------------------------------------------------------------------------------
    def _order(self, txid):
        return self._private("QueryOrders", txid=txid)[txid]

    def _cancel(self, txid_or_ref):
        try:
            self._private("CancelOrder", txid=str(txid_or_ref))
        except KrakenError as e:
            if "Unknown order" not in str(e):
                raise

    def _settle(self, txid, timeout=180):
        """Cancel (retrying with backoff) until QueryOrders reports a terminal status."""
        t0, delay = time.time(), 2
        while time.time() - t0 < timeout:
            try:
                o = self._order(txid)
                if o["status"] in TERMINAL:
                    return o
                self._cancel(txid)
            except Exception as e:
                log.warning("settle %s: %s", txid, e)
            time.sleep(delay); delay = min(delay * 2, 20)
        return None

    def _recover(self, ref):
        """After an ambiguous AddOrder failure: cancel anything carrying `ref`, return its txid.
        Needs two empty lookups a few seconds apart (an in-flight order can appear late)."""
        empty = 0
        for attempt in range(8):
            time.sleep(3)
            try:
                self._cancel(ref)
                found = {**self._private("OpenOrders", userref=ref)["open"],
                         **self._private("ClosedOrders", userref=ref)["closed"]}
                found = {t: o for t, o in found.items() if o["descr"]["pair"] == self.pair}
                if found:
                    return next(iter(found))
                empty += 1
                if empty >= 2:
                    return None
            except Exception as e:
                log.warning("recover userref %s attempt %d: %s", ref, attempt, e)
        raise Unrecoverable(f"could not determine whether order userref={ref} exists")

    def _work(self, side, target_size, deadline_s, ref_px, margin=False, reduce=False):
        """margin=True places 2:1 margin orders (shorts); reduce=True only closes an open margin position."""
        extra = {"leverage": "2"} if margin else {}
        if reduce:
            extra["reduce_only"] = "true"
        start = time.time()
        filled, notional, fees = 0.0, 0.0, 0.0
        delay = 5
        while time.time() - start < deadline_s:
            txid = None
            try:
                bid, ask = self.quote()
                px = min(bid, ref_px) if side == "buy" else max(ask, ref_px)
                px = floor_to(px, self.tick) if side == "buy" else math.ceil(px / self.tick - 1e-9) * self.tick
                want = target_size - filled
                if side == "sell" and not margin:
                    want = min(want, self.balances()[1])
                size = floor_to(want, self.lot)
                if size < self.ordermin or size * px < self.costmin:
                    break
                ref = self._new_ref()
                self.on_order(stage="placing", side=side, size=size, price=px, userref=ref)
                try:
                    txid = self._private("AddOrder", pair=self.pair, type=side, ordertype="limit",
                                         price=f"{px:.{self.price_decimals}f}", volume=f"{size:.8f}",
                                         oflags="post,fciq",
                                         userref=ref, **extra)["txid"][0]
                except KrakenError as e:
                    if e.fatal:
                        log.error("order refused, giving up: %s", e)
                        break
                    if e.ambiguous:
                        txid = self._recover(ref)
                    if txid is None:
                        log.warning("order rejected: %s", e)
                        time.sleep(delay if e.backoff else 5)
                        delay = min(delay * 2, 300) if e.backoff else delay
                        continue
                except requests.RequestException as e:
                    log.warning("AddOrder transport error (%s); recovering userref %s", e, ref)
                    txid = self._recover(ref)
                    if txid is None:
                        continue
                self.on_order(stage="placed", side=side, order_id=txid, userref=ref)
                waited = 0
                while waited < self.requote_s and time.time() - start < deadline_s:
                    time.sleep(15); waited += 15
                    try:
                        if self._order(txid)["status"] in TERMINAL:
                            break
                    except Exception as e:
                        log.warning("poll %s: %s", txid, e)
            except Unrecoverable as e:
                self.on_order(stage="UNSETTLED", userref=ref, error=str(e))
                break  # never stack a new order on one whose state is unknown
            except Exception as e:
                log.warning("order loop error: %s", e)
                time.sleep(5)
            if txid is None:
                continue
            o = self._settle(txid)  # never leave a resting order behind
            if o is None:
                self.on_order(stage="UNSETTLED", order_id=txid)
                break  # book the fills we have; startup reconciliation cancels it
            self.on_order(stage="done", order_id=txid, status=o["status"], reason=o.get("reason"))
            vol = float(o.get("vol_exec") or 0)
            if vol > 0:
                filled += vol
                notional += float(o["cost"])
                fees += float(o.get("fee") or 0)
            if o["status"] == "closed" and vol >= size - self.lot:
                break
        if filled == 0:
            return None
        return {"price": notional / filled, "size": filled, "fee": fees, "liquidity": "maker"}

    def buy_maker(self, usd, deadline_s=7200, ref_px=None):
        maker, _ = self.fees()
        ref_px = ref_px or self.quote()[0]
        usd = min(usd, self.max_notional)
        return self._work("buy", usd / (ref_px * (1 + maker)), deadline_s, ref_px)

    def sell_maker(self, btc, deadline_s=7200, ref_px=None, fallback_taker=True):
        ref_px = ref_px or self.quote()[1]
        res = self._work("sell", btc, deadline_s, ref_px)
        rem = btc - (res["size"] if res else 0.0)
        if fallback_taker and rem >= self.ordermin:
            log.warning("maker exit incomplete; market-selling remaining %.8f %s", rem, self.coin)
            try:
                res = _merge(res, self.market_sell(rem))
            except Exception:
                if res is None:
                    raise
                log.exception("taker fallback failed; returning the maker part that did fill")
        return res

    def _market(self, side, size, margin=False, reduce=False):
        extra = {"leverage": "2"} if margin else {}
        if reduce:
            extra["reduce_only"] = "true"
        ref = self._new_ref()
        self.on_order(stage="placing", side=f"market_{side}{'_margin' if margin else ''}", size=size, userref=ref)
        try:
            txid = self._private("AddOrder", pair=self.pair, type=side, ordertype="market",
                                 volume=f"{size:.8f}", oflags="fciq", userref=ref, **extra)["txid"][0]
        except (requests.RequestException, KrakenError) as e:
            if isinstance(e, KrakenError) and not e.ambiguous:
                raise
            txid = self._recover(ref)
            if txid is None:
                raise
        o, t0 = None, time.time()
        while time.time() - t0 < 90:
            try:
                o = self._order(txid)
            except Exception as e:
                log.warning("poll market order %s: %s", txid, e)
                time.sleep(2)
                continue
            if o["status"] in TERMINAL:  # market price protection can cancel the remainder
                vol = float(o.get("vol_exec") or 0)
                self.on_order(stage="done", order_id=txid, status=o["status"])
                if vol <= 0:
                    return None
                return {"price": float(o["cost"]) / vol, "size": vol, "fee": float(o.get("fee") or 0),
                        "liquidity": "taker"}
            time.sleep(2)
        self.on_order(stage="UNSETTLED", order_id=txid)
        raise RuntimeError(f"market order {txid} did not complete in 90s")

    def market_sell(self, btc):
        _, avail = self.balances()
        size = floor_to(min(btc, avail), self.lot)
        res = None
        for _ in range(3):  # re-issue the remainder if price protection cancelled part of it
            if size < self.ordermin:
                break
            try:
                part = self._market("sell", size)
            except Exception:
                if res is None:
                    raise
                log.exception("market sell remainder failed; returning what filled")
                break
            if not part:
                break
            res = _merge(res, part)
            size = floor_to(size - part["size"], self.lot)
        return res

    def market_buy(self, usd):
        _, taker = self.fees()
        _, ask = self.quote()
        size = floor_to(min(usd, self.max_notional) / (ask * 1.003 * (1 + taker)), self.lot)
        if size < self.ordermin:
            return None
        return self._market("buy", size)

    # --- margin (short bets) -------------------------------------------------------------------
    def short_maker(self, qty, deadline_s=1800, ref_px=None):
        """Open a short: post-only 2:1 margin sell (collateral is the account's USD)."""
        return self._work("sell", qty, deadline_s, ref_px or self.quote()[1], margin=True)

    def cover_maker(self, qty, deadline_s=1800, ref_px=None, fallback_taker=True):
        """Close a short: post-only reduce-only buy, then a market buy for whatever is left."""
        res = self._work("buy", qty, deadline_s, ref_px or self.quote()[0], margin=True, reduce=True)
        rem = floor_to(qty - (res["size"] if res else 0.0), self.lot)
        if fallback_taker and rem >= self.ordermin:
            try:
                res = _merge(res, self.market_cover(rem))
            except Exception:
                if res is None:
                    raise
                log.exception("market cover failed; returning the part that did fill")
        return res

    def market_cover(self, qty):
        size = floor_to(qty, self.lot)
        return self._market("buy", size, margin=True, reduce=True) if size >= self.ordermin else None
