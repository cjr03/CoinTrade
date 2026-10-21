"""Hourly decision loop: refresh candles -> features -> model -> policy -> broker.

    python -m btcbot.trader                 # headless paper trading (default, no keys needed)
    python -m btcbot.app                    # same loop + the terminal UI at http://127.0.0.1:8787
    python -m btcbot.trader --once          # single decision, then exit
    BTCBOT_EXCHANGE=kraken ...              # trade on Kraken Pro instead of Coinbase (default)
    BTCBOT_POLICY=policy_momentum.json ...  # pick a policy file in artifacts/ (default policy.json)
    BTCBOT_MODE=live BTCBOT_CONFIRM_LIVE=I_UNDERSTAND ...   # real money

Mirrors the backtested policy (artifacts/policy*.json, written by `btcbot.evaluate --export`):
long/flat with 0/50/100% sizing, post-only maker orders at the decision price, signal exit after
min-hold, sigma stop, time stop, and a drawdown kill switch that halts trading until a human
resumes it (the high-water mark is re-based on resume).

The bot only trades its own capital ledger (`state.ledger`), never the whole exchange account:
deposits/withdrawals move money in or out of that ledger.
"""
import argparse
import json
import logging
import math
import os
import pickle
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .broker import CoinbaseBroker, PaperBroker
from .data import download, load
from .features import build_features
from .coins import COINS
from .kraken import KrakenBroker, KrakenPaperBroker

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
STATE = ROOT / "state"
LATE_MINUTES = 15  # a bar is only traded within this many minutes of its close (as backtested)
BET_MAX_SHARE = float(os.environ.get("BTCBOT_BET_MAX_SHARE", 0.25))  # lifetime cap on cash given to the bet pool
log = logging.getLogger("btcbot")


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def now_iso():
    return datetime.now(timezone.utc).isoformat()


class Busy(RuntimeError):
    pass


class Trader:
    def __init__(self, mode, exchange="coinbase", policy_file="policy.json", coin="BTC", sink=None):
        tag = "" if policy_file == "policy.json" else "_" + Path(policy_file).stem
        self.live = mode == "live"
        self.exchange = exchange
        self.coin = coin
        # BTC keeps the original file names so existing state carries over; other coins get a suffix
        self.mode = f"{exchange}_{mode}{tag}" + ("" if coin == "BTC" else f"_{coin}")
        STATE.mkdir(exist_ok=True)
        self.state_path = STATE / f"{self.mode}.json"
        self.s = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self._migrate()
        self.sink = sink  # Portfolio callback that merges every coin's events for the UI
        self.policy = json.loads((ART / policy_file).read_text())
        self.bundle = None
        if self.policy["model"] != "momentum":
            if coin != "BTC":
                raise SystemExit(f"the ML policy was trained on BTC only; use the trend policy for {coin}")
            # model.pkl is produced locally by `btcbot.train --final`; never load one from elsewhere
            with open(ART / "model.pkl", "rb") as fh:
                self.bundle = pickle.load(fh)
            if self.policy["horizon"] != self.bundle["horizon"] or self.policy["model"] != self.bundle["model_type"]:
                raise SystemExit("policy.json and model.pkl were built for different model/horizon")
        if self.policy.get("exchange", "coinbase") != exchange:
            raise SystemExit(f"policy was backtested for {self.policy.get('exchange')} fees, not {exchange}")
        self.lock = threading.RLock()
        self.events = deque(maxlen=400)
        self.event_seq = 0
        self.busy = None
        capital = float(os.environ.get("BTCBOT_CAPITAL", 100))
        cap = float(os.environ.get("BTCBOT_MAX_NOTIONAL", 110))
        hook = self._order_event
        if coin != "BTC":
            capital = 0.0  # new coins start unfunded: 'deposit 20 eth' or 'move 20 btc eth' funds them
        if self.live and exchange == "kraken":
            self.broker = KrakenBroker(os.environ["KRAKEN_API_KEY"], os.environ["KRAKEN_API_SECRET"],
                                       max_notional=cap, on_order=hook, coin=coin,
                                       withdraw_key=os.environ.get("KRAKEN_WITHDRAW_KEY"),
                                       allow_withdraw=os.environ.get("BTCBOT_ALLOW_WITHDRAW") == "1")
        elif self.live:
            self.broker = CoinbaseBroker(os.environ["COINBASE_API_KEY"], os.environ["COINBASE_API_SECRET"],
                                         max_notional=cap, on_order=hook, coin=coin)
        else:
            paper = KrakenPaperBroker if exchange == "kraken" else PaperBroker
            self.broker = paper(self.s, *self.policy["fees_assumed"], capital=capital, max_notional=cap, coin=coin)
        if "ledger" not in self.s:
            usd = capital
            if self.live and usd > 0:
                usd = min(capital, self.broker.balances()[0])
            self.s["ledger"] = {"usd": usd, "qty": 0.0, "deposited": usd, "withdrawn": 0.0, "fees": 0.0}
        if "auto" not in self.s:
            self.s["auto"] = self._auto_default()

    def _auto_default(self):
        """Automatic trading starts on only where the backtest favoured the trend rule over holding."""
        path = ART / "coins_report.json"
        if self.coin == "BTC" or not path.exists():
            return True
        r = json.loads(path.read_text()).get(self.coin)
        return bool(r and r["trend"]["sharpe"] >= r["hold"]["sharpe"])

    def _migrate(self):
        """State written before multi-coin support called the coin balance 'btc'."""
        L = self.s.get("ledger")
        if L is not None and "btc" in L and "qty" not in L:
            L["qty"] = L.pop("btc")
        pos = self.s.get("position")
        if pos is not None and "btc" in pos and "qty" not in pos:
            pos["qty"] = pos.pop("btc")

    def _order_event(self, **kw):
        if kw.get("stage") == "UNSETTLED":
            self.s["needs_review"] = (f"order {kw.get('order_id') or kw.get('userref')} could not be confirmed "
                                      "closed; its fills may be missing from the ledger")
        self.journal(event="order", **kw)

    # --- persistence / journal ---------------------------------------------------------------
    def save(self):
        with self.lock:
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.s, indent=1, default=str))
            tmp.replace(self.state_path)

    def journal(self, **rec):
        rec["ts"] = now_iso()
        rec.setdefault("coin", self.coin)
        with open(STATE / f"{self.mode}_journal.jsonl", "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        clean = json.loads(json.dumps(rec, default=str))
        if self.sink:
            self.sink(clean)
        else:
            self.event_seq += 1
            self.events.append({"seq": self.event_seq, **clean})
        log.info("%s", rec)

    # --- ledger -------------------------------------------------------------------------------
    @property
    def ledger(self):
        return self.s["ledger"]

    def book_buy(self, fill, frac=None, sig=None, manual=False):
        L = self.ledger
        cost = fill["size"] * fill["price"] + fill["fee"]
        L["usd"] -= cost; L["qty"] += fill["size"]; L["fees"] += fill["fee"]
        pos = self.s.get("position")
        sigma_h = (sig or self.s.get("signal") or {}).get("sigma_h") or 0.05
        if pos:
            total = pos["qty"] + fill["size"]
            pos["entry_px"] = (pos["entry_px"] * pos["qty"] + fill["price"] * fill["size"]) / total
            pos["qty"] = total
            pos["frac"] = max(pos["frac"], frac or pos["frac"])
        else:
            self.s["position"] = pos = {"entry_time": now_iso(), "entry_px": fill["price"], "qty": fill["size"],
                                        "frac": frac or 1.0, "manual": manual}
        pos["stop_px"] = pos["entry_px"] * float(np.exp(-self.policy["stop_sigma"] * sigma_h))

    def book_sell(self, fill):
        L = self.ledger
        L["qty"] = max(L["qty"] - fill["size"], 0.0)
        L["usd"] += fill["size"] * fill["price"] - fill["fee"]; L["fees"] += fill["fee"]
        pos = self.s.get("position")
        if pos:
            pos["qty"] = max(pos["qty"] - fill["size"], 0.0)
            if pos["qty"] * fill["price"] < max(1.0, getattr(self.broker, "min_order_usd", 1.0)):
                self.s.pop("position")  # remaining dust stays in the ledger and in equity

    def equity(self, bid):
        return self.ledger["usd"] + self.ledger["qty"] * bid

    def _rescale_hwm(self, delta_usd):
        """Cash in/out changes equity but is not P&L: scale the high-water mark by the same ratio so the
        drawdown percentage the kill switch sees is unchanged. Call BEFORE the ledger changes."""
        px = (self.s.get("signal") or {}).get("close") or (self.s.get("position") or {}).get("entry_px") or 0.0
        eq = self.equity(px)
        if "hwm" not in self.s or eq <= 0:
            self.s["hwm_reset_pending"] = True  # no meaningful baseline yet: restart it at the next tick
            return
        self.s["hwm"] = max(self.s["hwm"] * (eq + delta_usd) / eq, 0.0)

    # --- signal -------------------------------------------------------------------------------
    def predict(self):
        """Refresh caches and predict from the bar that has just CLOSED."""
        if self.policy["model"] == "momentum":
            return self._predict_trend()
        for prod, gran, days in (("BTC-USD", "ONE_HOUR", 800), ("ETH-USD", "ONE_HOUR", 800),
                                 ("BTC-USD", "FIVE_MINUTE", 10)):
            download(prod, gran, days)
        now = pd.Timestamp.now(tz="UTC")
        last_closed = now.floor("h") - pd.Timedelta(hours=1)  # start time of the just-closed bar
        h1 = load("BTC-USD", "ONE_HOUR").loc[:last_closed]
        eth = load("ETH-USD", "ONE_HOUR").loc[:last_closed]
        m5 = load("BTC-USD", "FIVE_MINUTE").loc[:last_closed + pd.Timedelta(minutes=55)]
        if h1.empty or h1.index.max() != last_closed:
            raise RuntimeError(f"stale BTC 1h data: last bar {h1.index.max()}, expected {last_closed}")
        if eth.empty or eth.index.max() != last_closed:
            raise RuntimeError(f"stale ETH 1h data: last bar {eth.index.max()}, expected {last_closed}")
        if ((m5.index >= last_closed) & (m5.index < now.floor("h"))).sum() < 10:
            raise RuntimeError(f"stale 5m data for hour {last_closed}")
        h1 = h1.iloc[-24 * 900:]  # > 2y needed by the longest rolling windows (vol_pctile)
        f, bars = build_features(h1, m5.loc[h1.index.min():], eth.loc[h1.index.min():])
        row = f.iloc[[-1]]
        assert row.index[0] == last_closed
        base = {"bar": str(row.index[0]), "vol_pctile": float(row["vol_pctile"].iloc[0]),
                "close": float(bars["close"].iloc[-1]), "computed_at": now_iso()}
        X = row[self.bundle["columns"]]
        H = self.bundle["horizon"]
        scale = float(row["sigma"].iloc[0]) * np.sqrt(H)
        mt = self.bundle["model_type"]
        if mt == "ridge":
            lo, hi = self.bundle["clip"]
            pred = float(self.bundle["model"].predict(X.clip(lo, hi, axis=1))[0]) * scale
        elif mt == "lgbm":
            pred = float(np.mean([m.predict(X)[0] for m in self.bundle["models"]])) * scale
        else:
            p = float(np.mean([m.predict_proba(X)[0, 1] for m in self.bundle["models"]]))
            pred = float(self.bundle["iso"].predict([p])[0])
        return {**base, "pred": pred, "sigma_h": scale, "horizon": H}

    def _predict_trend(self):
        """Long while the last COMPLETED daily close is above its N-day SMA (same as the backtest).
        Needs only this coin's own hourly candles."""
        product = COINS[self.coin]["product"]
        download(product, "ONE_HOUR", 800)
        now = pd.Timestamp.now(tz="UTC")
        last_closed = now.floor("h") - pd.Timedelta(hours=1)
        h1 = load(product, "ONE_HOUR").loc[:last_closed]
        if h1.empty or h1.index.max() != last_closed:
            raise RuntimeError(f"stale {self.coin} 1h data: last bar {h1.index.max() if len(h1) else None}, "
                               f"expected {last_closed}")
        f, bars = build_features(h1.iloc[-24 * 900:])
        row = f.iloc[[-1]]
        daily = bars["close"].resample("D").last()
        n = self.policy["sma_days"]
        sma = daily.rolling(n).mean()
        above = (daily > sma).shift(1).reindex(bars.index, method="ffill").fillna(False)
        return {"bar": str(row.index[0]), "vol_pctile": float(row["vol_pctile"].fillna(0.5).iloc[0]),
                "close": float(bars["close"].iloc[-1]), "computed_at": now_iso(),
                "pred": 1.0 if bool(above.iloc[-1]) else -1.0, "sma_days": n,
                "sma": float(sma.iloc[-2]), "daily_close": float(daily.iloc[-2]),
                "sigma_h": float(row["sigma"].iloc[0]) * np.sqrt(24), "horizon": 24}

    def _full_size(self, sig):
        pol = self.policy
        return sig["pred"] >= 2 * max(pol["entry_thr"], pol["round_trip"]) and sig["vol_pctile"] < 0.8

    # --- the hourly decision ------------------------------------------------------------------
    def step(self):
        with self.lock:
            self.busy = "deciding"
            try:
                self._step()
            finally:
                self.busy = None
                self.save()

    def _step(self):
        pol = self.policy
        sig = self.predict()
        self.s["signal"] = sig
        late = (pd.Timestamp.now(tz="UTC") - (pd.Timestamp(sig["bar"]) + pd.Timedelta(hours=1))).total_seconds() / 60
        if self.s.get("last_bar") == sig["bar"] or late > LATE_MINUTES:
            self.journal(event="signal", note="observing only (bar already decided or too late to trade)", **sig)
            return
        self.s["last_bar"] = sig["bar"]
        if self.s.get("halted"):
            self.journal(event="halted", note=f"kill switch fired at {self.s['halted']}; type 'resume' to re-arm")
            return
        if self.s.get("needs_review"):
            self.journal(event="halted", note=f"needs review: {self.s['needs_review']}; type 'resume' when checked")
            return
        bid, ask = self.broker.quote()
        acct_usd, acct_qty = self.broker.balances()
        L = self.ledger
        if L["qty"] > acct_qty + 1e-8:
            # the exchange holds less coin than we booked: a sale we never recorded? never guess — stop
            self.s["needs_review"] = (f"ledger shows {L['qty']:.8f} {self.coin} but the exchange has {acct_qty:.8f}; "
                                      "check recent fills, then type 'resume'")
            self.journal(event="halted", note=self.s["needs_review"])
            return
        equity = self.equity(bid)
        if self.s.pop("hwm_reset_pending", False):
            self.s["hwm"] = equity
        self.s["hwm"] = max(self.s.get("hwm", equity), equity)
        self.journal(event="tick", **sig, bid=bid, equity=equity, hwm=self.s["hwm"], ledger=dict(L),
                     position=self.s.get("position"))

        pos = self.s.get("position")
        if pos and L["qty"] * bid < 1.0:
            self.s.pop("position"); pos = None
        # the kill switch protects every position, including manual ones and while the bot is off
        if pos and equity < self.s["hwm"] * (1 - pol["kill_dd"]):
            self.s["halted"] = now_iso()
            self.s["hwm_reset_pending"] = True
            fill = self._sell(lambda: self.broker.market_sell(L["qty"]), "kill")
            self.s.pop("position", None)
            self.journal(event="KILL_SWITCH", fill=fill, equity=equity, hwm=self.s["hwm"])
            return
        if pos and bid < pos["stop_px"]:
            fill = self._sell(lambda: self.broker.market_sell(L["qty"]), "stop")
            self.journal(event="exit", reason="stop", fill=fill, pnl_pct=fill["price"] / pos["entry_px"] - 1 if fill else None)
            return
        if not self.s.get("auto", True):
            return
        # fee/product checks only gate new orders; a failure here must never skip the protection above
        try:
            tradable = self.broker.product_ok()
            maker, taker = self.broker.fees()
            fee_ok = maker <= pol["fees_assumed"][0] + 1e-9
        except Exception as e:
            self.journal(event="skip", note=f"could not read {self.coin} fees/status ({e}); no new orders this hour")
            return
        if not tradable:
            self.journal(event="skip", note=f"{self.coin}/USD is not tradable on the exchange right now")
            return
        if not fee_ok:
            self.journal(event="skip", note=f"live maker fee {maker:.2%} above backtested {pol['fees_assumed'][0]:.2%}: no new entries")

        entry_thr = pol["entry_thr"]
        gated_out = not pol.get("gate_pass", False) and pol["model"] != "momentum" and self.live \
            and os.environ.get("BTCBOT_ALLOW_UNGATED") != "1"
        can_enter = fee_ok and entry_thr is not None and not gated_out
        if pos and pos.get("manual"):
            return  # your manual position: only the stop and the kill switch act on it
        if pos:
            held_h = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(pos["entry_time"])).total_seconds() / 3600
            reason = None
            if held_h >= pol["time_stop_mult"] * pol["horizon"]:
                reason = "time"
            elif held_h >= pol["min_hold"] and not (sig["pred"] >= pol["exit_thr"]):  # NaN exits too
                reason = "signal"
            if reason:
                self.busy = f"exiting ({reason})"
                entry = pos["entry_px"]
                fill = self._sell(lambda: self.broker.sell_maker(L["qty"], deadline_s=pol["exit_fill_hours"] * 3600,
                                                                 ref_px=sig["close"]), reason)
                if fill:
                    self.journal(event="exit", reason=reason, fill=fill, pnl_pct=fill["price"] / entry - 1)
                else:
                    self.journal(event="exit_unfilled", reason=reason)
            elif pol["half_size"] and pos["frac"] < 1 and can_enter and self._full_size(sig):
                self.busy = "scaling up"
                self._buy(min(equity - L["qty"] * bid, self._room(acct_usd)), sig, frac=1.0, event="scale_up")
        elif can_enter and sig["pred"] > entry_thr:
            frac = 1.0 if (not pol["half_size"] or self._full_size(sig)) else 0.5
            self.busy = f"entering ({frac:.0%})"
            self._buy(min(equity * frac, self._room(acct_usd)), sig, frac=frac, event="entry")
        elif gated_out and sig["pred"] > (entry_thr or float("inf")):
            self.journal(event="skip", note="signal says buy, but this ML policy failed its go-live gate: watching only")

    def _sell(self, order, reason):
        """Run a sell and always book whatever filled; an exception mid-order halts for review."""
        self.s["exiting"] = reason; self.save()
        fill = None
        try:
            fill = order()
        except Exception as e:
            self.s["needs_review"] = f"{reason} exit failed mid-order ({e}); check exchange fills vs ledger"
            raise
        finally:
            if fill:
                self.book_sell(fill)
            self.s.pop("exiting", None)
            self.save()
        return fill

    def _room(self, acct_usd):
        """Cash this sleeve may spend now. Live: its own ledger cash, never more than the account's total
        USD minus what the other sleeves have booked (total, so another coin's resting buy isn't counted
        twice), and never more than is actually available. Paper: each sleeve has its own balance."""
        L = self.ledger
        if not self.live:
            return min(L["usd"], acct_usd)
        return min(L["usd"], self.broker.total_usd() - self.other_sleeves_usd(), acct_usd)

    def _buy(self, usd, sig, frac, event):
        if usd < getattr(self.broker, "min_order_usd", 1.0):
            self.journal(event="skip", note=f"{self.coin} sleeve has ${max(usd, 0):.2f} to spend, below the "
                                            "exchange minimum; fund it with 'move' or 'deposit'")
            return
        self.s["entering"] = sig["bar"]; self.save()
        fill = None
        try:
            fill = self.broker.buy_maker(usd, deadline_s=self.policy["entry_fill_hours"] * 3600,
                                         ref_px=sig["close"])
        except Exception as e:
            # an exception after a partial fill could leave coins we don't know about: stop and ask
            self.s["needs_review"] = f"{event} failed mid-order ({e}); check exchange fills vs ledger"
            raise
        finally:
            if fill:
                self.book_buy(fill, frac=frac, sig=sig)
                self.journal(event=event, frac=frac, fill=fill, pred=sig["pred"])
            self.s.pop("entering", None)
        if not fill:
            self.journal(event=f"{event}_unfilled", pred=sig["pred"])

    # --- manual commands (from the UI) --------------------------------------------------------
    @contextmanager
    def _manual(self):
        """Manual commands never queue behind a working order: fail fast if the bot holds the lock."""
        if not self.lock.acquire(timeout=0.5):
            raise Busy(f"the bot is busy ({self.busy or 'working'}); try again when it finishes")
        try:
            yield
        finally:
            self.lock.release()

    @staticmethod
    def _amount(x, what):
        if not (isinstance(x, (int, float)) and math.isfinite(x) and 0 < x < 1e7):
            raise ValueError(f"{what} must be a positive amount")
        return float(x)

    other_sleeves_usd = staticmethod(lambda: 0.0)  # set by Portfolio: cash booked to the other coins

    def _spendable(self):
        """Sleeve cash, never more than the exchange holds after the other coins' cash is set aside."""
        return self._room(self.broker.balances()[0])

    def manual_buy(self, usd):
        usd = self._amount(usd, "buy amount")
        with self._manual():
            usd = min(usd, self._spendable())
            if usd < getattr(self.broker, "min_order_usd", 1.0):
                raise ValueError(f"bot cash is ${self.ledger['usd']:.2f}; order below the exchange minimum")
            fill = self.broker.market_buy(usd)
            if not fill:
                raise RuntimeError("order did not fill")
            self.book_buy(fill, frac=1.0, manual=not self.s.get("position"))
            self.journal(event="manual_buy", fill=fill)
            self.save()
            return fill

    def manual_sell(self, qty):
        qty = self._amount(qty, "sell amount")
        with self._manual():
            qty = min(qty, self.ledger["qty"])
            fill = self._sell(lambda: self.broker.market_sell(qty), "manual")
            if not fill:
                raise RuntimeError("order did not fill (below the exchange minimum?)")
            self.journal(event="manual_sell", fill=fill)
            self.save()
            return fill

    def deposit(self, usd):
        usd = round(self._amount(usd, "deposit"), 2)
        with self._manual():
            L = self.ledger
            if self.live:
                if self.broker.total_usd() - L["usd"] - self.other_sleeves_usd() + 1e-9 < usd:
                    self.broker.deposit(usd)  # raises with exchange-specific instructions
                note = "allocated existing exchange USD to the bot"
            else:
                self.broker.deposit(usd)
                note = "paper deposit"
            self._rescale_hwm(+usd)
            L["usd"] += usd; L["deposited"] += usd
            self.journal(event="deposit", amount=usd, note=note)
            self.save()
            return note

    def withdraw(self, usd):
        usd = math.floor(self._amount(usd, "withdrawal") * 100) / 100
        with self._manual():
            L = self.ledger
            if usd < 0.01 or usd > self._spendable() + 1e-9:
                raise ValueError(f"the bot can withdraw up to ${self._spendable():.2f}; sell first or withdraw less")
            if self.s.get("withdraw_pending"):
                raise RuntimeError(f"an earlier withdrawal is unconfirmed ({self.s['withdraw_pending']}); "
                                   "check the exchange, then type 'resume'")
            self.s["withdraw_pending"] = f"${usd:.2f} at {now_iso()}"; self.save()
            try:
                res = self.broker.withdraw(usd)
            except (PermissionError, NotImplementedError, ValueError):
                self.s.pop("withdraw_pending")  # refused before any money moved
                raise
            except Exception as e:  # transport error: the money may or may not have moved
                self.s["needs_review"] = f"withdrawal of ${usd:.2f} may have gone through ({e}); check the exchange"
                self.save()
                raise RuntimeError(self.s["needs_review"])
            # book first, with no further network calls, so a later failure can't lose the record
            self._rescale_hwm(-usd)
            L["usd"] -= usd; L["withdrawn"] += usd
            self.s.pop("withdraw_pending")
            self.journal(event="withdraw", amount=usd, result=res)
            self.save()
            return res

    def set_auto(self, on):
        with self._manual():
            self.s["auto"] = on
            self.journal(event="auto", on=on)
            self.save()

    def resume(self):
        """Clear a halt after the human has looked. Re-syncs the ledger down to what the exchange holds."""
        with self._manual():
            cleared = [k for k in ("halted", "needs_review", "withdraw_pending") if self.s.pop(k, None)]
            if cleared:
                self.s["hwm_reset_pending"] = True
                acct_usd, acct_qty = self.broker.balances()
                L = self.ledger
                if self.live:
                    L["usd"] = min(L["usd"], acct_usd)
                if L["qty"] > acct_qty:
                    L["qty"] = acct_qty
                    if self.s.get("position"):
                        self.s["position"]["qty"] = acct_qty
            self.journal(event="resume", cleared=cleared, ledger=dict(self.ledger))
            self.save()
            return cleared

    # --- snapshot for the UI -----------------------------------------------------------------
    def status(self):
        got = self.lock.acquire(timeout=0.2)
        try:
            s = json.loads(json.dumps(self.s, default=str))
        finally:
            if got:
                self.lock.release()
        return {"mode": "live" if self.live else "paper", "exchange": self.exchange, "busy": self.busy,
                "policy": self.policy, "state": s}

    # --- lifecycle ----------------------------------------------------------------------------
    def reconcile(self):
        """On startup: no resting orders may survive a restart; flag an interrupted entry."""
        with self.lock:
            cancelled = self.broker.cancel_open_orders()
            usd, qty = self.broker.balances()
            if self.s.get("entering"):
                self.s["needs_review"] = (f"previous run stopped while entering (bar {self.s.pop('entering')}); "
                                          f"exchange shows {qty:.8f} {self.coin} vs ledger {self.ledger['qty']:.8f}")
            if self.s.pop("exiting", None) or self.ledger["qty"] > qty + 1e-8:
                if self.ledger["qty"] > qty + 1e-8:
                    self.s["needs_review"] = (f"ledger shows {self.ledger['qty']:.8f} {self.coin} but the exchange "
                                              f"has {qty:.8f}; check fills, then type 'resume'")
            self.journal(event="startup", mode=self.mode, cancelled=cancelled, account_usd=usd, account_qty=qty,
                         ledger=self.ledger, position=self.s.get("position"))
            self.save()

    def run(self, once=False, stop=None, reconcile=True):
        if reconcile:
            self.reconcile()
        while not (stop and stop.is_set()):
            try:
                self.step()
            except Exception as e:
                log.exception("step failed; will retry next hour")
                self.journal(event="error", note=str(e))
            if once:
                return
            nxt = (time.time() // 3600 + 1) * 3600 + 20  # 20s after the hour so the bar is final
            while time.time() < nxt and not (stop and stop.is_set()):
                time.sleep(min(5, max(nxt - time.time(), 0.1)))


class Portfolio:
    """One Trader sleeve per coin, each with its own ledger, position and kill switch, plus an optional
    short-term bet pool. All share the exchange account (and API key), so cash booked to one is
    reserved from the others."""

    POOL = "BETS"

    def __init__(self, mode, exchange, policy_file, coins, bets=False):
        self.events = deque(maxlen=800)
        self.seq = 0
        self._ev_lock = threading.Lock()
        self.traders = {}
        self.pool = None
        for c in coins:
            t = Trader(mode, exchange, policy_file, coin=c, sink=self._sink)
            t.other_sleeves_usd = (lambda me=c: sum(o.ledger["usd"] for k, o in self.traders.items() if k != me)
                                   + (self.pool.ledger["usd"] if self.pool else 0.0))
            self.traders[c] = t
        first = next(iter(self.traders.values()))
        self.live, self.exchange, self.policy = first.live, exchange, first.policy
        self._fund_lock = threading.Lock()  # one bet at a time may sell coin for its stake
        if bets:
            from .bets import BetPool
            self.pool = BetPool(mode, exchange, coins, STATE, sink=self._sink,
                                reserved_elsewhere=lambda: sum(t.ledger["usd"] for t in self.traders.values()))
            self.pool.funder = self.fund_bet
        self._replay_trades()

    REPLAY = ("entry", "exit", "scale_up", "manual_buy", "manual_sell", "sell_for_bet", "KILL_SWITCH", "bet_open",
              "bet_close")

    def _replay_trades(self, keep=60):
        """Seed the UI's event feed with recent fills from the journals, so a restart doesn't blank them."""
        paths = [STATE / f"{t.mode}_journal.jsonl" for t in self.traders.values()]
        if self.pool:
            paths.append(self.pool.journal_path)
        recs = []
        for path in paths:
            if path.exists():
                for line in path.read_text().splitlines()[-2000:]:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("event") in self.REPLAY and r.get("fill"):
                        r.setdefault("coin", "BTC")
                        r["replayed"] = True
                        recs.append(r)
        for r in sorted(recs, key=lambda r: r.get("ts", ""))[-keep:]:
            self._sink(r)

    def _sink(self, rec):
        with self._ev_lock:
            self.seq += 1
            self.events.append({"seq": self.seq, **rec})

    def __getitem__(self, coin):
        return self.traders[coin]

    @property
    def coins(self):
        return list(self.traders)

    def status(self):
        return {"mode": "live" if self.live else "paper", "exchange": self.exchange, "policy": self.policy,
                "coins": {c: t.status() for c, t in self.traders.items()},
                "pool": self.pool.status() if self.pool else None}

    # --- cash between sleeves and the bet pool ------------------------------------------------
    def move(self, usd, src, dst):
        """Move cash between two ledgers (sleeves or the bet pool). Bookkeeping only: nothing is sent to
        the exchange, because every ledger lives in the same account. Paper also moves simulated cash."""
        usd = round(Trader._amount(usd, "amount"), 2)
        if self.POOL in (src, dst):
            return self._move_pool(usd, src, dst)
        a, b = self.traders[src], self.traders[dst]
        with a._manual(), b._manual():
            if usd > a.ledger["usd"] + 1e-9:
                raise ValueError(f"the {src} sleeve has ${a.ledger['usd']:.2f} in cash; sell some {src} first")
            if not self.live:  # paper sleeves each simulate their own exchange balance: move it too
                a.broker.withdraw(usd)
                b.broker.deposit(usd)
            a._rescale_hwm(-usd); b._rescale_hwm(+usd)
            a.ledger["usd"] -= usd; a.ledger["withdrawn"] += usd
            b.ledger["usd"] += usd; b.ledger["deposited"] += usd
            a.journal(event="move", amount=usd, to=dst)
            b.journal(event="move", amount=usd, frm=src)
            a.save(); b.save()

    def _move_pool(self, usd, src, dst):
        if not self.pool:
            raise ValueError("the bet pool is not running (start the bot with BTCBOT_BETS=1)")
        if src == self.POOL:
            t = self.traders[dst]
            with t._manual():
                self.pool.defund(usd)  # raises if the cash is tied up in open bets
                if not self.live:
                    t.broker.deposit(usd)
                t._rescale_hwm(+usd)
                t.ledger["usd"] += usd; t.ledger["deposited"] += usd
                t.journal(event="move", amount=usd, frm=self.POOL)
                t.save()
        else:
            t = self.traders[src]
            with t._manual():
                if usd > t.ledger["usd"] + 1e-9:
                    raise ValueError(f"the {src} sleeve has ${t.ledger['usd']:.2f} in cash; sell some {src} first")
                if not self.live:
                    t.broker.withdraw(usd)
                t._rescale_hwm(-usd)
                t.ledger["usd"] -= usd; t.ledger["withdrawn"] += usd
                t.journal(event="move", amount=usd, to=self.POOL)
                t.save()
                self.pool.fund(usd)

    def fund_bet(self, bet_coin, want):
        """A bet on `bet_coin` fired but the pool is `want` dollars short: raise it from the sleeves.
        Idle sleeve cash first (free); otherwise sell just enough of the largest other coin with a maker
        order at the ask (no taker fallback). Never sells the coin being bet on: selling BTC to bet on BTC
        keeps the same exposure and only pays two fees. Never lets what the pool has been given exceed
        BET_MAX_SHARE of the portfolio, so a losing pool is not refilled automatically.
        Returns dollars added; raises ValueError/Busy saying why when it can't."""
        with self._fund_lock:
            pool, L = self.pool, self.pool.ledger
            prices = {c: t.broker.quote()[0] for c, t in self.traders.items()}
            total = sum(t.equity(prices[c]) for c, t in self.traders.items()) + pool.equity(prices)
            given = L["deposited"] - L["withdrawn"]
            room = BET_MAX_SHARE * total - given
            mins = pool.brokers[bet_coin].min_order_usd
            want = min(want, room)
            if want < 0.01 or want + pool.free_usd() < mins:
                raise ValueError(f"the pool has already been given ${given:,.2f}, the {BET_MAX_SHARE:.0%} cap "
                                 f"(${BET_MAX_SHARE * total:,.2f}); 'move' more in by hand to go past it")

            def usable(c):
                t = self.traders[c]
                return not (t.s.get("halted") or t.s.get("needs_review") or t.s.get("withdraw_pending") or t.busy)

            cash = [c for c, t in self.traders.items() if usable(c) and t.ledger["usd"] >= want]
            if cash:
                self.move(round(want + 0.005, 2), cash[0], self.POOL)
                return round(want + 0.005, 2)
            sell_usd = {c: max(want, t.broker.min_order_usd * 1.1) for c, t in self.traders.items()}  # Kraken's floor
            sellable = sorted((c for c, t in self.traders.items() if c != bet_coin and usable(c)
                               and t.ledger["qty"] * prices[c] >= sell_usd[c] + 0.5),
                              key=lambda c: -self.traders[c].ledger["qty"] * prices[c])
            if not sellable:
                held = [c for c, t in self.traders.items() if t.ledger["qty"] * prices[c] >= want + 0.5]
                raise ValueError(f"the only coin to sell is {bet_coin.lower()} itself (selling it to bet on it just "
                                 "pays fees twice)" if held == [bet_coin] else
                                 "no sleeve has idle cash or another coin to sell for the stake")
            src = sellable[0]
            t = self.traders[src]
            with t._manual():
                t.busy = f"selling for a {bet_coin} bet"
                try:
                    bid, ask = t.broker.quote()
                    qty = min(sell_usd[src] * 1.006 / ask, t.ledger["qty"])  # +0.6%: the maker fee and rounding
                    if (t.ledger["qty"] - qty) * bid < t.broker.min_order_usd:
                        qty = t.ledger["qty"]  # never leave a remainder too small to sell
                    fill = t._sell(lambda: t.broker.sell_maker(qty, deadline_s=600, ref_px=ask, fallback_taker=False),
                                   "bet funding")
                    if not fill:
                        raise ValueError(f"the {src.lower()} sale for its stake did not fill within 10 minutes")
                    t.journal(event="sell_for_bet", fill=fill, bet=bet_coin)
                    t.save()
                finally:
                    t.busy = None
            got = round(min(fill["size"] * fill["price"] - fill["fee"], t.ledger["usd"]) - 0.005, 2)
            self.move(got, src, self.POOL)
            return got

    # --- cross-selling ---------------------------------------------------------------------------
    def _ready(self, coins):
        """Refuse up front (before any order) if a sleeve is halted, under review or busy with an order."""
        for c in coins:
            t = self.traders[c]
            if t.s.get("halted") or t.s.get("needs_review") or t.s.get("withdraw_pending"):
                raise ValueError(f"the {c} sleeve is halted or needs review; type 'resume {c.lower()}' first")
            if t.busy:
                raise Busy(f"the {c} sleeve is busy ({t.busy}); try again when it finishes")

    def convert(self, src, dst, usd=None, fraction=None):
        """Sell `usd` worth (or `fraction`, or all) of src and buy dst with the proceeds. Market orders, so
        about 0.8% taker fee per leg; dst becomes a manual holding (the trend bot won't sell it on a signal)."""
        self._ready([src, dst])
        a, b = self.traders[src], self.traders[dst]
        px = a.broker.quote()[0]
        held = a.ledger["qty"]
        if usd is not None:
            qty = min(Trader._amount(usd, "amount") / px, held)
        else:
            qty = held * (fraction if fraction is not None else 1.0)
        if qty <= 0:
            raise ValueError(f"the {src} sleeve holds no {src}")
        if (held - qty) * px < a.broker.min_order_usd:
            qty = held  # never leave a remainder too small to ever sell
        sold = a.manual_sell(qty)
        proceeds = round(sold["size"] * sold["price"] - sold["fee"], 2)
        cash = round(min(proceeds, a.ledger["usd"]) - 0.005, 2)
        try:
            self.move(cash, src, dst)
            bought = b.manual_buy(cash)
        except Exception as e:  # the sale already happened: say exactly where the money is
            where = dst if b.ledger["usd"] >= cash - 0.01 else src
            raise RuntimeError(f"sold {sold['size']:.6g} {src} for ${proceeds:.2f}, but the second step failed ({e}); "
                               f"the cash is waiting in the {where} sleeve") from e
        return sold, bought

    def rebalance(self, weights):
        """Redistribute the sleeves' combined value to target weights {coin: percent} (sum <= 100;
        the rest stays as cash in the first-named coin's sleeve). Sells first, then buys. Every step is
        reported; if one fails, the reply says what already happened and where the cash is."""
        total_w = sum(weights.values())
        if not 0 < total_w <= 100.0001:
            raise ValueError("weights must add up to 100% or less")
        self._ready(self.coins)
        prices = {c: t.broker.quote()[0] for c, t in self.traders.items()}
        values = {c: t.ledger["qty"] * prices[c] for c, t in self.traders.items()}
        total = sum(values.values()) + sum(t.ledger["usd"] for t in self.traders.values())
        target = {c: total * weights.get(c, 0.0) / 100 for c in self.traders}
        hub = next(iter(weights))
        done = []
        try:
            for c, t in self.traders.items():  # 1) sell what is over target
                excess = values[c] - target[c]
                if excess > t.broker.min_order_usd and t.ledger["qty"] > 0:
                    qty = min(excess / prices[c], t.ledger["qty"])
                    if (t.ledger["qty"] - qty) * prices[c] < t.broker.min_order_usd:
                        qty = t.ledger["qty"]  # no unsellable remainder
                    done.append(("sell", c, t.manual_sell(qty)))
            for c, t in self.traders.items():  # 2) gather spare cash in the hub sleeve
                if c != hub and t.ledger["usd"] >= 0.02:
                    self.move(round(t.ledger["usd"] - 0.005, 2), c, hub)
            short = {c: target[c] - t.ledger["qty"] * prices[c] for c, t in self.traders.items()}
            order = [c for c in self.traders if c != hub] + [hub]  # the hub buys last with what is left
            for c in order:  # 3) buy what is under target
                t = self.traders[c]
                want = min(short[c], self.traders[hub].ledger["usd"] if c != hub else t.ledger["usd"])
                if want < max(t.broker.min_order_usd, 0.02):
                    continue
                if c != hub:
                    self.move(round(want - 0.005, 2), hub, c)
                cash = min(want, t.ledger["usd"])
                if cash >= t.broker.min_order_usd:
                    done.append(("buy", c, t.manual_buy(cash)))
        except Exception as e:
            cash = ", ".join(f"{c.lower()} ${t.ledger['usd']:.2f}" for c, t in self.traders.items())
            trades = "; ".join(f"{s} {f['size']:.6g} {c}" for s, c, f in done) or "no trades"
            raise RuntimeError(f"rebalance stopped part-way ({e}). done so far: {trades}. cash now: {cash}") from e
        return done

    def run(self, once=False, stop=None):
        for c, t in self.traders.items():  # sequential: they share the exchange account
            for attempt in range(3):
                try:
                    t.reconcile()
                    break
                except Exception as e:
                    log.warning("%s reconcile attempt %d failed: %s", c, attempt + 1, e)
                    time.sleep(10 * (attempt + 1))
            else:
                # never let one coin's startup failure stop the others (or the BTC kill switch)
                t.s["needs_review"] = "startup reconciliation with the exchange failed; check open orders, then 'resume'"
                t.journal(event="halted", note=t.s["needs_review"])
                t.save()
        threads = []
        for c, t in self.traders.items():
            th = threading.Thread(target=t.run, kwargs={"once": once, "stop": stop, "reconcile": False},
                                  daemon=True, name=f"trader-{c}")
            th.start(); threads.append(th)
        if self.pool and not once:
            th = threading.Thread(target=self.pool.run, kwargs={"stop": stop}, daemon=True, name="bets")
            th.start(); threads.append(th)
        for th in threads:
            th.join()


def configure(argv=None):
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    a, _ = ap.parse_known_args(argv)
    mode = os.environ.get("BTCBOT_MODE", "paper")
    exchange = os.environ.get("BTCBOT_EXCHANGE", "coinbase")
    policy_file = os.environ.get("BTCBOT_POLICY", "policy.json")
    coins = [c.strip().upper() for c in os.environ.get("BTCBOT_COINS", "BTC").split(",") if c.strip()]
    unknown = [c for c in coins if c not in COINS]
    if unknown:
        raise SystemExit(f"unknown coin(s) {unknown}; choose from {list(COINS)}")
    if mode == "live" and os.environ.get("BTCBOT_CONFIRM_LIVE") != "I_UNDERSTAND":
        raise SystemExit("live mode requires BTCBOT_CONFIRM_LIVE=I_UNDERSTAND")
    STATE.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(STATE / f"{exchange}_{mode}.log")])
    return Portfolio(mode, exchange, policy_file, coins, bets=os.environ.get("BTCBOT_BETS") == "1"), a


def main():
    portfolio, a = configure()
    portfolio.run(once=a.once)


if __name__ == "__main__":
    main()
