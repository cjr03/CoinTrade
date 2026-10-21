"""Short-term bet pool: a small, fenced-off ledger that places up/down bets on every coin when a
calibrated direction model is confident enough.

    python -m btcbot.bets --train          # fit one model per coin -> artifacts/bets_<COIN>.pkl

Every out-of-sample test in this project says these bets lose money after Kraken's fees (README);
they run because the owner asked for them, so they are fenced in:
  * the pool trades only its own cash (funded with `move 15 btc bets`); trend sleeves never see it,
  * when a bet fires and the pool is short of cash, it may raise the stake from the sleeves (idle cash
    first, then selling another coin; 'bets fund off' stops this), but never past a lifetime cap on
    what the pool has been given (BTCBOT_BET_MAX_SHARE, default 25% of the portfolio), so losses are
    not refilled automatically,
  * one bet per coin at a time, stake <= half the pool, a 2-sigma stop on every bet,
  * the pool halts itself when its equity falls BETS_POOL_STOP (default 50%) below what was put in.
Bets are LONG-ONLY by default. Short bets (Kraken 2:1 margin, BTCBOT_BET_SHORTS=1 or 'bets shorts on')
are experimental: margin eligibility, reduce-only closes and Kraken's actual rollover charges are not
reconciled, so use them in paper mode only.
"""
import argparse
import json
import logging
import math
import os
import pickle
import threading
import time
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd

from .broker import PaperBroker
from .coins import COINS
from .data import download, load
from .features import build_features, feature_columns, regularize
from .kraken import KrakenBroker, KrakenError, KrakenPaperBroker
from .train import LGB_PARAMS

ART_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "artifacts"
HORIZONS = (4, 12, 24)
THRESHOLDS = (0.55, 0.6, 0.65, 0.7, 0.75)
MIN_BETS_PER_YEAR = 12  # "active": a threshold that almost never fires is not what was asked for
MAKER, TAKER, MARGIN_OPEN, ROLL_4H = 0.004, 0.008, 0.0002, 0.0002
LATE_MINUTES = 15
log = logging.getLogger("btcbot")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# --- model -------------------------------------------------------------------------------------
def cross_coin(coin):
    return "ETH" if coin == "BTC" else "BTC"


def coin_frame(coin, days=None):
    h1 = load(COINS[coin]["product"], "ONE_HOUR")
    cross = load(COINS[cross_coin(coin)]["product"], "ONE_HOUR")
    if days:
        h1 = h1[h1.index >= h1.index.max() - pd.Timedelta(days=days)]
    f, bars = build_features(h1, None, cross.loc[h1.index.min():])
    return f, bars


class Calibrated:
    """LightGBM seeds averaged, then mapped to a probability with monotone binned calibration."""

    def __init__(self, models, xs, ys, cols):
        self.models, self.xs, self.ys, self.cols = models, xs, ys, cols

    def raw(self, X):
        return np.mean([m.predict_proba(X[self.cols])[:, 1] for m in self.models], axis=0)

    def prob(self, X):
        return np.interp(self.raw(X), self.xs, self.ys)


def fit_calibrated(fit, es, cal, cols, y_col, seeds=(0, 1, 2)):
    params = {**LGB_PARAMS, "objective": "binary", "n_jobs": 4, "metric": "auc"}
    params.pop("alpha", None)
    models = []
    for s in seeds:
        m = lgb.LGBMClassifier(**params, random_state=s)
        m.fit(fit[cols], fit[y_col], eval_set=[(es[cols], es[y_col])], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        models.append(m)
    tmp = Calibrated(models, [0, 1], [0, 1], cols)
    raw = tmp.raw(cal)
    q = np.unique(np.quantile(raw, np.linspace(0, 1, 21)))
    xs, ys = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        sel = (raw >= lo) & (raw <= hi)
        if sel.sum() >= 20:
            xs.append(raw[sel].mean()); ys.append(cal[y_col].values[sel].mean())
    ys = np.maximum.accumulate(np.array(ys))  # monotone: a higher score never means a lower probability
    return Calibrated(models, np.array(xs), ys, cols)


SHORTS_IN_TRAINING = os.environ.get("BTCBOT_BET_SHORTS", "0") == "1"


def simulate_bets(p, fwd, H, thr, shorts=SHORTS_IN_TRAINING):
    """One bet at a time; long when p >= thr, short when p <= 1-thr; exit after H hours."""
    out, i, n = [], 0, len(p)
    while i < n:
        side = 1 if p[i] >= thr else (-1 if shorts and p[i] <= 1 - thr else 0)
        if side and np.isfinite(fwd[i]):
            g = side * (math.exp(fwd[i]) - 1)
            cost = 2 * MAKER + ((MARGIN_OPEN + ROLL_4H * math.ceil(H / 4)) if side < 0 else 0)
            out.append((side, g - cost))
            i += H
        else:
            i += 1
    return out


def train_coin(coin, verbose=True):
    """Pick horizon + threshold on the most recent 6 months (held out, purged); require activity."""
    f, bars = coin_frame(coin, days=4 * 365)
    cols = feature_columns(f)
    best = None
    for H in HORIZONS:
        d = f[cols].copy()
        d["fwd"] = np.log(bars["close"].shift(-H) / bars["close"])
        d["y"] = (d["fwd"] > 0).astype(float)
        d = d.dropna(subset=["fwd"])
        end = d.index.max()
        purge = pd.Timedelta(hours=H)
        val_start = end - pd.Timedelta(days=182)
        cal_start = val_start - pd.Timedelta(days=90)
        es_start = cal_start - pd.Timedelta(days=60)
        fit = d[(d.index < es_start - purge) & (d.index >= es_start - pd.Timedelta(days=900))]
        es = d[(d.index >= es_start) & (d.index < cal_start - purge)]
        cal = d[(d.index >= cal_start) & (d.index < val_start - purge)]
        val = d[d.index >= val_start]
        model = fit_calibrated(fit, es, cal, cols, "y")
        pv = model.prob(val)
        years = len(val) / 8760
        for thr in THRESHOLDS:
            bets = simulate_bets(pv, val["fwd"].values, H, thr)
            per_year = len(bets) / years
            if per_year < MIN_BETS_PER_YEAR:
                continue
            nets = np.array([b[1] for b in bets])
            score = float(np.sum(np.log1p(nets)))
            stats = {"H": H, "threshold": thr, "bets_per_year": round(per_year, 1),
                     "win_rate": float((nets > 0).mean()), "avg_net_bp": float(nets.mean() * 1e4),
                     "final_from_100": float(100 * np.prod(1 + nets)), "short_share": float(np.mean([b[0] < 0 for b in bets])),
                     "val_start": str(val.index.min().date()), "val_end": str(val.index.max().date())}
            if verbose:
                print(f"{coin} H={H:2d} p>={thr:.2f}: {per_year:5.0f} bets/yr  win {stats['win_rate']:.0%}  "
                      f"avg {stats['avg_net_bp']:+6.1f}bp  $100 -> ${stats['final_from_100']:.2f}")
            if best is None or score > best[0]:
                best = (score, H, thr, model, stats)
    if best is None:
        raise RuntimeError(f"{coin}: no horizon/threshold reaches {MIN_BETS_PER_YEAR} bets a year")
    _, H, thr, model, stats = best
    bundle = {"coin": coin, "H": H, "threshold": thr, "model": model, "cross": cross_coin(coin),
              "validation": stats, "trained_at": now_iso()}
    with open(ART_DIR / f"bets_{coin}.pkl", "wb") as fh:
        pickle.dump(bundle, fh)
    if verbose:
        print(f"{coin}: chose H={H}h p>={thr} -> validation {stats}")
    return bundle


# --- the pool ----------------------------------------------------------------------------------
class BetPool:
    def __init__(self, mode, exchange, coins, state_dir, sink=None, reserved_elsewhere=lambda: 0.0):
        self.live = mode == "live"
        self.exchange = exchange
        self.coins = [c for c in coins if (ART_DIR / f"bets_{c}.pkl").exists()]
        self.state_path = state_dir / f"{exchange}_{mode}_bets.json"
        self.journal_path = state_dir / f"{exchange}_{mode}_bets_journal.jsonl"
        self.s = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.s.setdefault("ledger", {"usd": 0.0, "deposited": 0.0, "withdrawn": 0.0, "fees": 0.0, "realized": 0.0})
        self.s.setdefault("bets", {})
        self.s.setdefault("committed", {})  # stake reserved for orders in flight, per coin
        self.s.setdefault("auto", True)
        self.s.setdefault("shorts", False)
        self.s.setdefault("autofund", True)
        if self.live and os.environ.get("BTCBOT_BET_SHORTS") != "1":
            self.s["shorts"] = False  # live margin shorts need an explicit opt-in
        stale = {k: v for k, v in self.s.get("committed", {}).items() if v}
        closing = self.s.get("closing") or {}
        if stale or closing:
            self.s["needs_review"] = (f"previous run stopped mid-order (opening {sorted(stale)}, closing {sorted(closing)}); "
                                      "check Kraken orders and balances, fix the bet ledger, then 'bets resume'")
        self.s["committed"] = {}
        self.s["closing"] = {}
        self.lock = threading.RLock()
        self.sink = sink
        self.reserved_elsewhere = reserved_elsewhere  # cash booked to the trend sleeves
        self.stop_frac = float(os.environ.get("BTCBOT_BETS_POOL_STOP", 0.5))
        self.funder = None  # set by Portfolio: (bet coin, dollars wanted) -> dollars added, or raises with why not
        self.models = {}
        for c in self.coins:
            # bets_<COIN>.pkl is produced locally by `python -m btcbot.bets --train`; never load one from elsewhere
            with open(ART_DIR / f"bets_{c}.pkl", "rb") as fh:
                self.models[c] = pickle.load(fh)
        self.brokers = {c: self._broker(c) for c in self.coins}
        self.busy = {}

    def _order_event(self, coin, **kw):
        if kw.get("stage") == "UNSETTLED":
            with self.lock:
                self.s["needs_review"] = (f"{coin} bet order {kw.get('order_id') or kw.get('userref')} could not be "
                                          "confirmed closed; its fills may be missing from the bet ledger")
                self.save()
        self.journal(event="order", coin=coin, pool=True, **kw)

    def _broker(self, coin):
        hook = lambda **kw: self._order_event(coin, **kw)
        cap = float(os.environ.get("BTCBOT_MAX_NOTIONAL", 110))
        if self.live:
            if self.exchange != "kraken":
                raise SystemExit("the bet pool supports Kraken only (margin shorts)")
            return KrakenBroker(os.environ["KRAKEN_API_KEY"], os.environ["KRAKEN_API_SECRET"], max_notional=cap,
                                on_order=hook, coin=coin)
        paper_state = self.s.setdefault("paper", {}).setdefault(coin, {})
        paper = KrakenPaperBroker if self.exchange == "kraken" else PaperBroker
        return paper(paper_state, MAKER, TAKER, capital=0.0, max_notional=cap, coin=coin)

    # --- persistence ------------------------------------------------------------------------
    def save(self):
        with self.lock:
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.s, indent=1, default=str))
            tmp.replace(self.state_path)

    def journal(self, **rec):
        rec["ts"] = now_iso()
        rec.setdefault("pool", True)
        with open(self.journal_path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        if self.sink:
            self.sink(json.loads(json.dumps(rec, default=str)))
        log.info("bets %s", rec)

    # --- ledger -------------------------------------------------------------------------------
    @property
    def ledger(self):
        return self.s["ledger"]

    def free_usd(self):
        """Pool cash not tied up: minus short collateral and stakes reserved for orders in flight."""
        tied = sum(b["stake"] for b in self.s["bets"].values() if b["side"] < 0)
        return self.ledger["usd"] - tied - sum(self.s["committed"].values())

    def equity(self, prices):
        eq = self.ledger["usd"]
        for c, b in self.s["bets"].items():
            px = prices.get(c) or b["entry_px"]
            eq += b["qty"] * px if b["side"] > 0 else b["qty"] * (b["entry_px"] - px)
        return eq

    def status(self):
        with self.lock:
            s = json.loads(json.dumps(self.s, default=str))
        s.pop("paper", None)
        s["models"] = {c: {"H": m["H"], "threshold": m["threshold"], "validation": m["validation"]}
                       for c, m in self.models.items()}
        s["busy"] = {c: b for c, b in self.busy.items() if b}
        return s

    def fund(self, usd):
        """Cash moved in from a trend sleeve (bookkeeping; paper also moves the simulated cash)."""
        with self.lock:
            self.ledger["usd"] += usd; self.ledger["deposited"] += usd
            self.journal(event="pool_fund", amount=usd)
            self.save()

    def defund(self, usd):
        with self.lock:
            if usd > self.free_usd() + 1e-9:
                raise ValueError(f"the bet pool has ${max(self.free_usd(), 0):.2f} free (the rest is in open bets)")
            self.ledger["usd"] -= usd; self.ledger["withdrawn"] += usd
            self.journal(event="pool_defund", amount=usd)
            self.save()

    def set_auto(self, on):
        with self.lock:
            self.s["auto"] = on
            self.journal(event="pool_auto", on=on)
            self.save()

    def resume(self):
        with self.lock:
            cleared = [k for k in ("halted", "needs_review") if self.s.pop(k, None)]
            self.journal(event="pool_resume", cleared=cleared)
            self.save()
            return cleared

    # --- signal -------------------------------------------------------------------------------
    def predict(self, coin):
        m = self.models[coin]
        for c in (coin, m["cross"]):
            download(COINS[c]["product"], "ONE_HOUR", 800)
        now = pd.Timestamp.now(tz="UTC")
        last_closed = now.floor("h") - pd.Timedelta(hours=1)
        h1 = load(COINS[coin]["product"], "ONE_HOUR").loc[:last_closed]
        cross = load(COINS[m["cross"]]["product"], "ONE_HOUR").loc[:last_closed]
        if h1.empty or h1.index.max() != last_closed or cross.empty or cross.index.max() != last_closed:
            raise RuntimeError(f"stale data for {coin} bets (last bar {h1.index.max() if len(h1) else None})")
        h1 = h1.iloc[-24 * 900:]
        f, bars = build_features(h1, None, cross.loc[h1.index.min():])
        row = f.iloc[[-1]]
        p = float(m["model"].prob(row)[0])
        sigma_h = float(row["sigma"].iloc[0]) * math.sqrt(m["H"])
        return {"bar": str(row.index[0]), "p_up": p, "close": float(bars["close"].iloc[-1]), "sigma_h": sigma_h,
                "H": m["H"], "threshold": m["threshold"]}

    # --- one coin's hourly decision -------------------------------------------------------------
    def step(self, coin):
        br = self.brokers[coin]
        bet = self.s["bets"].get(coin)
        if bet:  # exits never wait for a fresh signal: stale data must not keep a losing bet open
            bid, ask = br.quote()
            start = pd.Timestamp(bet.get("bar_close") or bet["opened"])
            held_h = (pd.Timestamp.now(tz="UTC") - start).total_seconds() / 3600
            adverse = (bid < bet["stop_px"]) if bet["side"] > 0 else (ask > bet["stop_px"])
            if adverse or held_h >= bet["H"] - 0.25:
                self._close(coin, bet, "stop" if adverse else "time", {"close": bid if bet["side"] > 0 else ask})
            return
        sig = self.predict(coin)
        late = (pd.Timestamp.now(tz="UTC") - (pd.Timestamp(sig["bar"]) + pd.Timedelta(hours=1))).total_seconds() / 60
        with self.lock:
            self.s.setdefault("signals", {})[coin] = sig
            already = self.s.setdefault("last_bar", {}).get(coin) == sig["bar"]
            if already or late > LATE_MINUTES:
                return
            self.s["last_bar"][coin] = sig["bar"]
            self.save()
        bid, ask = br.quote()
        if self.s.get("halted") or self.s.get("needs_review") or not self.s.get("auto", True):
            return
        thr = sig["threshold"]
        side = 1 if sig["p_up"] >= thr else (-1 if self.s.get("shorts") and sig["p_up"] <= 1 - thr else 0)
        if side:
            self._open(coin, side, sig, bid, ask)

    def _target(self, coin):
        """The stake a bet wants before cash limits: half the pool, at least the exchange minimum."""
        pool_eq = self.ledger["usd"] + sum(b["stake"] for b in self.s["bets"].values() if b["side"] > 0)
        return min(max(self.brokers[coin].min_order_usd * 1.1, pool_eq * 0.5), self.brokers[coin].max_notional)

    def _stake(self, coin):
        with self.lock:
            stake = min(self.free_usd(), self._target(coin))
            if self.live:
                acct_total = self.brokers[coin].total_usd()
                stake = min(stake, acct_total - self.reserved_elsewhere()
                            - sum(b["stake"] for b in self.s["bets"].values() if b["side"] < 0)
                            - sum(v for k, v in self.s["committed"].items() if k != coin))
            if stake < self.brokers[coin].min_order_usd:
                return 0.0
            self.s["committed"][coin] = stake  # reserve before any order goes out
            self.save()
            return stake

    def _open(self, coin, side, sig, bid, ask):
        with self.lock:
            short = self._target(coin) - self.free_usd()
        why = f"bet pool too small for a {coin} bet; fund it with 'move 15 btc bets'"
        if short > 0.01 and side > 0 and self.funder and self.s.get("autofund", True):
            self.busy[coin] = "raising the stake"
            try:
                self.funder(coin, short)
            except Exception as e:  # nothing was sold, or the sale is fully booked: just say why
                why = f"no {coin} bet: {e}"
            finally:
                self.busy[coin] = None
        stake = self._stake(coin)
        if not stake:
            self.journal(event="pool_skip", coin=coin, note=why)
            return
        br = self.brokers[coin]
        self.busy[coin] = f"opening {'long' if side > 0 else 'short'}"
        if not self.live:  # paper: the simulated broker only needs to see this bet's stake
            br.s["paper"].update(usd=stake, btc=0.0)
        fill = None
        try:
            if side > 0:
                fill = br.buy_maker(stake, deadline_s=1800, ref_px=sig["close"])
            else:
                fill = br.short_maker(stake / (ask * (1 + MAKER)), deadline_s=1800, ref_px=sig["close"])
        except KrakenError as e:
            if side < 0 and (e.fatal or "Margin" in str(e) or "position" in str(e)):
                with self.lock:
                    self.s["shorts"] = False
                self.journal(event="pool_shorts_off", coin=coin,
                             note=f"Kraken refused a margin short ({e}); bets continue long only")
            else:
                with self.lock:
                    self.s["needs_review"] = f"{coin} bet failed while opening ({e}); check Kraken orders/positions"
                raise
        except Exception as e:
            with self.lock:
                self.s["needs_review"] = f"{coin} bet failed while opening ({e}); check Kraken orders/positions"
            raise
        finally:
            with self.lock:
                self.s["committed"].pop(coin, None)
                if fill:
                    L = self.ledger
                    if side > 0:
                        L["usd"] -= fill["size"] * fill["price"] + fill["fee"]
                    else:
                        L["usd"] -= fill["fee"]
                    L["fees"] += fill["fee"]
                    stop = fill["price"] * math.exp(-2 * side * sig["sigma_h"])
                    self.s["bets"][coin] = {"side": side, "qty": fill["size"], "entry_px": fill["price"],
                                            "stake": fill["size"] * fill["price"], "opened": now_iso(),
                                            "bar_close": (str(pd.Timestamp(sig["bar"]) + pd.Timedelta(hours=1))
                                                          if sig.get("bar") else now_iso()),
                                            "H": sig["H"], "stop_px": stop, "p_up": sig["p_up"],
                                            "fee_open": fill["fee"]}
                    self.journal(event="bet_open", coin=coin, side="long" if side > 0 else "short", fill=fill,
                                 p_up=sig["p_up"], H=sig["H"])
                self.busy[coin] = None
                self.save()
        if not fill:
            self.journal(event="bet_unfilled", coin=coin, side="long" if side > 0 else "short", p_up=sig["p_up"])

    def _close(self, coin, bet, reason, sig):
        br = self.brokers[coin]
        self.busy[coin] = f"closing ({reason})"
        with self.lock:  # crash-safety: a restart with this marker set must not sell again
            self.s.setdefault("closing", {})[coin] = {"qty": bet["qty"], "at": now_iso()}
            self.save()
        if not self.live:
            br.s["paper"].update(usd=0.0, btc=bet["qty"])
        fill = None
        try:
            if bet["side"] > 0:
                fill = (br.market_sell(bet["qty"]) if reason == "stop"
                        else br.sell_maker(bet["qty"], deadline_s=1800, ref_px=sig["close"]))
            else:
                fill = (br.market_cover(bet["qty"]) if reason == "stop"
                        else br.cover_maker(bet["qty"], deadline_s=1800, ref_px=sig["close"]))
        except Exception as e:
            with self.lock:
                self.s["needs_review"] = f"{coin} bet failed while closing ({e}); check Kraken orders/positions"
            raise
        finally:
            with self.lock:
                if fill:
                    L = self.ledger
                    done = min(fill["size"], bet["qty"])
                    share = done / bet["qty"] if bet["qty"] else 1.0
                    fee_open = bet.get("fee_open", 0.0) * share  # the entry fee belongs in this bet's P&L too
                    if bet["side"] > 0:
                        L["usd"] += done * fill["price"] - fill["fee"]
                        pnl = done * (fill["price"] - bet["entry_px"]) - fill["fee"] - fee_open
                    else:
                        held_h = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(bet["opened"])).total_seconds() / 3600
                        margin = bet["stake"] * (MARGIN_OPEN + ROLL_4H * math.ceil(max(held_h, 0.01) / 4))
                        pnl = done * (bet["entry_px"] - fill["price"]) - fill["fee"] - margin - fee_open
                        L["usd"] += pnl + fee_open  # the entry fee already left cash when the short opened
                        L["fees"] += margin
                    L["fees"] += fill["fee"]
                    L["realized"] += pnl
                    bet["qty"] -= done
                    bet["fee_open"] = bet.get("fee_open", 0.0) - fee_open
                    bet["stake"] *= (1 - share)
                    if bet["qty"] * fill["price"] < br.min_order_usd:
                        if bet["qty"] > 0:  # below the exchange minimum: cannot be sold; written off, kept on record
                            self.s.setdefault("dust", {})[coin] = self.s.get("dust", {}).get(coin, 0.0) + bet["qty"]
                        self.s["bets"].pop(coin, None)
                    else:
                        self.s["bets"][coin] = bet
                    self.s.setdefault("history", []).append({"coin": coin, "side": bet["side"], "reason": reason,
                                                             "entry": bet["entry_px"], "exit": fill["price"],
                                                             "pnl": pnl, "closed": now_iso()})
                    self.s["history"] = self.s["history"][-200:]
                    self.journal(event="bet_close", coin=coin, reason=reason, fill=fill, pnl=pnl,
                                 side="long" if bet["side"] > 0 else "short")
                else:
                    self.journal(event="bet_close_unfilled", coin=coin, reason=reason)
                self.s.get("closing", {}).pop(coin, None)
                self.busy[coin] = None
                self._check_pool_stop()
                self.save()

    def _check_pool_stop(self):
        L = self.ledger
        basis = L["deposited"] - L["withdrawn"]
        if basis > 0 and not self.s["bets"] and L["usd"] < basis * (1 - self.stop_frac) and not self.s.get("halted"):
            self.s["halted"] = now_iso()
            self.journal(event="pool_halted", note=f"bet pool is down {1 - L['usd'] / basis:.0%} from what was put in; "
                                                   "betting stopped (type 'bets resume' to restart)")

    # --- lifecycle ----------------------------------------------------------------------------
    def run_coin(self, coin, stop=None):
        while not (stop and stop.is_set()):
            try:
                self.step(coin)
            except Exception as e:
                log.exception("bet step %s failed", coin)
                self.journal(event="pool_error", coin=coin, note=str(e))
                self.save()
            nxt = (time.time() // 3600 + 1) * 3600 + 40  # after the trend sleeves (+20s) have refreshed data
            while time.time() < nxt and not (stop and stop.is_set()):
                time.sleep(min(5, max(nxt - time.time(), 0.1)))

    def run(self, stop=None):
        if self.live:
            for c, br in self.brokers.items():
                try:
                    br.cancel_open_orders()
                except Exception as e:
                    log.warning("bet pool startup cancel %s: %s", c, e)
        threads = [threading.Thread(target=self.run_coin, args=(c, stop), daemon=True, name=f"bets-{c}")
                   for c in self.coins]
        for th in threads:
            th.start()
        self.journal(event="pool_startup", coins=self.coins, ledger=dict(self.ledger), bets=self.s["bets"],
                     shorts=self.s.get("shorts"))
        for th in threads:
            th.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--coins", default=",".join(COINS))
    a = ap.parse_args()
    if a.train:
        # import by package name so pickled models reference btcbot.bets.Calibrated, not __main__
        from btcbot.bets import train_coin as train
        for c in a.coins.split(","):
            train(c.strip().upper())


if __name__ == "__main__":
    main()
