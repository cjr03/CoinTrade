"""Fee-aware event-driven backtest of the long/flat policy on hourly bars.

Decision at close of bar t (price close[t]).  A post-only limit order at close[t] is only
considered filled if price trades *strictly through* it within the next `fill_bars` bars
(low < px for buys, high > px for sells).  Otherwise the entry is abandoned, or an exit is
forced through with a taker order at the next close (+ slippage).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 24 * 365


@dataclass
class Policy:
    maker_fee: float = 0.006
    taker_fee: float = 0.012
    entry_margin: float = 0.01      # required edge above round-trip cost (log-return units)
    exit_thr: float | None = None   # exit when pred < exit_thr (default -maker_fee)
    horizon: int = 24
    min_hold: int = 24
    stop_sigma: float = 3.0         # stop at entry - stop_sigma * sigma_H
    time_stop_mult: int = 3
    fill_bars: int = 2
    exit_fill_bars: int = 2
    half_size: bool = True          # 50% position unless edge >= 2x threshold and vol is calm
    taker_slip: float = 0.0002
    kill_dd: float = 0.25           # flatten on 25% drawdown from HWM ...
    kill_cooldown: int = 168        # ... then sit out a week and reset HWM (live bot halts for a human)
    min_order_usd: float = 1.0

    @property
    def round_trip(self):
        return (1 + self.maker_fee) / (1 - self.maker_fee) - 1

    @property
    def entry_thr(self):
        return self.round_trip + self.entry_margin


@dataclass
class Result:
    equity: pd.Series
    position: pd.Series
    trades: list = field(default_factory=list)
    fees: float = 0.0
    halted: bool = False


def simulate(close, high, low, pred, sigma_h, policy, vol_pctile=None, capital=100.0,
             entry_thr=None, exit_thr=None):
    """pred: expected log return over the horizon at each bar close (NaN = no signal).
    sigma_h: horizon volatility (log units) used for the stop.  entry_thr / exit_thr: optional
    per-bar arrays overriding the policy thresholds; NaN entry threshold means 'do not open'."""
    idx = close.index
    c, h, l = close.values, high.values, low.values
    p = np.asarray(pred, dtype=float)
    sh = np.asarray(sigma_h, dtype=float)
    vp = np.full(len(c), 0.5) if vol_pctile is None else np.nan_to_num(np.asarray(vol_pctile, float), nan=0.5)
    thr = np.broadcast_to(np.asarray(policy.entry_thr if entry_thr is None else entry_thr, float), (len(c),))
    if exit_thr is None:
        exit_thr = -policy.maker_fee if policy.exit_thr is None else policy.exit_thr
    xt = np.broadcast_to(np.asarray(exit_thr, float), (len(c),))
    n = len(c)

    cash, btc, fees = capital, 0.0, 0.0
    frac = 0.0
    entry_px, entry_t, stop_px = 0.0, -1, 0.0
    hwm = capital
    halted = False
    halt_until = -1
    eq = np.empty(n)
    pos = np.empty(n)
    trades = []
    t = 0
    busy_until = -1  # bars during which an order is working; no new decisions
    fill_k = -1      # bar at which holdings changed in the current decision (for marking)

    def buy(target_frac, t):
        nonlocal cash, btc, fees, frac, fill_k
        equity = cash + btc * c[t]
        want_usd = target_frac * equity - btc * c[t]
        if want_usd < policy.min_order_usd:
            return t
        px = c[t]
        end = min(t + policy.fill_bars, n - 1)
        for k in range(t + 1, end + 1):
            if l[k] < px:
                usd = min(want_usd, cash)
                fee = usd * policy.maker_fee
                qty = (usd - fee) / px
                cash -= usd; btc += qty; fees += fee
                frac = target_frac
                trades.append(("buy", idx[k], px, usd, fee, "maker"))
                fill_k = k
                return k
        return end  # not filled: abandon entry

    def sell(t, reason, force_taker=False):
        nonlocal cash, btc, fees, frac, fill_k
        if btc * c[t] < policy.min_order_usd:
            cash += btc * c[t]; btc = 0.0; frac = 0.0
            fill_k = t
            return t
        px = c[t]
        if not force_taker:
            end = min(t + policy.exit_fill_bars, n - 1)
            for k in range(t + 1, end + 1):
                if h[k] > px:
                    gross = btc * px
                    fee = gross * policy.maker_fee
                    cash += gross - fee; fees += fee
                    trades.append(("sell", idx[k], px, gross, fee, "maker:" + reason))
                    btc = 0.0; frac = 0.0
                    fill_k = k
                    return k
            t = end
        px = c[t] * (1 - policy.taker_slip)
        gross = btc * px
        fee = gross * policy.taker_fee
        cash += gross - fee; fees += fee
        trades.append(("sell", idx[t], px, gross, fee, "taker:" + reason))
        btc = 0.0; frac = 0.0
        fill_k = t
        return t

    # full size only when edge is >= 2x the larger of (threshold, round-trip cost) and vol is calm
    full_ok = (p >= 2 * np.maximum(thr, policy.round_trip)) & (vp < 0.8)

    while t < n:
        pre_cash, pre_btc = cash, btc
        fill_k = t
        equity = cash + btc * c[t]
        if halt_until >= 0 and t > halt_until:
            hwm, halt_until = equity, -1  # cooldown over: restart drawdown tracking
        if t > halt_until:
            hwm = max(hwm, equity)
        if t > busy_until and t > halt_until:
            if btc > 0 and equity < hwm * (1 - policy.kill_dd):
                busy_until = sell(t, "kill", force_taker=True)
                halted = True
                halt_until = busy_until + policy.kill_cooldown
            elif btc > 0:
                held = t - entry_t
                stop_hit = c[t] < stop_px
                time_up = held >= policy.time_stop_mult * policy.horizon
                signal_exit = held >= policy.min_hold and (np.isnan(p[t]) or p[t] < xt[t])
                if stop_hit or time_up or signal_exit:
                    reason = "stop" if stop_hit else "time" if time_up else "signal"
                    busy_until = sell(t, reason, force_taker=stop_hit)
                elif policy.half_size and frac < 1.0 and not np.isnan(thr[t]) and full_ok[t]:
                    busy_until = buy(1.0, t)
            elif not np.isnan(p[t]) and not np.isnan(thr[t]) and p[t] > thr[t]:
                target = 1.0
                if policy.half_size and not full_ok[t]:
                    target = 0.5
                k = buy(target, t)
                if btc > 0:
                    entry_px, entry_t = c[t], k
                    stop_px = entry_px * np.exp(-policy.stop_sigma * sh[t]) if np.isfinite(sh[t]) else 0.0
                busy_until = k
        # mark-to-market for every bar up to busy_until
        upto = max(t, min(busy_until, n - 1))
        for k in range(t, upto + 1):
            cb, bb = (pre_cash, pre_btc) if k < fill_k else (cash, btc)  # order still working
            eq[k] = cb + bb * c[k]
            pos[k] = (bb * c[k]) / eq[k] if eq[k] > 0 else 0.0
            if k > t and k > halt_until:
                hwm = max(hwm, eq[k])
        t = upto + 1

    return Result(pd.Series(eq, idx), pd.Series(pos, idx), trades, fees, halted)


def metrics(equity, trades=None, fees=None):
    r = equity.pct_change().fillna(0.0)
    years = max(len(equity) / HOURS_PER_YEAR, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0]
    sd = r.std()
    dd = (equity / equity.cummax() - 1).min()
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    out = {
        "final": float(equity.iloc[-1]),
        "total_ret": float(total - 1),
        "cagr": float(total ** (1 / years) - 1),
        "sharpe": float(r.mean() / sd * np.sqrt(HOURS_PER_YEAR)) if sd > 0 else 0.0,
        "sortino": float(r.mean() / downside * np.sqrt(HOURS_PER_YEAR)) if downside and downside > 0 else 0.0,
        "max_dd": float(dd),
        "exposure": None,
    }
    if trades is not None:
        out["round_trips"] = sum(1 for tr in trades if tr[0] == "sell")
        out["round_trips_per_year"] = out["round_trips"] / years
    if fees is not None:
        out["fees"] = float(fees)
    return out


def sharpe(r):
    sd = r.std()
    return r.mean() / sd * np.sqrt(HOURS_PER_YEAR) if sd > 0 else 0.0


def block_bootstrap_delta_sharpe(ra, rb, block=168, reps=1000, seed=0):
    """Stationary block bootstrap p-value for H0: sharpe(ra) <= sharpe(rb)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(ra), np.asarray(rb)
    n = len(a)
    obs = sharpe(pd.Series(a)) - sharpe(pd.Series(b))
    diffs = np.empty(reps)
    pos = np.arange(n)
    for i in range(reps):
        # stationary bootstrap: new block starts with prob 1/block; offset within block grows by 1
        new_block = rng.random(n) < 1.0 / block
        new_block[0] = True
        block_id = np.cumsum(new_block) - 1
        starts = rng.integers(n, size=block_id[-1] + 1)
        first = np.flatnonzero(new_block)
        idx = (starts[block_id] + pos - first[block_id]) % n
        diffs[i] = sharpe(pd.Series(a[idx])) - sharpe(pd.Series(b[idx]))
    # centered bootstrap: distribution of (diff* - obs) approximates null around 0
    p = float(np.mean(diffs - obs >= obs))
    return float(obs), p
