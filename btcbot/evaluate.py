"""Turn walk-forward predictions into net-of-fee out-of-sample backtests + go-live gate.

Policy thresholds (entry/exit) for test fold k are chosen using ONLY the validation
segments of folds 0..k (each of which ends before test fold k starts).
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import Policy, block_bootstrap_delta_sharpe, metrics, simulate
from .train import OUT, load_dataset

MARGINS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)   # absolute edge over round-trip cost
QUANTILES = (0.90, 0.95, 0.98)                    # or: trade the top-q of past val predictions
FEES = {
    "base_0.60_1.20": (0.006, 0.012),           # Coinbase Advanced lowest tier (worst case)
    "optimistic_0.40_0.60": (0.004, 0.006),     # Coinbase Advanced $10K+ tier
    "kraken_0.40_0.80": (0.004, 0.008),         # Kraken Pro, read live from this account's TradeVolume
}
EXCHANGE = {"base_0.60_1.20": "coinbase", "optimistic_0.40_0.60": "coinbase", "kraken_0.40_0.80": "kraken"}
BIG = 1.0


def _sim(bars, pred, sigma_h, vp, policy, entry, exit_):
    return simulate(bars["close"], bars["high"], bars["low"], pred, sigma_h, policy, vp,
                    entry_thr=entry, exit_thr=exit_)


def _sharpe(r):
    return r.mean() / r.std() * np.sqrt(8760) if r.std() > 0 else 0.0


def candidates(policy, past_preds):
    out = [("abs", m, policy.round_trip + m, -policy.maker_fee) for m in MARGINS]
    for q in QUANTILES:
        out.append(("q", q, float(np.nanquantile(past_preds, q)), float(np.nanquantile(past_preds, 0.5))))
    return out


def select_thresholds(oos, model, bars, f, policy):
    """For test fold k pick (entry, exit) thresholds maximizing Sharpe over pooled val folds 0..k.
    Returns {k: (label, entry, exit)}; entry=NaN means stay flat."""
    H = policy.horizon
    chosen = {}
    for k in sorted(oos.fold.unique()):
        v_all = oos[(oos.fold <= k) & (oos.segment == "val")]
        best, best_s = None, 0.0
        for kind, val, entry, exit_ in candidates(policy, v_all[model].values):
            rets = []
            for j, v in v_all.groupby("fold"):
                b = bars.loc[v.index]
                res = _sim(b, v[model].values, (f["sigma"].loc[v.index] * np.sqrt(H)).values,
                           f["vol_pctile"].loc[v.index].values, policy, entry, exit_)
                rets.append(res.equity.pct_change().fillna(0.0))
            sc = _sharpe(pd.concat(rets))
            if sc > best_s:
                best, best_s = (f"{kind}={val}", entry, exit_), sc
        chosen[k] = best or ("flat", np.nan, np.nan)
        chosen[k] = (*chosen[k], best_s)
    return chosen


def run_strategy(oos, model, bars, f, policy, chosen=None):
    chosen = chosen or select_thresholds(oos, model, bars, f, policy)
    t = oos[oos.segment == "test"]
    entry = np.array([chosen[k][1] for k in t.fold])
    # flat folds keep the default exit rule so a carried-over position is managed as validated
    exit_ = np.array([chosen[k][2] if np.isfinite(chosen[k][2]) else -policy.maker_fee for k in t.fold])
    b = bars.loc[t.index]
    res = _sim(b, t[model].values, (f["sigma"].loc[t.index] * np.sqrt(policy.horizon)).values,
               f["vol_pctile"].loc[t.index].values, policy, entry, exit_)
    return res, chosen


def buy_and_hold(bars, fee):
    c = bars["close"]
    qty = 100.0 * (1 - fee) / c.iloc[0]
    return qty * c


def momentum(bars, policy, full_close, sma_days=50):
    """Long when the last completed daily close is above its 50d SMA (checked hourly).
    The SMA is computed on full history so the test window starts warmed up."""
    daily = full_close.resample("D").last()
    sig = (daily > daily.rolling(sma_days).mean()).shift(1)  # only completed days
    s = sig.reindex(bars.index, method="ffill").fillna(False).values.astype(bool)
    pred = np.where(s, BIG, -BIG)
    pol = replace(policy, half_size=False, stop_sigma=np.inf, time_stop_mult=10**9, min_hold=1,
                  entry_margin=0.0, kill_dd=1.0)
    return simulate(bars["close"], bars["high"], bars["low"], pred, np.full(len(s), np.nan), pol)


def opening_buys(trades):
    return [tr for i, tr in enumerate(trades) if tr[0] == "buy" and (i == 0 or trades[i - 1][0] != "buy")]


def random_entries(bars, policy, n_entries, exposure, seeds=200):
    """Null: same number of entries and same time-in-market as the strategy, random timing."""
    n = len(bars)
    n_entries = max(n_entries, 1)
    hold = int(max(1, round(exposure * n / n_entries)))
    pol = replace(policy, half_size=False, stop_sigma=np.inf, entry_margin=0.0, kill_dd=1.0,
                  min_hold=hold, time_stop_mult=10**9)
    sharpes = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        starts = np.sort(rng.choice(np.arange(0, n - hold - 3, hold + 3), size=min(n_entries, (n - hold - 3) // (hold + 3)), replace=False))
        pred = np.full(n, -BIG)
        for st in starts:
            pred[st] = BIG
            pred[st + 1:st + 3] = BIG  # keep signal on while the maker order is working
        res = simulate(bars["close"], bars["high"], bars["low"], pred, np.full(n, np.nan), pol)
        sharpes.append(metrics(res.equity)["sharpe"])
    return np.array(sharpes)


def fold_table(equity, oos):
    """Per-fold returns that compound exactly to the total (base = previous fold's last bar)."""
    t = oos[oos.segment == "test"]
    r = equity.pct_change().fillna(0.0)
    rows = []
    for k, g in t.groupby("fold"):
        rows.append({"fold": int(k), "start": str(g.index.min().date()),
                     "ret": float((1 + r.loc[g.index]).prod() - 1)})
    return pd.DataFrame(rows)


def evaluate_config(H, model, oos, bars, f, fee_name, fees, n_random=200):
    policy = Policy(maker_fee=fees[0], taker_fee=fees[1], horizon=H, min_hold=min(H, 24))
    res, chosen = run_strategy(oos, model, bars, f, policy)
    t = oos[oos.segment == "test"]
    b = bars.loc[t.index]
    bh = buy_and_hold(b, fees[0])
    mom = momentum(b, policy, bars["close"])
    m_s, m_bh, m_mom = metrics(res.equity, res.trades, res.fees), metrics(bh), metrics(mom.equity, mom.trades, mom.fees)
    m_s["exposure"] = float(res.position.mean())
    m_mom["exposure"] = float(mom.position.mean())
    r_s, r_bh, r_mom = (x.pct_change().fillna(0) for x in (res.equity, bh, mom.equity))
    d_bh, p_bh = block_bootstrap_delta_sharpe(r_s, r_bh, reps=500)
    d_mom, p_mom = block_bootstrap_delta_sharpe(r_s, r_mom, reps=500)
    n_open = len(opening_buys(res.trades))
    rnd = random_entries(b, policy, n_open, m_s["exposure"], seeds=n_random) if n_open else np.array([0.0])
    ft = fold_table(res.equity, oos)
    ft_bh = fold_table(bh, oos)
    t_dn = oos[(oos.segment == "test")].dropna(subset=["fwd"])
    ic_by_fold = t_dn.groupby("fold").apply(lambda g: g[model].corr(g["fwd"], method="spearman"))
    gate = {
        "beats_bh_sharpe_p<0.10": bool(d_bh > 0 and p_bh < 0.10),
        "beats_momentum_sharpe_p<0.10": bool(d_mom > 0 and p_mom < 0.10),
        "positive_in_>=60%_folds": bool((ft.ret > 0).mean() >= 0.60),
        "positive_in_2022": bool((1 + ft[ft.start.str.startswith("2022")].ret).prod() > 1),
    }
    return {
        "H": H, "model": model, "fees": fee_name,
        "strategy": m_s, "buy_hold": m_bh, "momentum": m_mom,
        "delta_sharpe_vs_bh": d_bh, "p_vs_bh": p_bh, "delta_sharpe_vs_mom": d_mom, "p_vs_mom": p_mom,
        "random_entry_p": float(np.mean(rnd >= m_s["sharpe"])), "n_entries": n_open,
        "ic_mean": float(ic_by_fold.mean()), "ic_tstat": float(ic_by_fold.mean() / ic_by_fold.std() * np.sqrt(len(ic_by_fold))),
        "folds_positive": float((ft.ret > 0).mean()), "folds_beat_bh": float((ft.ret > ft_bh.ret).mean()),
        "fold_returns": ft.to_dict("records"), "fold_returns_bh": ft_bh.ret.tolist(),
        "chosen": {int(k): v[0] for k, v in chosen.items()},
        "val_sharpe": float(chosen[max(chosen)][3]),
        "gate": gate, "gate_pass": all(gate.values()),
        "_policy": policy, "_chosen": chosen,
    }, res, bh, mom


def export_policy(r, path):
    """Write the live policy from the latest fold's validation-chosen thresholds."""
    pol, chosen = r["_policy"], r["_chosen"]
    label, entry, exit_, val_sr = chosen[max(chosen)]
    out = {
        "model": r["model"], "horizon": r["H"], "threshold_rule": label, "val_sharpe": val_sr,
        "entry_thr": None if not np.isfinite(entry) else float(entry),  # None => stay flat
        "exit_thr": float(exit_) if np.isfinite(exit_) else -pol.maker_fee,
        "round_trip": pol.round_trip,
        "fees_assumed": [pol.maker_fee, pol.taker_fee],
        "entry_fill_hours": pol.fill_bars, "exit_fill_hours": pol.exit_fill_bars,
        "min_hold": pol.min_hold, "time_stop_mult": pol.time_stop_mult, "stop_sigma": pol.stop_sigma,
        "half_size": pol.half_size, "kill_dd": pol.kill_dd, "gate_pass": r["gate_pass"],
        "exchange": EXCHANGE[r["fees"]],
    }
    path.write_text(json.dumps(out, indent=1))
    return out


def export_momentum_policy(path, fee_name, sma_days=50, kill_dd=0.35):
    """Policy for the daily SMA trend filter (the pre-registered fallback when the ML gate fails)."""
    maker, taker = FEES[fee_name]
    out = {
        "model": "momentum", "sma_days": sma_days, "horizon": 24, "threshold_rule": f"close>SMA{sma_days}",
        "entry_thr": 0.0, "exit_thr": 0.0, "round_trip": (1 + maker) / (1 - maker) - 1,
        "fees_assumed": [maker, taker], "entry_fill_hours": 2, "exit_fill_hours": 2,
        "min_hold": 0, "time_stop_mult": 10**9, "stop_sigma": 1e9, "half_size": False,
        "kill_dd": kill_dd, "gate_pass": False, "exchange": EXCHANGE[fee_name],
    }
    path.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="12,24,72")
    ap.add_argument("--models", default="lgbm,ridge,lgbm_tb")
    ap.add_argument("--tag", default="", help="suffix of oos files, e.g. _no5m")
    ap.add_argument("--export", default="", help="H,model,fee_name to write artifacts/policy.json")
    ap.add_argument("--fees", default=",".join(FEES), help="fee scenarios to evaluate")
    ap.add_argument("--export-momentum", default="", help="fee_name: write the SMA-50 trend policy instead")
    a = ap.parse_args()
    if a.export_momentum:
        print(json.dumps(export_momentum_policy(OUT / "policy.json", a.export_momentum), indent=1))
        raise SystemExit
    f, bars = load_dataset(use_5m=a.tag != "_no5m")
    report, curves = [], {}
    for H in map(int, a.horizons.split(",")):
        oos = pd.read_parquet(OUT / f"oos_H{H}{a.tag}.parquet")
        for model in a.models.split(","):
            for fee_name, fees in FEES.items():
                if fee_name not in a.fees.split(","):
                    continue
                if a.export and a.export != f"{H},{model},{fee_name}":
                    continue
                r, res, bh, mom = evaluate_config(H, model, oos, bars, f, fee_name, fees)
                if a.export:
                    print(json.dumps(export_policy(r, OUT / "policy.json"), indent=1))
                report.append({k: v for k, v in r.items() if not k.startswith("_")})
                curves[f"H{H}_{model}_{fee_name}"] = res.equity
                curves[f"buy_hold_{fee_name}"] = bh
                curves[f"momentum_{fee_name}_H{H}"] = mom.equity
                s = r["strategy"]
                print(f"H={H:3d} {model:8s} {fee_name:22s} | strat CAGR {s['cagr']:+.1%} SR {s['sharpe']:.2f} "
                      f"DD {s['max_dd']:.1%} RT/yr {s['round_trips_per_year']:.0f} exp {s['exposure']:.0%} "
                      f"| B&H CAGR {r['buy_hold']['cagr']:+.1%} SR {r['buy_hold']['sharpe']:.2f} DD {r['buy_hold']['max_dd']:.0%} "
                      f"| MOM CAGR {r['momentum']['cagr']:+.1%} SR {r['momentum']['sharpe']:.2f} | IC {r['ic_mean']:.3f} (t={r['ic_tstat']:.1f}) "
                      f"| p_bh {r['p_vs_bh']:.2f} p_mom {r['p_vs_mom']:.2f} p_rand {r['random_entry_p']:.2f} "
                      f"| folds+ {r['folds_positive']:.0%} | val SR {r['val_sharpe']:.2f} "
                      f"| GATE {'PASS' if r['gate_pass'] else 'fail'}", flush=True)
    if not a.export:
        with open(OUT / f"report{a.tag}_H{a.horizons.replace(',', '-')}{'_' + a.fees if a.fees != ','.join(FEES) else ''}.json", "w") as fh:
            json.dump(report, fh, indent=1, default=float)
        pd.DataFrame(curves).to_parquet(OUT / f"equity_curves{a.tag}_H{a.horizons.replace(',', '-')}{'_' + a.fees if a.fees != ','.join(FEES) else ''}.parquet")
