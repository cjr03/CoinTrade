"""Per-coin evidence for the trend policy: SMA-50 trend filter vs buy & hold at Kraken fees.

    python -m btcbot.coin_eval        # writes artifacts/coins_report.json

Each coin is tested on its longest gap-free stretch of Coinbase hourly data from 2019 on (XRP was
delisted from Coinbase 2021-01 .. 2023-07, so only its post-relisting stretch counts). Portfolio
mixes are compared on the window where every coin has data.
"""
import json

import numpy as np
import pandas as pd

from .backtest import Policy, metrics, simulate
from .coins import COINS
from .data import load
from .features import regularize
from .train import OUT

FEES = (0.004, 0.008)  # Kraken Pro tier 1 (this account)


def clean_hourly(coin):
    """Longest stretch from 2019 on without a gap of more than 3 days, as a regular hourly series."""
    df = load(COINS[coin]["product"], "ONE_HOUR")
    df = df[df.index >= pd.Timestamp("2018-09-01", tz="UTC")]  # 50-day warm-up before 2019
    gaps = df.index.to_series().diff() > pd.Timedelta(days=3)
    seg = gaps.cumsum()
    longest = seg.value_counts().idxmax()
    return regularize(df[seg == longest], "h")


def trend_curve(bars, sma_days=50, kill_dd=0.35):
    daily = bars["close"].resample("D").last()
    sig = (daily > daily.rolling(sma_days).mean()).shift(1)
    s = sig.reindex(bars.index, method="ffill").fillna(False).values.astype(bool)
    pol = Policy(maker_fee=FEES[0], taker_fee=FEES[1], half_size=False, stop_sigma=np.inf,
                 time_stop_mult=10**9, min_hold=1, kill_dd=kill_dd)
    res = simulate(bars["close"], bars["high"], bars["low"], np.where(s, 1.0, -1.0), np.full(len(s), np.nan), pol)
    return res.equity, res


def hold_curve(bars):
    return 100.0 * (1 - FEES[0]) / bars["close"].iloc[0] * bars["close"]


def summarize(eq, trades=None, fees=None):
    m = metrics(eq, trades, fees)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if v is not None}


def main():
    report, curves = {}, {}
    for coin in COINS:
        bars = clean_hourly(coin)
        test = bars[bars.index >= max(bars.index[0] + pd.Timedelta(days=60), pd.Timestamp("2019-01-01", tz="UTC"))]
        eq, res = trend_curve(bars)
        eq = eq.loc[test.index] / eq.loc[test.index].iloc[0] * 100
        hold = hold_curve(test)
        curves[coin] = {"trend": eq, "hold": hold}
        report[coin] = {"start": str(test.index[0].date()), "end": str(test.index[-1].date()),
                        "trend": summarize(eq, [t for t in res.trades if t[1] >= test.index[0]], None),
                        "hold": summarize(hold)}
        r = report[coin]
        print(f"{coin}  {r['start']} .. {r['end']}  trend CAGR {r['trend']['cagr']:+.1%} SR {r['trend']['sharpe']:.2f} "
              f"DD {r['trend']['max_dd']:.0%}  RT/yr {r['trend'].get('round_trips_per_year', 0):.1f} | "
              f"hold CAGR {r['hold']['cagr']:+.1%} SR {r['hold']['sharpe']:.2f} DD {r['hold']['max_dd']:.0%}")

    # portfolio mixes on the window where every coin has clean data (equal weight, no rebalancing)
    start = max(c["trend"].index[0] for c in curves.values())
    end = min(c["trend"].index[-1] for c in curves.values())
    window = {}
    for coin, c in curves.items():
        for k in ("trend", "hold"):
            s = c[k].loc[start:end]
            window[(coin, k)] = s / s.iloc[0] * 100
    mixes = {
        "BTC trend only": [("BTC", "trend")],
        "BTC+ETH trend": [("BTC", "trend"), ("ETH", "trend")],
        "all 4 trend": [(c, "trend") for c in COINS],
        "all 4 hold": [(c, "hold") for c in COINS],
        "BTC hold only": [("BTC", "hold")],
    }
    report["_portfolios"] = {"start": str(start.date()), "end": str(end.date()), "mixes": {}}
    print(f"\nportfolios {start.date()} .. {end.date()} (equal weight, $100 split at start, no rebalancing)")
    for name, legs in mixes.items():
        eq = sum(window[leg] for leg in legs) / len(legs)
        m = summarize(eq)
        report["_portfolios"]["mixes"][name] = m
        print(f"  {name:<16} CAGR {m['cagr']:+.1%}  Sharpe {m['sharpe']:.2f}  max DD {m['max_dd']:.0%}  final ${m['final']:,.0f}")
    (OUT / "coins_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
