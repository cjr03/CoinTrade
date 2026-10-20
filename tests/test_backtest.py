import numpy as np
import pandas as pd
import pytest

from btcbot.backtest import Policy, simulate
from btcbot.features import build_features


def bars(closes, lows=None, highs=None):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC")
    c = pd.Series(closes, idx, dtype=float)
    l = pd.Series(lows if lows is not None else c - 1, idx, dtype=float)
    h = pd.Series(highs if highs is not None else c + 1, idx, dtype=float)
    return c, h, l


def test_round_trip_pnl_matches_hand_calculation():
    c, h, l = bars([100, 100, 110, 110, 110])
    pred = np.array([1.0, np.nan, np.nan, np.nan, np.nan])
    pol = Policy(maker_fee=0.006, half_size=False, min_hold=1, stop_sigma=99)
    r = simulate(c, h, l, pred, np.full(5, 0.01), pol)
    qty = 100 * (1 - 0.006) / 100        # maker buy at 100, filled on bar 1 (low 99 < 100)
    expect = qty * 110 * (1 - 0.006)     # signal exit decided at bar 2 close, maker fill on bar 3
    assert r.equity.iloc[-1] == pytest.approx(expect)
    assert [t[0] for t in r.trades] == ["buy", "sell"]


def test_unfilled_entry_is_abandoned_and_marked_flat():
    c, h, l = bars([100, 101, 102, 103], lows=[99, 100.5, 101.5, 102.5])  # never trades below 100
    pred = np.array([1.0, np.nan, np.nan, np.nan])
    r = simulate(c, h, l, pred, np.full(4, 0.01), Policy(half_size=False))
    assert r.trades == []
    assert (r.equity == 100).all() and (r.position == 0).all()


def test_equity_is_flat_until_fill_bar():
    c, h, l = bars([100, 90, 99, 110, 120], lows=[99.5, 100, 98, 109, 119])  # fills only at bar 2
    pred = np.array([1.0] + [np.nan] * 4)
    r = simulate(c, h, l, pred, np.full(5, 0.01), Policy(half_size=False, min_hold=1, stop_sigma=99))
    assert r.equity.iloc[1] == 100 and r.position.iloc[1] == 0
    assert r.position.iloc[2] == pytest.approx(1.0)


def test_stop_exits_with_taker_fee():
    c, h, l = bars([100, 100, 80, 80])
    pred = np.array([1.0, 1.0, 1.0, 1.0])
    pol = Policy(maker_fee=0.006, taker_fee=0.012, half_size=False, stop_sigma=1.0, taker_slip=0.0, kill_dd=1.0)
    r = simulate(c, h, l, pred, np.full(4, 0.05), pol)
    assert r.trades[-1][5] == "taker:stop"


def test_features_are_causal():
    rng = np.random.default_rng(0)
    n = 24 * 400
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    c = 10000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"open": c, "high": c * 1.002, "low": c * 0.998, "close": c,
                       "volume": rng.uniform(1, 10, n)}, idx)
    full, _ = build_features(df)
    cut = idx[-200]
    trunc, _ = build_features(df.loc[:cut])
    pd.testing.assert_series_equal(full.loc[cut], trunc.loc[cut], check_names=False)
