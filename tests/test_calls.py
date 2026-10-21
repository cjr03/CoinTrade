import numpy as np
import pandas as pd
import pytest

from btcbot.calls import Caller, features, live_odds, score


def candles(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": rng.uniform(1, 2, n)},
                        index=idx)


def test_features_never_see_the_future():
    df, btc = candles(), candles(seed=1)
    f, _ = features(df, btc)
    cut = 400
    shocked, shocked_btc = df.copy(), btc.copy()
    shocked.iloc[cut + 1:, :] *= 3  # change everything after bar `cut`
    shocked_btc.iloc[cut + 1:, :] *= 3
    g, _ = features(shocked, shocked_btc)
    pd.testing.assert_frame_equal(f.iloc[:cut + 1], g.iloc[:cut + 1])


def test_target_is_the_next_bar():
    df = candles()
    _, y = features(df)
    assert y.iloc[10] == pytest.approx(np.log(df["close"].iloc[11] / df["close"].iloc[10]))


def test_score_counts_only_confident_calls_and_charges_fees():
    p = np.array([0.7, 0.3, 0.52, 0.6])
    ys = np.log(np.array([1.01, 0.99, 1.02, 0.995]))  # up, down, up, down
    s = score(p, ys, 0.6)
    assert s["calls"] == 3 and s["hit_rate"] == 2 / 3  # 0.52 is below the bar; 0.6 called up but it fell
    assert round(s["after_fees_bp"]) == round(s["edge_bp"] - 80)


def test_live_call_is_scored_once_its_bar_closes(tmp_path):
    c = Caller([], tmp_path)
    prev = {"bar": "2026-01-01 00:00:00+00:00", "from": "2026-01-01 00:15:00+00:00",
            "until": "2026-01-01 00:30:00+00:00", "dir": "down", "p_up": 0.4, "price": 100.0}  # no Kraken start: Coinbase
    close = pd.Series([99.0], index=[pd.Timestamp("2026-01-01 00:15:00+00:00")])
    c._score("BTC", prev, close)
    c._score("BTC", prev, close)  # a second refresh must not count it twice
    h = c.s["history"]["BTC"]
    assert len(h) == 1 and h[0]["hit"] is True and round(h[0]["move_bp"]) == -100
    c._score("BTC", {**prev, "dir": None, "bar": "x"}, close)  # 'no call' is never scored
    assert len(c.s["history"]["BTC"]) == 1


def test_live_odds_start_at_the_models_lean_and_settle_at_the_end():
    assert live_odds(100, 100, 0.0003, 0.0, 900) == 0.5          # flat at the open, no lean: a coin flip
    assert live_odds(100, 100, 0.0003, 0.3, 900) > 0.6           # the model's opening lean shows at the open
    assert live_odds(100, 100.01, 0.0003, 0.0, 0) == 1.0         # window over: settled on the price vs the start
    assert live_odds(100, 100, 0.0003, 0.0, 0) == 1.0            # a tie settles up ("at least"), as on Kalshi
    assert live_odds(100, 99.99, 0.0003, 0.0, 0) == 0.0


def test_live_odds_firm_up_as_time_runs_out_and_stay_capped():
    early, late = live_odds(100, 100.03, 0.0003, 0.0, 800), live_odds(100, 100.03, 0.0003, 0.0, 90)
    assert 0.5 < early < late <= 0.98                          # same lead, less time left: surer, never certain
    assert live_odds(100, 95, 0.0003, 0.0, 450) == 0.02          # capped


def test_settlement_average_shrinks_the_last_minute():
    from btcbot.calls import tau_eff
    assert tau_eff(600) == 10 - 2 / 3                          # ten minutes left: the average trims a third of a minute
    assert tau_eff(60) == 1 / 3 and tau_eff(30) == 0.5 ** 3 / 3   # continuous at one minute, then shrinks fast


def test_terminal_and_server_odds_agree():
    import time
    from datetime import datetime, timezone
    from btcbot.calls import A0, A1
    from btcbot.tui import live_odds as tui_odds
    now = time.time()
    call = {"start": 100.0, "sigma1m": 0.0003, "z0": 0.2, "a0": A0, "a1": A1,
            "from": datetime.fromtimestamp(now - 300, timezone.utc).isoformat(),
            "until": datetime.fromtimestamp(now + 600, timezone.utc).isoformat()}
    for price in (99.98, 100.0, 100.01, 100.05):
        assert tui_odds(call, price, now) == pytest.approx(live_odds(100.0, price, 0.0003, 0.2, 600))


def test_kalshi_ticker_names_the_window_by_its_new_york_close():
    from btcbot.calls import kalshi_ticker
    assert kalshi_ticker("BTC", pd.Timestamp("2026-10-21 05:30:00+00:00")) == "KXBTC15M-26OCT210130-30"  # EDT
    assert kalshi_ticker("XRP", pd.Timestamp("2026-12-01 17:00:00+00:00")) == "KXXRP15M-26DEC011200-00"  # EST
