import math
import time

import pytest

from btcbot.forecast import HORIZONS, QS, Forecaster
from btcbot.tui import fc_at, render_chart


def fitted(trend_up=0.0023):
    return {"horizons": list(HORIZONS), "qs": list(QS), "trend_drift_per_day": {"up": trend_up, "down": -0.0005},
            "fit": {str(h): {"w_short": 0.5, "zq": [-1.2 + 0.05, -0.55 + 0.05, 0.05, 0.55 + 0.05, 1.2 + 0.05]} for h in HORIZONS},
            "test_coverage": {}}


def forecaster(fit=None):
    f = Forecaster([])
    f.fits["BTC"] = fit or fitted()
    f.vol = lambda coin: (0.0004, 0.0005)  # no network: fixed per-minute volatility
    return f


def test_curve_tilts_with_the_models_and_is_not_constant():
    f = forecaster()
    flat = f.curve("BTC", [])
    assert all(abs(p["mu"]) < 1e-12 and p["q50"] == pytest.approx(p["mu"]) for p in flat["points"])  # no leans: centred
    c = f.curve("BTC", [("15m model", 0.6, 15), ("bet model", 0.61, 1440), ("trend", "up", None)])
    mus = [p["mu"] for p in c["points"]]
    assert all(b > a for a, b in zip(mus, mus[1:]))  # every model leans up: the median climbs with the horizon
    week = c["points"][-1]
    assert week["mu"] == pytest.approx(0.0023 * 7 + c["parts"][0]["mu"] + c["parts"][1]["mu"])  # short leans stop at their horizon
    down = f.curve("BTC", [("bet model", 0.4, 240)])["points"]
    assert down[-1]["mu"] < 0


def test_bands_widen_with_time_and_are_ordered():
    pts = forecaster().curve("BTC", [])["points"]
    for p in pts:
        assert p["q10"] < p["q25"] < p["q50"] < p["q75"] < p["q90"]
    widths = [p["q90"] - p["q10"] for p in pts]
    assert all(b > a for a, b in zip(widths, widths[1:]))


def test_interpolation_between_horizons():
    fc = forecaster().curve("BTC", [("trend", "up", None)])
    a, b = fc["points"][2], fc["points"][3]  # 15 and 30 minutes
    mid = fc_at(fc, math.sqrt(a["h"] * b["h"]))  # halfway on log-time
    assert mid["q90"] == pytest.approx((a["q90"] + b["q90"]) / 2)
    assert fc_at(fc, 0)["q90"] == 0 and fc_at(fc, 10 ** 6)["q90"] == fc["points"][-1]["q90"]


@pytest.mark.parametrize("view", ["10m", "1h", "1d", "1w"])
def test_chart_draws_the_forecast_at_every_view(view):
    fc = forecaster().curve("BTC", [("trend", "up", None)])
    now = time.time()
    data = {"kind": "candles", "secs": 60, "bars": [[int(now - (60 - i) * 60), 100, 100.2, 99.8, 100.1, 1] for i in range(60)]}
    t = render_chart(120, 30, view, data, [[now, 100.1]], 100.1, {"state": {}, "policy": None}, [], fc)
    assert "80%" in t.plain and "▒" in t.plain and "━" in t.plain
