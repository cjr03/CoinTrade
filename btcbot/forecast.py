"""Price forecast curves for the chart: a median path and 50% / 80% bands from now out to a week.

    python -m btcbot.forecast --fit         # fit bands per coin -> artifacts/forecast_<COIN>.json

The bands come from volatility, not from a price model: at each horizon h (1 minute .. 1 week) the
spread is sqrt(h) times a blend of the last hour's 1-minute volatility and the last month's, with the
blend weight fitted per horizon (short horizons lean on the last hour, long ones on the month). Real
returns have fatter tails than a bell curve, so the band edges are the empirical quantiles of
standardized returns. Fitted on the first 70% of 180 days of 1-minute candles, then checked on the last
30%: the reported coverage is how often the actual price ended inside each band.

The median tilts with what the models lean towards, converted to a drift the way a calibrated
probability implies one (mu = inverse-normal(P(up)) * sigma): the 15-minute call model over its
window (right 54-56% of the time on unseen months), and the trend rule's long-run average drift in its
current state (2019 on). The bet model is left out: its own validation lost money on every coin, so its
probabilities have not earned a place in a price forecast. Every test in this project says these
leans are small; the curve shows them honestly.
"""
import argparse
import json
import math
import threading
import time
from statistics import NormalDist

import numpy as np
import pandas as pd

from .coins import COINS
from .data import load
from .features import regularize
from .kraken import public as kraken_public

ART_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "artifacts"
HORIZONS = (1, 5, 15, 30, 60, 120, 240, 480, 720, 1440, 2880, 4320, 7200, 10080)  # minutes
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
NORM = NormalDist()


def vol_inputs(m):
    """Per-minute volatility from the last hour (1-minute moves) and the last 30 days (hourly moves)."""
    lr = np.log(m["close"]).diff()
    short = lr.rolling(60).std()
    hourly = np.log(m["close"].resample("60min").last()).diff()
    long = (hourly.rolling(720).std() / math.sqrt(60)).reindex(m.index, method="ffill")
    return short, long


# --- fitting -----------------------------------------------------------------------------------
def fit_coin(coin, verbose=True):
    m = regularize(load(COINS[coin]["product"], "ONE_MINUTE"), "1min")
    short, long = vol_inputs(m)
    close = m["close"]
    origins = m.index[(m.index.minute % 15 == 0)]
    origins = origins[(origins >= m.index[0] + pd.Timedelta(days=31)) & (origins <= m.index[-1] - pd.Timedelta(minutes=10080))]
    cut = origins[int(len(origins) * 0.7)]
    out = {"coin": coin, "horizons": list(HORIZONS), "qs": list(QS), "fit": {}, "test_coverage": {},
           "train": [str(origins[0]), str(cut)], "test": [str(cut), str(origins[-1])]}
    s2, l2 = short.reindex(origins).values ** 2, long.reindex(origins).values ** 2
    c0 = close.reindex(origins).values
    for h in HORIZONS:
        r = np.log(close.reindex(origins + pd.Timedelta(minutes=h)).values / c0)
        ok = np.isfinite(r) & np.isfinite(s2) & np.isfinite(l2) & (s2 > 0) & (l2 > 0)
        tr, te = ok & (origins < cut), ok & (origins >= cut)
        best = None
        for w in WEIGHTS:  # Gaussian likelihood picks the blend of last-hour and last-month volatility
            var = h * (w * s2[tr] + (1 - w) * l2[tr])
            nll = np.mean(0.5 * np.log(var) + r[tr] ** 2 / (2 * var))
            if best is None or nll < best[0]:
                best = (nll, float(w))
        w = best[1]
        sd = np.sqrt(h * (w * s2 + (1 - w) * l2))
        z_tr = r[tr] / sd[tr]
        q = [float(np.quantile(z_tr, x)) for x in QS]
        z_te = r[te] / sd[te]
        cov80 = float(np.mean((z_te >= q[0]) & (z_te <= q[4])))
        cov50 = float(np.mean((z_te >= q[1]) & (z_te <= q[3])))
        out["fit"][str(h)] = {"w_short": w, "zq": q}
        out["test_coverage"][str(h)] = {"80": cov80, "50": cov50, "n": int(te.sum())}
        if verbose:
            print(f"{coin} {h:>5}m  w_short {w:.2f}  z10..z90 {q[0]:+.2f} {q[1]:+.2f} {q[2]:+.2f} {q[3]:+.2f} {q[4]:+.2f}"
                  f"  test: 80% band held {cov80:.0%}, 50% band held {cov50:.0%}  (n={int(te.sum())})")
    # the trend rule's long-run drift in each state (daily close vs its 50-day average), from hourly history
    h1 = regularize(load(COINS[coin]["product"], "ONE_HOUR"), "h")
    daily = h1["close"].resample("D").last()
    up = (daily > daily.rolling(50).mean()).shift(1)
    fwd = np.log(daily.shift(-1) / daily)
    d = pd.DataFrame({"up": up, "fwd": fwd}).dropna()
    d = d[d.index >= pd.Timestamp("2019-01-01", tz="UTC")]
    out["trend_drift_per_day"] = {"up": float(d.loc[d["up"] == True, "fwd"].mean()),  # noqa: E712
                                  "down": float(d.loc[d["up"] == False, "fwd"].mean())}  # noqa: E712
    (ART_DIR / f"forecast_{coin}.json").write_text(json.dumps(out, indent=1))
    if verbose:
        t = out["trend_drift_per_day"]
        print(f"{coin} trend drift per day: up {t['up']:+.3%}, down {t['down']:+.3%}")
    return out


# --- live curves -------------------------------------------------------------------------------
class Forecaster:
    """Builds each coin's curve from the fitted bands, live volatility from Kraken and the models' leans."""

    def __init__(self, coins):
        self.fits = {}
        for c in coins:
            path = ART_DIR / f"forecast_{c}.json"
            if path.exists():
                self.fits[c] = json.loads(path.read_text())
        self._vol = {}  # coin -> (fetched, short, long)
        self.lock = threading.Lock()

    def vol(self, coin):
        with self.lock:
            hit = self._vol.get(coin)
            if hit and time.time() - hit[0] < 60:
                return hit[1], hit[2]
        c = COINS[coin]
        m1 = kraken_public("OHLC", pair=c["kraken_pair"], interval=1, since=int(time.time() - 3900))[c["kraken_key"]]
        closes = np.array([float(r[4]) for r in m1])[-61:]
        short = float(np.diff(np.log(closes)).std())
        long = hit[2] if hit and time.time() - hit[0] < 900 else None
        if long is None:
            h1 = kraken_public("OHLC", pair=c["kraken_pair"], interval=60)[c["kraken_key"]]
            hc = np.array([float(r[4]) for r in h1])
            long = float(np.diff(np.log(hc)).std() / math.sqrt(60))
        with self.lock:
            self._vol[coin] = (time.time(), short, long)
        return short, long

    def curve(self, coin, leans):
        """leans: [(kind, P(up), horizon minutes)] from the models, plus ('trend', 'up'|'down', None)."""
        fit = self.fits[coin]
        short, long = self.vol(coin)
        hs = fit["horizons"]
        sd = {h: math.sqrt(h * (fit["fit"][str(h)]["w_short"] * short ** 2 + (1 - fit["fit"][str(h)]["w_short"]) * long ** 2))
              for h in hs}

        def sd_at(h):  # volatility at any horizon, interpolated on log-minutes
            return float(np.interp(math.log(h), [math.log(x) for x in hs], [sd[x] for x in hs]))

        parts = []
        for kind, p, H in leans:
            if kind == "trend":
                per_day = fit["trend_drift_per_day"]["up" if p == "up" else "down"]
                parts.append({"kind": "trend", "state": p, "per_day": per_day})
            elif p is not None:
                mu = NORM.inv_cdf(min(0.98, max(0.02, p))) * sd_at(H)
                parts.append({"kind": kind, "p_up": p, "H": H, "mu": mu})

        def drift(h):
            total = 0.0
            for part in parts:
                if part["kind"] == "trend":
                    total += part["per_day"] * h / 1440
                else:
                    total += part["mu"] * min(h / part["H"], 1.0)  # a model says nothing past its horizon
            return total

        pts = []
        for h in hs:
            raw = fit["fit"][str(h)]["zq"]
            zq = [z - raw[2] for z in raw]  # centre on the models' drift, not on the fit period's own trend
            if h >= 1440:  # a day or more: too few independent samples to trust a lopsided shape; keep the width only
                w80, w50 = (zq[4] - zq[0]) / 2, (zq[3] - zq[1]) / 2
                zq = [-w80, -w50, 0.0, w50, w80]
            mu = drift(h)
            pts.append({"h": h, "mu": mu, **{f"q{int(q * 100)}": mu + sd[h] * z for q, z in zip(QS, zq)}})
        return {"coin": coin, "asof": time.time(), "points": pts, "parts": parts, "sigma1m": short,
                "sigma1m_month": long, "coverage": fit["test_coverage"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--coins", default=",".join(COINS))
    a = ap.parse_args()
    if a.fit:
        for c in a.coins.split(","):
            fit_coin(c)


if __name__ == "__main__":
    main()
