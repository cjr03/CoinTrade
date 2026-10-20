"""Strictly causal hourly feature matrix built from 1h + 5m Coinbase candles.

Convention: a row indexed at hour t describes the bar that STARTS at t and closes at t+1h.
Every feature in row t uses only bars whose close is <= t+1h, i.e. the decision is taken
right after bar t closes, at price close[t].
"""
import numpy as np
import pandas as pd


def regularize(df, freq):
    """Reindex onto a full grid; forward-fill prices across gaps with zero volume."""
    idx = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    out = df.reindex(idx)
    out["was_missing"] = out["close"].isna().astype(np.int8)
    out["close"] = out["close"].ffill()
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0.0)
    return out


def ewma_vol(r, halflife=72):
    return np.sqrt((r ** 2).ewm(halflife=halflife, min_periods=halflife).mean())


def rolling_z(x, window, min_periods=None):
    m = x.rolling(window, min_periods=min_periods or window // 2).mean()
    s = x.rolling(window, min_periods=min_periods or window // 2).std()
    return (x - m) / s.replace(0, np.nan)


def intrabar_from_5m(m5):
    """Hourly aggregates of 5m bars, keyed by the hour each 5m bar belongs to."""
    r = np.log(m5["close"]).diff()
    hour = m5.index.floor("h")
    g = pd.DataFrame({
        "r": r, "r2": r ** 2, "up": (r > 0).astype(float),
        "sv": np.sign(r) * m5["volume"] * m5["close"],
    }).groupby(hour)
    return pd.DataFrame({
        "rv5": g["r2"].sum(), "up5": g["up"].mean(), "sv5": g["sv"].sum(),
        "skew5": g["r"].skew(), "n5": g["r"].count(),
    })


def build_features(h1, m5=None, eth=None):
    h1 = regularize(h1, "h")
    c, hi, lo, v = h1["close"], h1["high"], h1["low"], h1["volume"]
    lr = np.log(c).diff()
    sigma = ewma_vol(lr)  # per-hour vol
    f = pd.DataFrame(index=h1.index)
    f["sigma"] = sigma

    for n in (1, 4, 12, 24, 72, 168, 336, 720):
        f[f"ret_{n}"] = np.log(c / c.shift(n)) / (sigma * np.sqrt(n))
    for n in (24, 168, 720, 1200):
        f[f"sma_gap_{n}"] = (c / c.rolling(n).mean() - 1) / (sigma * np.sqrt(n))
    ema = c.ewm(span=168).mean()
    f["ema168_slope"] = np.log(ema / ema.shift(24)) / (sigma * np.sqrt(24))
    for n in (720, 2160):
        f[f"dd_{n}"] = c / c.rolling(n).max() - 1

    # volatility family
    park = (np.log(hi / lo) ** 2) / (4 * np.log(2))
    gk = 0.5 * np.log(hi / lo) ** 2 - (2 * np.log(2) - 1) * np.log(c / h1["open"]) ** 2
    for n in (24, 168):
        f[f"park_{n}"] = np.sqrt(park.rolling(n).mean()) / sigma
        f[f"gk_{n}"] = np.sqrt(gk.clip(lower=0).rolling(n).mean()) / sigma
    rv24 = lr.rolling(24).std()
    rv168 = lr.rolling(168).std()
    f["vol_ratio"] = rv24 / rv168
    f["sigma_logz"] = rolling_z(np.log(sigma), 24 * 365)
    rv720 = lr.rolling(720).std()
    f["vol_pctile"] = rv720.rolling(24 * 730, min_periods=24 * 180).rank(pct=True)

    # volume / flow
    dv = v * c
    f["logvol_z"] = rolling_z(np.log1p(dv.rolling(24).sum()), 720)
    clv = ((c - lo) - (hi - c)) / (hi - lo).replace(0, np.nan)
    f["clv_24"] = clv.fillna(0).rolling(24).mean()
    f["amihud_24"] = rolling_z(np.log1p((lr.abs() / dv.replace(0, np.nan)).rolling(24).mean() * 1e9), 720)

    # 5m intrabar statistics
    if m5 is not None:
        ib = intrabar_from_5m(regularize(m5, "5min")).reindex(f.index)
        valid = ib["n5"] >= 10
        rv5 = ib["rv5"].where(valid)
        for n in (24, 72, 168):
            f[f"rv5_{n}"] = np.sqrt(rv5.rolling(n, min_periods=n // 2).mean()) / sigma
        f["up5_24"] = ib["up5"].where(valid).rolling(24, min_periods=12).mean()
        f["skew5_24"] = ib["skew5"].where(valid).rolling(24, min_periods=12).mean()
        sv = ib["sv5"].where(valid)
        scale = dv.rolling(720, min_periods=168).mean()
        for n in (1, 4, 24):
            f[f"sv5_{n}"] = sv.rolling(n, min_periods=1).sum() / (scale * n)

    # cross-asset
    if eth is not None:
        e = regularize(eth, "h")["close"].reindex(f.index).ffill()
        elr = np.log(e).diff()
        for n in (24, 168):
            f[f"eth_rel_{n}"] = (np.log(e / e.shift(n)) - np.log(c / c.shift(n))) / (sigma * np.sqrt(n))
        f["eth_corr_168"] = lr.rolling(168).corr(elr)

    # regime + seasonality
    daily_c = c.resample("D").last()
    mom200 = np.sign(np.log(daily_c / daily_c.shift(200))).shift(1)  # use only completed days
    f["mom200_sign"] = mom200.reindex(f.index, method="ffill")
    hrs = f.index.hour + 1  # decision is taken at bar close
    f["hod_sin"], f["hod_cos"] = np.sin(2 * np.pi * hrs / 24), np.cos(2 * np.pi * hrs / 24)
    dow = f.index.dayofweek
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    f["weekend"] = (dow >= 5).astype(np.int8)
    f["was_missing_24"] = h1["was_missing"].rolling(24).sum()

    f = f.replace([np.inf, -np.inf], np.nan)
    return f, h1


def make_target(close, sigma, horizon):
    """Forward log return over `horizon` hours, and its volatility-scaled version."""
    fwd = np.log(close.shift(-horizon) / close)
    return fwd, fwd / (sigma * np.sqrt(horizon))


FEATURE_EXCLUDE = {"sigma"}


def feature_columns(f):
    return [c for c in f.columns if c not in FEATURE_EXCLUDE]
