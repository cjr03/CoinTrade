"""Historical candle download from Coinbase Advanced Trade public market endpoint (no auth)."""
import argparse
import threading
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.coinbase.com/api/v3/brokerage/market/products/{product}/candles"
GRAN_SECONDS = {"ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900, "ONE_HOUR": 3600}
MAX_CANDLES = 350
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_file_locks = defaultdict(threading.Lock)  # sleeves and the bet pool refresh the same files from different threads


def fetch_window(session, product, granularity, start, end):
    params = {"start": str(start), "end": str(end), "granularity": granularity, "limit": MAX_CANDLES}
    for attempt in range(6):
        r = session.get(BASE.format(product=product), params=params, timeout=20)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json().get("candles", [])
    raise RuntimeError(f"failed after retries: {product} {start}-{end}")


def candles_to_df(candles):
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(candles)
    df["start"] = pd.to_datetime(df["start"].astype(int), unit="s", utc=True)
    df = df.set_index("start").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def download(product="BTC-USD", granularity="ONE_MINUTE", days=730, end_ts=None):
    """Download `days` of candles ending at end_ts, appending to/resuming from a parquet cache."""
    with _file_locks[(product, granularity)]:
        return _download(product, granularity, days, end_ts)


def _download(product, granularity, days, end_ts):
    step = GRAN_SECONDS[granularity]
    path = DATA_DIR / f"{product}_{granularity}.parquet"
    DATA_DIR.mkdir(exist_ok=True)
    end_ts = int(end_ts or time.time()) // step * step
    start_ts = end_ts - days * 86400

    existing = pd.read_parquet(path) if path.exists() else None
    if existing is not None and len(existing):
        have_lo = int(existing.index.min().timestamp())
        have_hi = int(existing.index.max().timestamp())
        ranges = []
        if start_ts < have_lo:
            ranges.append((start_ts, have_lo))
        if end_ts > have_hi:
            ranges.append((have_hi, end_ts))
    else:
        ranges = [(start_ts, end_ts)]

    session = requests.Session()
    frames = [existing] if existing is not None else []
    window = step * MAX_CANDLES
    for lo, hi in ranges:
        n = (hi - lo) // window + 1
        for i, s in enumerate(range(lo, hi, window)):
            e = min(s + window - step, hi)
            frames.append(candles_to_df(fetch_window(session, product, granularity, s, e)))
            time.sleep(0.11)  # stay under public rate limit (~10 req/s)
            if i % 200 == 0:
                print(f"{product} {granularity}: {i}/{n} windows", flush=True)

    df = pd.concat([f for f in frames if f is not None and len(f)])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(path)
    print(f"saved {len(df)} rows -> {path} [{df.index.min()} .. {df.index.max()}]")
    return df


def load(product="BTC-USD", granularity="ONE_MINUTE"):
    with _file_locks[(product, granularity)]:
        return pd.read_parquet(DATA_DIR / f"{product}_{granularity}.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", default="BTC-USD,ETH-USD")
    ap.add_argument("--granularity", default="ONE_MINUTE")
    ap.add_argument("--days", type=int, default=730)
    a = ap.parse_args()
    for p in a.products.split(","):
        download(p, a.granularity, a.days)
