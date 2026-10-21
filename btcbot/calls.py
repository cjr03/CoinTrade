"""15-minute up/down calls for every coin: recommendations shown in the desks, never traded by the bot.

    python -m btcbot.calls --train         # fit one model per coin -> artifacts/calls_<COIN>.txt + .json

Each coin gets a small LightGBM classifier on causal 15-minute features. It is trained on the oldest
70% of two years of Coinbase candles, calibrated and given its confidence bar on the next 15%, and
scored once on the newest 15%, which it never saw. A call is made only when the calibrated chance of
the next 15 minutes closing up (or down) clears that bar. The desks show the tested hit rate and what
the calls would have made after Kraken's round-trip fee, plus a live scorecard of the calls so far.

Windows are the clock's quarter hours (2:15-2:30, 2:30-2:45, ...), settled the way Kalshi's 15-minute
markets are: the price to beat is the average price over the 60 seconds before the window opens, and the
result is the average over the 60 seconds before it closes (Kalshi uses the CF Benchmarks index; we use
Kraken's per-minute VWAP). While a window runs, the odds move with the live price and the time left
(live_odds): a random walk using the last hour's 1-minute volatility, with the settlement average's
smaller variance, and a spread factor fitted on six months of 1-minute candles for all four coins. The
model's opening lean fades as the window runs out. Kalshi's own price is polled for comparison.

Models are saved as LightGBM text plus JSON (no pickles), so loading one never runs code.
"""
import argparse
import json
import logging
import math
import threading
import time
from datetime import datetime, timezone
from statistics import NormalDist

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .coins import COINS
from .data import download, load
from .features import regularize
from .kraken import public as kraken_public

ART_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "artifacts"
GRAN, BAR = "FIFTEEN_MINUTE", pd.Timedelta(minutes=15)
ROUND_TRIP = 0.008  # Kraken tier 1: 0.40% maker in, 0.40% maker out
THRESHOLDS = np.round(np.arange(0.51, 0.605, 0.005), 3)
MIN_COVERAGE = 0.10  # a bar that almost never fires is not a recommendation service
NORM = NormalDist()
# spread factor a = A0 + A1 * (fraction of the window left), fitted on 180 days of 1-minute candles, all four
# coins, first 70% of windows; scored on the last 30% (Kalshi-style 60-second averages at both ends)
A0, A1 = 1.40, -0.40  # last-hour 1m vol; test Brier 0.148 vs 0.250 coin flip, within ~3 points in every decile
KALSHI = {"BTC": "KXBTC15M", "ETH": "KXETH15M", "SOL": "KXSOL15M", "XRP": "KXXRP15M"}
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2/markets"
log = logging.getLogger("btcbot")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def bars(coin):
    return regularize(load(COINS[coin]["product"], GRAN), "15min")


def features(df, btc=None):
    """Causal features for each closed 15-minute bar, and the next bar's log return as the target."""
    c = df["close"]
    lc = np.log(c)
    lr = lc.diff()
    vol96 = lr.rolling(96).std()
    f = pd.DataFrame(index=df.index)
    for k in (1, 2, 4, 8, 16, 96):
        f[f"r{k}"] = lc.diff(k)
    for k in (1, 4, 16):
        f[f"z{k}"] = f[f"r{k}"] / (vol96 * np.sqrt(k))
    f["vol16"] = lr.rolling(16).std()
    f["vol_ratio"] = f["vol16"] / vol96
    f["range"] = (df["high"] - df["low"]) / c
    lv = np.log1p(df["volume"])
    f["vol_z"] = lv - lv.rolling(96).mean()
    f["dev16"] = (c / c.rolling(16).mean() - 1) / vol96
    hour = df.index.hour + df.index.minute / 60
    f["h_sin"], f["h_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow"] = df.index.dayofweek
    if btc is not None:
        bl = np.log(btc["close"]).reindex(df.index).ffill()
        f["btc_r1"], f["btc_r4"] = bl.diff(), bl.diff(4)
    return f, lr.shift(-1)


def calibrate(meta, raw):
    return np.interp(raw, meta["iso_x"], meta["iso_y"])


def tau_eff(rem_s):
    """Minutes of random-walk variance left in the result: the 60-second settlement average has a third of a
    minute's variance in its own minute, so it is less uncertain than the price at the close."""
    rem = rem_s / 60
    return rem - 2 / 3 if rem > 1 else rem ** 3 / 3


def live_odds(start, price, sigma1m, z0, rem_s, window_s=900):
    """Chance the window settles at or above `start`, with the price now at `price` and `rem_s` seconds left.
    sigma1m: 1-minute volatility (last hour); z0: the model's opening lean (inverse-normal of its P(up))."""
    if not (start and price and sigma1m) or sigma1m <= 0:
        return None
    if rem_s <= 0:
        return 1.0 if price >= start else 0.0
    frac = min(rem_s / window_s, 1.0)
    z = math.log(price / start) / (sigma1m * math.sqrt(tau_eff(rem_s)))
    return min(0.98, max(0.02, NORM.cdf((A0 + A1 * frac) * z + z0 * math.sqrt(frac))))


def kraken_minutes(coin, since):
    key = COINS[coin]["kraken_key"]
    return kraken_public("OHLC", pair=COINS[coin]["kraken_pair"], interval=1, since=int(since))[key]


def kraken_avg_before(coin, t, rows=None):
    """Kraken's average price over the 60 seconds before `t`: the VWAP of the 1-minute candle ending at t
    (its close if nothing traded). Kalshi averages the CF Benchmarks index over the same 60 seconds."""
    t = int(pd.Timestamp(t).timestamp())
    rows = rows or kraken_minutes(coin, t - 600)
    for r in rows:
        if int(r[0]) == t - 60:
            vwap = float(r[5])
            return vwap if vwap > 0 and float(r[6]) > 0 else float(r[4])
    before = [r for r in rows if int(r[0]) < t]
    if not before:
        raise RuntimeError(f"no Kraken {coin} candle before {pd.Timestamp(t, unit='s')}")
    return float(before[-1][4])


def kraken_sigma1m(rows, t):
    """1-minute volatility over the hour before `t`, from Kraken's 1-minute closes."""
    t = int(pd.Timestamp(t).timestamp())
    closes = np.array([float(r[4]) for r in rows if t - 3660 <= int(r[0]) < t])
    if len(closes) < 30:
        raise RuntimeError("not enough Kraken minutes for volatility")
    return float(np.diff(np.log(closes)).std())


def kalshi_ticker(coin, close):
    """Kalshi names each 15-minute market by its close in New York time: KXBTC15M-26OCT210130-30."""
    from zoneinfo import ZoneInfo
    et = pd.Timestamp(close).tz_convert(ZoneInfo("America/New_York"))
    return f"{KALSHI[coin]}-{et.strftime('%y%b%d%H%M').upper()}-{et.strftime('%M')}"


def kalshi_market(coin):
    """The open Kalshi 15-minute up/down market for `coin` (public market data, read only). Fetched by ticker:
    the single-market endpoint is live, while the market list is served from a cache up to ~15 s old."""
    import requests
    close = pd.Timestamp.now(tz="UTC").floor("15min") + BAR
    r = requests.get(f"{KALSHI_API}/{kalshi_ticker(coin, close)}", timeout=8)
    if r.status_code == 404:  # naming changed? find it in the (cached) list instead
        r = requests.get(KALSHI_API, params={"series_ticker": KALSHI[coin], "status": "open"}, timeout=8)
        r.raise_for_status()
        ms = [m for m in r.json().get("markets") or [] if pd.Timestamp(m["close_time"]) == close]
        if not ms:
            return None
        m = ms[0]
    else:
        r.raise_for_status()
        m = r.json()["market"]
    f = lambda k: float(m[k]) if m.get(k) not in (None, "") else None  # noqa: E731
    return {"ticker": m["ticker"], "strike": m.get("floor_strike"), "from": m["open_time"], "until": m["close_time"],
            "yes_bid": f("yes_bid_dollars"), "yes_ask": f("yes_ask_dollars"), "last": f("last_price_dollars"),
            "fetched": now_iso()}


# --- training --------------------------------------------------------------------------------
def score(p, ys, thr):
    """Hit rate and dollars of the calls a confidence bar would have made on (p, next-bar log returns)."""
    conf = np.maximum(p, 1 - p)
    m = conf >= thr
    side = np.where(p >= 0.5, 1.0, -1.0)
    right = side * ys > 0
    signed = side * (np.exp(ys) - 1)  # what a bet in the called direction made, before fees
    return {"coverage": float(m.mean()), "calls": int(m.sum()),
            "hit_rate": float(right[m].mean()) if m.any() else None,
            "avg_move_bp": float(np.abs(np.exp(ys[m]) - 1).mean() * 1e4) if m.any() else None,
            "edge_bp": float(signed[m].mean() * 1e4) if m.any() else None,
            "after_fees_bp": float((signed[m] - ROUND_TRIP).mean() * 1e4) if m.any() else None,
            "all_bars_hit_rate": float(right.mean())}


def train_coin(coin, verbose=True):
    df = bars(coin)
    btc = None if coin == "BTC" else bars("BTC")
    f, y = features(df, btc)
    data = f.assign(y=y)
    data = data[df["was_missing"] == 0].dropna()
    n = len(data)
    a, b = int(n * 0.70), int(n * 0.85)
    tr, va, te = data.iloc[:a], data.iloc[a + 1:b], data.iloc[b + 1:]  # one-bar embargo: targets look one bar ahead
    cols = list(f.columns)
    clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=300,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    clf.fit(tr[cols], tr["y"] > 0, eval_set=[(va[cols], va["y"] > 0)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    raw_va = clf.predict_proba(va[cols])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_va, va["y"] > 0)
    meta = {"coin": coin, "features": cols, "iso_x": iso.X_thresholds_.tolist(), "iso_y": iso.y_thresholds_.tolist()}
    p_va = calibrate(meta, raw_va)
    best = None
    for thr in THRESHOLDS:  # the confidence bar is chosen on validation only
        s = score(p_va, va["y"].values, thr)
        if s["coverage"] >= MIN_COVERAGE and (best is None or s["hit_rate"] > best[1]["hit_rate"] + 1e-9):
            best = (float(thr), s)
    thr = best[0] if best else 0.5
    p_te = calibrate(meta, clf.predict_proba(te[cols])[:, 1])
    test = score(p_te, te["y"].values, thr)
    test["up_share"] = float((te["y"] > 0).mean())  # 'always say up' baseline
    meta.update(threshold=thr, validation=best[1] if best else None, test=test,
                train=[str(tr.index[0]), str(tr.index[-1])], test_span=[str(te.index[0]), str(te.index[-1])],
                trained_at=now_iso(), best_iteration=int(clf.best_iteration_ or clf.n_estimators))
    clf.booster_.save_model(str(ART_DIR / f"calls_{coin}.txt"))
    (ART_DIR / f"calls_{coin}.json").write_text(json.dumps(meta, indent=1))
    if verbose:
        print(f"{coin}: call when P >= {thr:.3f} | test {meta['test_span'][0][:10]}..{meta['test_span'][1][:10]}: "
              f"{test['calls']} calls ({test['coverage']:.0%} of bars), right {test['hit_rate']:.1%} "
              f"(every bar {test['all_bars_hit_rate']:.1%}, always-up {test['up_share']:.1%}), "
              f"avg move {test['avg_move_bp']:.0f}bp, edge {test['edge_bp']:+.1f}bp, after fees {test['after_fees_bp']:+.0f}bp")
    return meta


# --- live calls --------------------------------------------------------------------------------
class Caller:
    """At each quarter hour: one call per coin for the window that has just started (opening odds, the
    price to beat, volatility for the live odds), and a score for the window that has just ended."""

    def __init__(self, coins, state_dir, sink=None):
        self.models = {}
        for c in coins:
            txt, js = ART_DIR / f"calls_{c}.txt", ART_DIR / f"calls_{c}.json"
            if txt.exists() and js.exists():
                self.models[c] = (lgb.Booster(model_file=str(txt)), json.loads(js.read_text()))
        self.path = state_dir / "calls.json"
        self.s = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.s.setdefault("current", {})
        self.s.setdefault("history", {})
        self.sink = sink
        self.lock = threading.Lock()
        self.kalshi = {}  # coin -> the open Kalshi market, refreshed every ~3 seconds

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.s, indent=1, default=str))
        tmp.replace(self.path)

    def _call(self, coin, booster, meta):
        last = pd.Timestamp.now(tz="UTC").floor("15min") - BAR  # start of the bar that has just closed
        for attempt in range(6):  # Coinbase publishes the closed bar a few seconds after the quarter hour
            download(COINS[coin]["product"], GRAN, 3)
            df = bars(coin)
            if df.index.max() >= last:
                break
            time.sleep(5)
        else:
            raise RuntimeError(f"stale {coin} 15m data: last bar {df.index.max()}, expected {last}")
        df = df.loc[:last]
        btc = None
        if "btc_r1" in meta["features"]:
            if coin != "BTC":
                download(COINS["BTC"]["product"], GRAN, 3)
            btc = bars("BTC").loc[:last]
        f, _ = features(df.iloc[-400:], btc.iloc[-400:] if btc is not None else None)
        row = f.iloc[[-1]][meta["features"]]
        if row.isna().any(axis=None):
            raise RuntimeError(f"incomplete {coin} features for {last}")
        p = float(calibrate(meta, booster.predict(row.values))[0])
        conf = max(p, 1 - p)
        t0 = last + BAR  # the window that has just opened
        try:
            rows = kraken_minutes(coin, t0.timestamp() - 3720)
            start, src = kraken_avg_before(coin, t0, rows), "kraken"
            sigma1m = kraken_sigma1m(rows, t0)
        except Exception as e:
            log.warning("15m call %s: Kraken minutes unavailable (%s); using Coinbase", coin, e)
            start, src = float(df["close"].iloc[-1]), "coinbase"
            sigma1m = float(np.log(df["close"]).diff().iloc[-4:].std() / math.sqrt(15))
        return {"bar": str(last), "p_up": p, "conf": conf, "price": start, "start": start, "start_src": src,
                "sigma1m": sigma1m, "z0": NORM.inv_cdf(min(0.98, max(0.02, p))),
                "dir": ("up" if p >= 0.5 else "down") if conf >= meta["threshold"] else None,
                "from": str(last + BAR), "until": str(last + 2 * BAR), "threshold": meta["threshold"],
                "issued": now_iso()}

    def _score(self, coin, prev, close):
        """Score the previous call once its window has ended, on the same exchange as its start price."""
        if prev.get("dir") is None or pd.Timestamp(prev["until"]) > pd.Timestamp.now(tz="UTC"):
            return
        h = self.s["history"].get(coin) or []
        if h and h[-1]["bar"] == prev["bar"]:
            return
        if prev.get("start_src") == "kraken":
            try:
                end_px = kraken_avg_before(coin, prev["until"])
            except Exception as e:
                log.warning("15m score %s: %s", coin, e)
                return
        else:
            end = pd.Timestamp(prev["from"])  # the Coinbase bar covering the window
            if end not in close.index:
                return
            end_px = close.loc[end]
        move = end_px / prev["price"] - 1
        hit = move >= 0 if prev["dir"] == "up" else move < 0  # ties settle 'up', as on Kalshi
        h = self.s["history"].setdefault(coin, [])
        if not h or h[-1]["bar"] != prev["bar"]:
            h.append({"bar": prev["bar"], "dir": prev["dir"], "p_up": prev["p_up"], "move_bp": move * 1e4,
                      "hit": bool(hit)})
            del h[:-200]

    def refresh(self):
        for coin, (booster, meta) in self.models.items():
            try:
                call = self._call(coin, booster, meta)
            except Exception as e:
                log.warning("15m call %s: %s", coin, e)
                with self.lock:
                    self.s["current"][coin] = {"error": str(e), "issued": now_iso()}
                continue
            with self.lock:
                prev = self.s["current"].get(coin)
                if prev and "bar" in prev:
                    self._score(coin, prev, bars(coin)["close"])
                self.s["current"][coin] = call
        with self.lock:
            self.save()

    def snapshot(self):
        with self.lock:
            out = {}
            for coin, (_, meta) in self.models.items():
                day = self.s["history"].get(coin, [])[-96:]  # the last day of calls
                net = sum((x["move_bp"] if x["dir"] == "up" else -x["move_bp"]) - ROUND_TRIP * 1e4 for x in day)
                out[coin] = {**self.s["current"].get(coin, {}), "kalshi": self.kalshi.get(coin), "a0": A0, "a1": A1,
                             "record": {"n": len(day), "hits": sum(x["hit"] for x in day), "net_bp": net},
                             "tested": {k: meta["test"][k] for k in
                                        ("hit_rate", "coverage", "after_fees_bp", "all_bars_hit_rate", "calls")}}
            return out

    def poll_kalshi(self, stop=None):
        while not (stop and stop.is_set()):
            for coin in self.models:
                try:
                    m = kalshi_market(coin)
                except Exception as e:
                    log.debug("kalshi %s: %s", coin, e)
                    continue
                with self.lock:
                    self.kalshi[coin] = m
            for _ in range(6):  # every ~3 seconds: four requests to a live, uncached endpoint
                if stop and stop.is_set():
                    return
                time.sleep(0.5)

    def run(self, stop=None):
        threading.Thread(target=self.poll_kalshi, kwargs={"stop": stop}, daemon=True, name="kalshi").start()
        while not (stop and stop.is_set()):
            try:
                self.refresh()
            except Exception:
                log.exception("15m calls failed; retrying next quarter hour")
            nxt = (time.time() // 900 + 1) * 900 + 3  # just after each quarter hour; _call waits for the bar
            while time.time() < nxt and not (stop and stop.is_set()):
                time.sleep(min(5, max(nxt - time.time(), 0.1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--coins", default=",".join(COINS))
    a = ap.parse_args()
    if a.train:
        for c in a.coins.split(","):
            train_coin(c)


if __name__ == "__main__":
    main()
