"""Purged walk-forward training of the candidate models.

For every quarterly test fold:  [train: 3y] -purge- [val: 3m] -purge- [test: 3m]
Train rows are purged so their label window (t, t+H] never reaches into val; val rows are
purged so their labels never reach into test.  Early stopping uses an inner purged holdout
carved from the END of the train window, so val/test are never seen during fitting.
Outputs out-of-sample predictions (expected log return over H) for val and test segments.
"""
import argparse
import warnings
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load
from .features import build_features, feature_columns, make_target

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts"
MODELS = ("lgbm", "ridge", "lgbm_tb")
SEEDS = (0, 1, 2)
ROUND_TRIP = 0.0121  # worst-case maker round trip used for triple-barrier width

LGB_PARAMS = dict(
    objective="huber", alpha=1.5, learning_rate=0.02, num_leaves=15, min_child_samples=300,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
    n_estimators=1500, verbose=-1, n_jobs=6,
)


def load_dataset(use_5m=True):
    h1 = load("BTC-USD", "ONE_HOUR")
    m5 = load("BTC-USD", "FIVE_MINUTE") if use_5m else None
    eth = load("ETH-USD", "ONE_HOUR")
    f, bars = build_features(h1, m5, eth)
    return f, bars


def triple_barrier(close, high, low, sigma, horizon, min_width=2 * ROUND_TRIP):
    """1 if the upper barrier is touched before the lower one within `horizon` bars."""
    c, h, l = close.values, high.values, low.values
    width = np.maximum(min_width, sigma.values * np.sqrt(horizon))
    up, dn = c * np.exp(width), c * np.exp(-width)
    n = len(c)
    first_up = np.full(n, np.inf)
    first_dn = np.full(n, np.inf)
    for k in range(1, horizon + 1):
        hk = np.concatenate([h[k:], np.full(k, np.nan)])
        lk = np.concatenate([l[k:], np.full(k, np.nan)])
        first_up = np.where((hk >= up) & np.isinf(first_up), k, first_up)
        first_dn = np.where((lk <= dn) & np.isinf(first_dn), k, first_dn)
    label = (first_up < first_dn).astype(float)
    label[n - horizon:] = np.nan
    return pd.Series(label, close.index)


def make_folds(index, horizon, start="2019-01-01", train_years=3, val_months=3, test_months=3):
    folds = []
    last = index.max()
    ts = pd.Timestamp(start, tz="UTC")
    purge = pd.Timedelta(hours=horizon)
    while ts + pd.DateOffset(months=1) <= last:
        te = min(ts + pd.DateOffset(months=test_months), last + pd.Timedelta(hours=1))
        vs = ts - pd.DateOffset(months=val_months)
        tr_s = vs - pd.DateOffset(years=train_years)
        folds.append(dict(
            train=(tr_s, vs - purge), val=(vs, ts - purge), test=(ts, te),
        ))
        ts = ts + pd.DateOffset(months=test_months)
    return folds


def _slice(df, a, b):
    return df[(df.index >= a) & (df.index < b)]


def fit_predict(model, Xtr, ytr, Xin, yin, Xs, fwd_in=None):
    """Fit on (Xtr,ytr) with early stopping on (Xin,yin); predict a list of frames."""
    if model == "ridge":
        lo, hi = Xtr.quantile(0.005), Xtr.quantile(0.995)
        clip = lambda X: X.clip(lo, hi, axis=1)
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0))
        pipe.fit(clip(Xtr), ytr)
        return [pipe.predict(clip(X)) for X in Xs], {"model": pipe, "clip": (lo, hi)}

    if model == "lgbm":
        preds, boosters = [np.zeros(len(X)) for X in Xs], []
        for s in SEEDS:
            m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=s)
            m.fit(Xtr, ytr, eval_set=[(Xin, yin)], callbacks=[lgb.early_stopping(100, verbose=False)])
            boosters.append(m)
            for i, X in enumerate(Xs):
                preds[i] += m.predict(X) / len(SEEDS)
        return preds, {"models": boosters}

    if model == "lgbm_tb":
        params = {**LGB_PARAMS, "objective": "binary"}
        params.pop("alpha")
        # early-stop on the first half of the inner holdout, calibrate on the second half
        half = len(Xin) // 2
        Xes, yes, Xcal, fcal = Xin.iloc[:half], yin.iloc[:half], Xin.iloc[half:], fwd_in[half:]
        boosters = []
        p_cal = np.zeros(len(Xcal))
        preds = [np.zeros(len(X)) for X in Xs]
        for s in SEEDS:
            m = lgb.LGBMClassifier(**params, random_state=s)
            m.fit(Xtr, ytr, eval_set=[(Xes, yes)], callbacks=[lgb.early_stopping(100, verbose=False)])
            boosters.append(m)
            p_cal += m.predict_proba(Xcal)[:, 1] / len(SEEDS)
            for i, X in enumerate(Xs):
                preds[i] += m.predict_proba(X)[:, 1] / len(SEEDS)
        # calibrate P(up-barrier first) -> expected forward log return on unseen data
        ok = ~np.isnan(fcal)
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal[ok], fcal[ok])
        return [iso.predict(p) for p in preds], {"models": boosters, "iso": iso}
    raise ValueError(model)


def walk_forward(f, bars, horizon, models=MODELS, verbose=True):
    bars = bars.loc[f.index]
    cols = feature_columns(f)
    fwd, y = make_target(bars["close"], f["sigma"], horizon)
    tb = triple_barrier(bars["close"], bars["high"], bars["low"], f["sigma"], horizon)
    data = f[cols].copy()
    data["_y"], data["_fwd"], data["_tb"], data["_sig"] = y, fwd, tb, f["sigma"]
    data = data[data.index >= pd.Timestamp("2015-12-01", tz="UTC")]  # warmup for long windows
    purge = pd.Timedelta(hours=horizon)

    rows = []
    for k, fold in enumerate(make_folds(data.index, horizon)):
        tr = _slice(data, *fold["train"]).dropna(subset=["_y", "_tb"])
        va = _slice(data, *fold["val"])
        te = _slice(data, *fold["test"])
        # inner purged holdout: last 15% of train for early stopping/calibration
        cut = tr.index[int(len(tr) * 0.85)]
        fit, inner = tr[tr.index < cut - purge], tr[tr.index >= cut]
        out = pd.DataFrame(index=va.index.append(te.index))
        out["segment"] = ["val"] * len(va) + ["test"] * len(te)
        out["fold"] = k
        out["fwd"] = pd.concat([va["_fwd"], te["_fwd"]])
        scale = pd.concat([va["_sig"], te["_sig"]]) * np.sqrt(horizon)
        for model in models:
            ycol = "_tb" if model == "lgbm_tb" else "_y"
            (pv, pt), _ = fit_predict(model, fit[cols], fit[ycol], inner[cols], inner[ycol],
                                      [va[cols], te[cols]], fwd_in=inner["_fwd"].values)
            p = np.concatenate([pv, pt])
            out[model] = p if model == "lgbm_tb" else p * scale.values
        rows.append(out)
        if verbose:
            tt = out[out.segment == "test"].dropna(subset=["fwd"])
            ics = {m: round(tt[m].corr(tt["fwd"], method="spearman"), 3) for m in models}
            print(f"H={horizon} fold {k:2d} test {fold['test'][0].date()} n_train={len(fit)} IC={ics}", flush=True)
    return pd.concat(rows)


def train_final(f, bars, horizon, model="lgbm", train_years=3):
    """Fit the production model on the most recent `train_years` of labelled data."""
    cols = feature_columns(f)
    fwd, y = make_target(bars["close"], f["sigma"], horizon)
    tb = triple_barrier(bars["close"], bars["high"], bars["low"], f["sigma"], horizon)
    d = f[cols].copy()
    d["_y"], d["_fwd"], d["_tb"] = y, fwd, tb
    d = d.dropna(subset=["_y", "_tb"])
    d = d[d.index >= d.index.max() - pd.DateOffset(years=train_years)]
    cut = d.index[int(len(d) * 0.85)]
    fit, inner = d[d.index < cut - pd.Timedelta(hours=horizon)], d[d.index >= cut]
    ycol = "_tb" if model == "lgbm_tb" else "_y"
    _, bundle = fit_predict(model, fit[cols], fit[ycol], inner[cols], inner[ycol], [inner[cols]],
                            fwd_in=inner["_fwd"].values)
    bundle.update(model_type=model, horizon=horizon, columns=cols,
                  trained_through=str(d.index.max()), n_train=len(fit))
    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="12,24,72")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--final", action="store_true", help="train production model (needs --model/--horizon)")
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--no-5m", action="store_true", help="ablation: hourly + ETH features only")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    f, bars = load_dataset(use_5m=not a.no_5m)
    tag = "_no5m" if a.no_5m else ""
    if a.final:
        bundle = train_final(f, bars, a.horizon, a.model)
        with open(OUT / "model.pkl", "wb") as fh:
            pickle.dump(bundle, fh)
        print(json.dumps({k: v for k, v in bundle.items() if isinstance(v, (str, int))}, indent=1))
    else:
        for H in map(int, a.horizons.split(",")):
            preds = walk_forward(f, bars, H, models=a.models.split(","))
            preds.to_parquet(OUT / f"oos_H{H}{tag}.parquet")
