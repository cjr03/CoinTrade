"""Zero-shot foundation-model challenger: Chronos-Bolt / Chronos-2 forecasts of 24h BTC log price.

Scored only on 2025-01-01 onward to limit overlap with the models' pretraining corpora.
Output: artifacts/chronos_{name}.parquet with columns pred (expected 24h log return), spread (q90-q10).
"""
import argparse

import numpy as np
import pandas as pd
import torch

from .data import load
from .features import regularize
from .train import OUT

MODELS = {"bolt_small": "amazon/chronos-bolt-small", "chronos2": "amazon/chronos-2"}


def run(name, horizon=24, context=512, step=6, start="2025-01-01", batch=64):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(MODELS[name], device_map="cpu", torch_dtype=torch.float32)
    close = regularize(load("BTC-USD", "ONE_HOUR"), "h")["close"]
    logp = np.log(close.values)
    idx = close.index
    pos = [i for i in range(len(idx)) if idx[i] >= pd.Timestamp(start, tz="UTC") and i % step == 0]
    rows = []
    for b in range(0, len(pos), batch):
        chunk = pos[b:b + batch]
        # context ends at bar i inclusive (decision at close of bar i); normalize by last value
        ctx = [torch.tensor(logp[i - context + 1:i + 1] - logp[i], dtype=torch.float32) for i in chunk]
        q, mean = pipe.predict_quantiles(ctx, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9])
        for j, i in enumerate(chunk):
            qj = q[j][0] if isinstance(q, list) else q[j]  # Chronos-2 returns per-series (variates, H, Q)
            rows.append((idx[i], float(qj[-1, 1]), float(qj[-1, 2] - qj[-1, 0])))
        print(f"{name}: {b + len(chunk)}/{len(pos)}", flush=True)
    out = pd.DataFrame(rows, columns=["t", "pred", "spread"]).set_index("t")
    out["fwd"] = pd.Series(np.log(close.shift(-horizon) / close).values, idx).reindex(out.index)
    out.to_parquet(OUT / f"chronos_{name}.parquet")
    d = out.dropna()
    print(name, "IC", d["pred"].corr(d["fwd"], method="spearman"), "n", len(d))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="bolt_small,chronos2")
    a = ap.parse_args()
    for m in a.models.split(","):
        run(m)
