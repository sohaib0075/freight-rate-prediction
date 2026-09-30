"""Backtest the day-level specification on its own.

The day-level term is the only part of the model that has to extrapolate past
the end of the training data, so it is chosen by a dedicated backtest rather
than by end-to-end holdout scores (which are dominated by cross-sectional noise
and by whichever single month happens to be anomalous).

Procedure: fit the cross-sectional GBM out-of-fold over the whole development
set, average its residual per calendar day to get the daily level series, then
roll an origin forward and score each candidate specification on the next 30
and 61 days.

    python experiments/day_level_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D                                          # noqa: E402
from src import features as F                                      # noqa: E402
from src.model import CROSS_SECTION, FreightRateModel              # noqa: E402

ORIGINS = [212, 243, 273]          # 1 Aug, 1 Sep, 1 Oct (days since 1 Jan 2025)
HORIZONS = [30, 61]


def main() -> None:
    train_raw, valid_raw = D.prepare(Path("data"))
    pooled = pd.concat([train_raw, valid_raw])
    ref = F.build_reference([train_raw, valid_raw])
    dev = F.transform(train_raw, ref)

    y = FreightRateModel.target(dev)
    model = FreightRateModel()
    inliers = model._flag_inliers(dev, y)
    kept, y_kept = dev[inliers].reset_index(drop=True), y[inliers]

    residual = y_kept - model._oof(kept, y_kept, CROSS_SECTION, model.n_rounds, seed=1)
    level = pd.DataFrame({"date": kept["date"].to_numpy(), "r": residual}) \
              .groupby("date")["r"].agg(["mean", "size"])

    idx = level.index
    t = (idx - pd.Timestamp("2025-01-01")).days.to_numpy(dtype=float)
    columns = {
        "mi": np.log(pooled.groupby("date")["market_index"].mean().reindex(idx).to_numpy()),
        "q": pooled.groupby("date")["quote_signal"].mean().reindex(idx).to_numpy(),
        "dom": idx.day.to_numpy(dtype=float),
        "t": t,
    }
    dow = idx.dayofweek.to_numpy()
    dow_dummies = [(dow == j) * 1.0 for j in range(1, 7)]

    specs = {
        "mi":             ["mi"],
        "mi+t":           ["mi", "t"],
        "mi+q":           ["mi", "q"],
        "mi+q+t":         ["mi", "q", "t"],
        "mi+q+t+dom":     ["mi", "q", "t", "dom"],
        "mi+q+t+dow":     ["mi", "q", "t", "dow"],
        "mi+q+t+dom+dow": ["mi", "q", "t", "dom", "dow"],
    }

    values, weights = level["mean"].to_numpy(), np.sqrt(level["size"].to_numpy())
    header = "".join(f"  o{o}/h{h:<4d}" for o in ORIGINS for h in HORIZONS)
    print(f"daily level sd {values.std():.5f} over {len(values)} days\n")
    print(f"{'specification':16s}{header}   mean    last-origin")
    for name, spec in specs.items():
        design = np.column_stack(
            [np.ones(len(t))]
            + [c for s in spec for c in (dow_dummies if s == "dow" else [columns[s]])]
        )
        row = []
        for origin in ORIGINS:
            seen = t < origin
            w = weights[seen]
            coef, *_ = np.linalg.lstsq(design[seen] * w[:, None], values[seen] * w, rcond=None)
            predicted = design @ coef
            for horizon in HORIZONS:
                window = (t >= origin) & (t < origin + horizon)
                row.append(np.sqrt(np.mean((values[window] - predicted[window]) ** 2))
                           if window.any() else np.nan)
        print(f"{name:16s}" + "".join(f"  {v:9.5f}" for v in row)
              + f"  {np.nanmean(row):.5f}  {np.nanmean(row[-2:]):.5f}")


if __name__ == "__main__":
    main()
