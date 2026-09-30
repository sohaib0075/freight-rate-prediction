"""Backtest the day-level specification on its own.

The day-level term is the only part of the model that has to extrapolate past
the end of the training data, so it is chosen by a dedicated backtest rather
than by end-to-end holdout scores, which are dominated by cross-sectional noise
and by whichever single month happens to be anomalous.

Procedure: fit the cross-sectional GBM out-of-fold over the whole development
set and average its residual per calendar day to get the daily level series
(slow, so it is cached).  Then roll an origin forward every seven days, fit
each candidate specification on everything before that origin, and score its
forecast of the next 30 and 61 days.

    python experiments/day_level_backtest.py                # uses the cache if present
    python experiments/day_level_backtest.py --refresh      # recompute the level series

What it shows, on the data as supplied:

* Calendar time is doing the work.  Dropping it roughly doubles the forecast
  error at every horizon; nothing else recovers it.
* Day-of-week and day-of-month belong in the trees, not here.  Adding them to
  the day model looks good averaged over all origins, but that average is
  carried by the June-July origins, where every specification is poor.  At the
  origins that resemble the real task - seven months or more of history - they
  are consistently the worst of the three.
* The quote term is a genuine coin flip: ahead at some origins, behind at
  others, within noise of dropping it.  It is kept because it is strongly
  significant in-sample, and because November and December sit almost exactly
  at the training mean of the quote series, so it barely moves the answer
  either way.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D                                          # noqa: E402
from src import features as F                                      # noqa: E402
from src.model import CROSS_SECTION, FreightRateModel              # noqa: E402

CACHE = Path("report/day_level_series.csv")
HORIZONS = [30, 61]
ORIGIN_STEP = 7
FIRST_ORIGIN, LAST_ORIGIN = 150, 290          # days since 1 Jan 2025
ENOUGH_HISTORY = 212                          # 1 Aug: seven months in, like the real task

SPECS = {
    "mi":             ["mi"],
    "mi+t":           ["mi", "t"],
    "mi+q":           ["mi", "q"],
    "mi+q+t":         ["mi", "q", "t"],          # shipped
    "mi+q+t+dom":     ["mi", "q", "t", "dom"],
    "mi+q+t+dow":     ["mi", "q", "t", "dow"],
    "mi+q+t+dom+dow": ["mi", "q", "t", "dom", "dow"],
}


def daily_level(refresh: bool) -> pd.DataFrame:
    """The daily level series, plus the two market series it is regressed on.

    Everything the backtest needs is cached in one small table, so a fresh
    clone can reproduce the specification choice without the raw CSVs and
    without the several-minute out-of-fold fit.  ``--refresh`` rebuilds it.
    """
    if CACHE.is_file() and not refresh:
        return pd.read_csv(CACHE, parse_dates=["date"]).set_index("date")

    train_raw, valid_raw = D.prepare(Path("data"))
    dev = F.transform(train_raw, F.build_reference([train_raw, valid_raw]))
    y = FreightRateModel.target(dev)
    model = FreightRateModel()
    inliers = model._flag_inliers(dev, y)
    kept, y_kept = dev[inliers].reset_index(drop=True), y[inliers]
    residual = y_kept - model._oof(kept, y_kept, CROSS_SECTION, model.n_rounds, seed=1)

    level = (pd.DataFrame({"date": kept["date"].to_numpy(), "r": residual})
             .groupby("date")["r"].agg(["mean", "size"]))
    pooled = pd.concat([train_raw, valid_raw])
    level["market_index"] = pooled.groupby("date")["market_index"].mean().reindex(level.index)
    level["quote_signal"] = pooled.groupby("date")["quote_signal"].mean().reindex(level.index)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    level.to_csv(CACHE)
    return level


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="recompute the daily level series instead of using the cache")
    args = parser.parse_args()

    level = daily_level(args.refresh)

    idx = level.index
    t = (idx - F.ORIGIN).days.to_numpy(dtype=float)
    values, weights = level["mean"].to_numpy(), np.sqrt(level["size"].to_numpy())
    dow = idx.dayofweek.to_numpy()
    dow_dummies = [(dow == j) * 1.0 for j in range(1, 7)]
    columns = {
        "mi": np.log(level["market_index"].to_numpy()),
        "q": level["quote_signal"].to_numpy(),
        "dom": idx.day.to_numpy(dtype=float),
        "t": t,
    }

    designs = {
        name: np.column_stack(
            [np.ones(len(t))]
            + [c for s in spec for c in (dow_dummies if s == "dow" else [columns[s]])])
        for name, spec in SPECS.items()
    }

    print(f"daily level sd {values.std():.5f} over {len(values)} days")
    origins = range(FIRST_ORIGIN, LAST_ORIGIN + 1, ORIGIN_STEP)

    for horizon in HORIZONS:
        scores, used = {name: [] for name in SPECS}, []
        for origin in origins:
            window = (t >= origin) & (t < origin + horizon)
            if window.sum() < 15:
                continue
            used.append(origin)
            seen = t < origin
            w = weights[seen]
            for name, design in designs.items():
                coef, *_ = np.linalg.lstsq(design[seen] * w[:, None], values[seen] * w, rcond=None)
                predicted = design @ coef
                scores[name].append(np.sqrt(np.mean((values[window] - predicted[window]) ** 2)))

        table = pd.DataFrame(scores, index=[F.ORIGIN + pd.Timedelta(days=o) for o in used])
        recent = table[np.array(used) >= ENOUGH_HISTORY]
        print(f"\n=== forecast RMSE of the daily level, horizon {horizon} days, "
              f"{len(table)} origins ===")
        print(table.round(5).to_string(float_format=lambda v: f"{v:.5f}"))
        summary = pd.DataFrame({
            "all origins (mean)": table.mean(),
            "all origins (median)": table.median(),
            f"origins from {(F.ORIGIN + pd.Timedelta(days=ENOUGH_HISTORY)).date()} (mean)":
                recent.mean(),
        })
        print("\n" + summary.round(5).to_string())


if __name__ == "__main__":
    main()
