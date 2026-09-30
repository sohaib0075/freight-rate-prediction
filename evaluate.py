"""Validation harness: forward-in-time holdouts on the development data.

Why not K-fold
--------------
The task is to predict Nov-Dec 2025 from Jan-Oct 2025, so every holdout here is
a *future* block of calendar time.  A random K-fold would put loads from the
same day on both sides of the split; because roughly half the variance in log
rate-per-mile is a day-level market effect, that leaks the answer and reports a
score the model cannot reproduce on a genuinely unseen month.  It would also
hide the upward drift in the series entirely.

Three views are produced:

``forward``   train Jan-Aug, score Sep-Oct - one 2-month block, the same
              horizon as the real Nov-Dec task.
``rolling``   expanding-origin backtest with four successive 1-month holdouts
              (Jul, Aug, Sep, Oct), which shows how stable the estimate is.
``decomposition``
              each holdout's log error split into the day-level part (mean
              error per calendar day - can the model place the market?) and the
              within-day part (everything else - can it price a load?).  These
              are the two halves the model treats separately, so they are
              reported separately.

Metrics are reported on every holdout row, and a median-based pair
(MedAPE, MedAE) is reported alongside the means.  ~1.3% of labels are corrupted
by a large multiplicative factor; a median is unaffected by them, whereas
RMSE on a contaminated holdout is dominated by them and says little about the
model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import data as D
from src import features as F
from src.model import FreightRateModel


def metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - truth
    pct = err / truth
    log_err = np.log(pred) - np.log(truth)
    return {
        "n": int(len(truth)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(np.mean(np.abs(pct)) * 100),
        "MedAPE_%": float(np.median(np.abs(pct)) * 100),
        "MedAE": float(np.median(np.abs(err))),
        "R2": float(1 - np.sum(err ** 2) / np.sum((truth - truth.mean()) ** 2)),
        "median_bias_%": float(np.median(pct) * 100),
        "p90_abs_log_error": float(np.quantile(np.abs(log_err), 0.90)),
    }


def decompose(dates: np.ndarray, log_err: np.ndarray) -> dict:
    """Split log error into a day-level and a within-day component."""
    series = pd.Series(log_err, index=dates)
    day = series.groupby(level=0).median()
    within = series - series.groupby(level=0).transform("median")
    return {
        "day_level_mean": float(day.mean()),
        "day_level_sd": float(day.std()),
        "day_level_rmse": float(np.sqrt(np.mean(day.to_numpy() ** 2))),
        "within_day_MedAE": float(np.median(np.abs(within))),
    }


def score_split(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    model = FreightRateModel().fit(train)
    pred = model.predict(test)
    truth = test["posted_rate"].to_numpy()
    out = metrics(truth, pred)
    out.update(decompose(test["date"].to_numpy(), np.log(pred) - np.log(truth)))
    out["drift_per_100_days"] = model.day_stats_["drift_per_100_days"]
    out["day_level_r2"] = model.day_stats_["r2"]
    out["rows_dropped_as_corrupt_%"] = 100 * (1 - float(model.inlier_mask_.mean()))
    return out


def show(label: str, r: dict) -> None:
    print(f"{label:16s} MedAPE {r['MedAPE_%']:5.3f}%  MAPE {r['MAPE_%']:5.3f}%  "
          f"MedAE ${r['MedAE']:6.2f}  MAE ${r['MAE']:7.2f}  R2 {r['R2']:6.3f}  "
          f"day-rmse {r['day_level_rmse']:.4f}  within {r['within_day_MedAE']:.4f}  "
          f"bias {r['median_bias_%']:+.3f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="report/validation_metrics.json")
    args = ap.parse_args()

    train_raw, valid_raw = D.prepare(Path(args.data_dir))
    ref = F.build_reference([train_raw, valid_raw])
    dev = F.transform(train_raw, ref)

    results: dict = {}

    cut = pd.Timestamp("2025-09-01")
    print(f"forward holdout: train < {cut.date()} ({int((dev.date < cut).sum()):,} rows), "
          f"test >= {cut.date()} ({int((dev.date >= cut).sum()):,} rows)")
    results["forward_sep_oct"] = score_split(dev[dev.date < cut], dev[dev.date >= cut])
    show("Sep-Oct", results["forward_sep_oct"])

    print("\nrolling origin, one month at a time:")
    rolling = {}
    for month in [7, 8, 9, 10]:
        start = pd.Timestamp(f"2025-{month:02d}-01")
        end = start + pd.offsets.MonthBegin(1)
        rolling[f"2025-{month:02d}"] = score_split(
            dev[dev.date < start], dev[(dev.date >= start) & (dev.date < end)])
        show(f"  {start:%b}", rolling[f"2025-{month:02d}"])
    results["rolling"] = rolling
    results["rolling_mean"] = {
        k: float(np.mean([rolling[m][k] for m in rolling]))
        for k in ["MedAPE_%", "MAPE_%", "MedAE", "MAE", "R2", "day_level_rmse", "within_day_MedAE"]
    }
    print("rolling mean :", {k: round(v, 4) for k, v in results["rolling_mean"].items()})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
