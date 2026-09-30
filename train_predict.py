"""Train the freight-rate model and write both required prediction files.

    python train_predict.py                     # uses ./data, writes ./validation_predictions.csv
    python train_predict.py --data-dir data --out validation_predictions.csv

Outputs
-------
validation_predictions.csv          load_id,predicted_rate for all 12,000 loads
data/december_chart_inputs.csv      the same file with predicted_rate filled in
report/model_summary.json           fitted trend, drop rate, feature importance
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

FIXED_PICKUP, FIXED_DELIVERY = "Lexington", "Fort Wayne"


def build_december_frame(december: pd.DataFrame, valid: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Give the 31 December chart rows the columns the model expects.

    The chart file carries only pickup / delivery / distance / equipment /
    weight / date, so three things have to be supplied:

    * **Coordinates** - looked up from the city table built off the supplied
      data (every city's lat/lon is a single fixed pair).
    * **market_index and quote_signal** - both are *daily market* series and
      both are given for every December date inside ``validation.csv``, so the
      day's mean over the validation loads is used.  Nothing about the label is
      involved; these are input columns of the December window that the task
      hands us.
    * **weight/market-index missing flags** - zero, the values are complete.
    """
    frame = december.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    coords = ref["city_coords"]
    frame["pickup_lat"] = coords.loc[FIXED_PICKUP, "lat"]
    frame["pickup_lon"] = coords.loc[FIXED_PICKUP, "lon"]
    frame["delivery_lat"] = coords.loc[FIXED_DELIVERY, "lat"]
    frame["delivery_lon"] = coords.loc[FIXED_DELIVERY, "lon"]

    daily_market = valid.groupby("date")["market_index"].mean()
    daily_quote = valid.groupby("date")["quote_signal"].mean()
    missing = sorted(set(frame["date"]) - set(daily_market.index))
    if missing:
        raise SystemExit(f"no market data for December dates: {missing[:5]}")
    frame["market_index"] = frame["date"].map(daily_market)
    frame["quote_signal"] = frame["date"].map(daily_quote)
    frame["market_index_daily"] = frame["market_index"]
    frame["market_index_dev"] = 0.0
    frame["weight_missing"] = np.int8(0)
    frame["weight_sign_flipped"] = np.int8(0)
    frame["market_index_missing"] = np.int8(0)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="validation_predictions.csv")
    ap.add_argument("--december-out", default=None,
                    help="defaults to <data-dir>/december_chart_inputs.csv (edited in place)")
    ap.add_argument("--summary", default="report/model_summary.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    december_out = Path(args.december_out or data_dir / "december_chart_inputs.csv")

    train_raw, valid_raw = D.prepare(data_dir)
    ref = F.build_reference([train_raw, valid_raw])
    train = F.transform(train_raw, ref)
    valid = F.transform(valid_raw, ref)

    print(f"train {len(train):,} rows  {train.date.min().date()} .. {train.date.max().date()}")
    print(f"valid {len(valid):,} rows  {valid.date.min().date()} .. {valid.date.max().date()}")

    model = FreightRateModel().fit(train)
    stats = model.day_stats_
    print(f"drift {stats['drift_per_100_days']:+.4f} log-units / 100 days "
          f"(t = {stats['t_stat']['t']:.1f})")
    print(f"day-level R2 {stats['r2']:.3f}, residual sd {stats['residual_sd']:.5f}")
    print("day-level coefficients: "
          + json.dumps({k: round(v, 5) for k, v in stats['coef'].items()}))
    print("month weights: "
          + json.dumps({str(k): round(float(v), 3) for k, v in model.month_weight_.items()}))
    print(f"dropped as corrupt labels: {100 * (1 - model.inlier_mask_.mean()):.2f}%")
    print(f"repaired sign-flipped weights: {int(train.weight_sign_flipped.sum()):,}; "
          f"imputed weights: {int(train.weight_missing.sum()):,}; "
          f"imputed market_index: {int(train.market_index_missing.sum()):,}")
    print(f"out-of-fold residual sd (log): {model.oof_residual_sd_:.5f}")

    # ---- 12,000 validation loads -------------------------------------------
    template = pd.read_csv(data_dir / "validation_predictions_template.csv")
    predictions = pd.Series(model.predict(valid), index=valid["load_id"].to_numpy())
    template["predicted_rate"] = template["load_id"].map(predictions).round(2)
    if template["predicted_rate"].isna().any():
        raise SystemExit("some template load_ids were not predicted")
    template.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(template):,} rows, "
          f"mean ${template.predicted_rate.mean():,.2f}, "
          f"min ${template.predicted_rate.min():,.2f}, max ${template.predicted_rate.max():,.2f})")

    # ---- 31 fixed December rows --------------------------------------------
    december_raw = pd.read_csv(data_dir / "december_chart_inputs.csv")
    december = F.transform(build_december_frame(december_raw, valid_raw, ref), ref)
    december_raw["predicted_rate"] = np.round(model.predict(december), 2)
    december_raw.to_csv(december_out, index=False)
    print(f"wrote {december_out}  (31 rows, ${december_raw.predicted_rate.min():,.2f} .. "
          f"${december_raw.predicted_rate.max():,.2f})")

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps({
        "day_level_fit": model.day_stats_,
        "month_noise_sd": {str(k): float(v) for k, v in model.month_noise_.items()},
        "month_weight": {str(k): float(v) for k, v in model.month_weight_.items()},
        "rows_dropped_as_corrupt_pct": 100 * (1 - float(model.inlier_mask_.mean())),
        "n_train_rows_used": int(model.inlier_mask_.sum()),
        "oof_residual_sd_log": model.oof_residual_sd_,
        "repairs": {
            "weight_sign_flipped": int(train.weight_sign_flipped.sum()),
            "weight_imputed": int(train.weight_missing.sum()),
            "market_index_imputed": int(train.market_index_missing.sum()),
        },
        "feature_importance_gain": model.feature_importance().round(1).to_dict(),
        "validation_prediction_summary": {
            "mean": float(template.predicted_rate.mean()),
            "median": float(template.predicted_rate.median()),
            "min": float(template.predicted_rate.min()),
            "max": float(template.predicted_rate.max()),
        },
        "december": dict(zip(december_raw.date, december_raw.predicted_rate.astype(float))),
    }, indent=2))
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
