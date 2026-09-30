"""Loading and cleaning for the freight-rate dataset.

Data-quality issues handled here (see report for evidence):
  1. ``weight`` has sign-flipped values (~0.6% negative, |w| distribution is
     identical to the positive one)  -> take the absolute value.
  2. ``weight`` is missing for ~0.6% of rows          -> median impute + flag.
  3. ``market_index`` is missing for ~0.8% of rows.  It is a *market-wide daily*
     index (within-day sd 0.025 vs overall sd 0.166), so the same-day mean of
     the other rows is a near-exact reconstruction -> impute from the day, then
     fall back to a global median.  This uses only feature columns (never the
     label), so it is safe to compute over train+validation together.
  4. ``posted_rate`` carries multiplicative label corruption on ~1.3% of rows
     (a random factor of roughly 2.3-6.6x up or 0.16-0.46x down).  These are
     *labels*, so they are handled in the model layer, not here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "load_id", "pickup", "delivery", "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon", "distance", "equipment", "weight",
    "date", "market_index", "quote_signal",
]


def load_raw(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_test, validation) with parsed dates."""
    train = pd.read_csv(data_dir / "train_test.csv", parse_dates=["date"])
    valid = pd.read_csv(data_dir / "validation.csv", parse_dates=["date"])
    return train, valid


def _daily_index_table(frames: list[pd.DataFrame]) -> pd.Series:
    """Daily mean of ``market_index`` pooled over every frame supplied."""
    pooled = pd.concat([f[["date", "market_index"]] for f in frames], ignore_index=True)
    return pooled.groupby("date")["market_index"].mean()


def clean(frame: pd.DataFrame, daily_index: pd.Series, weight_median: float) -> pd.DataFrame:
    out = frame.copy()

    # 1 + 2: weight
    out["weight_missing"] = out["weight"].isna().astype(np.int8)
    out["weight_sign_flipped"] = (out["weight"] < 0).fillna(False).astype(np.int8)
    out["weight"] = out["weight"].abs()
    out["weight"] = out["weight"].fillna(weight_median)

    # 3: market index
    out["market_index_missing"] = out["market_index"].isna().astype(np.int8)
    day_fill = out["date"].map(daily_index)
    out["market_index"] = out["market_index"].fillna(day_fill)
    out["market_index"] = out["market_index"].fillna(daily_index.median())

    # daily (de-noised) market level and the row-level deviation from it
    out["market_index_daily"] = out["date"].map(daily_index).fillna(out["market_index"])
    out["market_index_dev"] = out["market_index"] - out["market_index_daily"]
    return out


def prepare(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean both frames using statistics pooled from feature columns only."""
    train, valid = load_raw(data_dir)
    daily_index = _daily_index_table([train, valid])
    weight_median = float(pd.concat([train["weight"], valid["weight"]]).abs().median())
    return (
        clean(train, daily_index, weight_median),
        clean(valid, daily_index, weight_median),
    )
