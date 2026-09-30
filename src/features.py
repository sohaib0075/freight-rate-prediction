"""Feature engineering.

This module only *builds* columns; which of them the model consumes is decided
in ``model.py`` (``CROSS_SECTION`` for the trees, ``DAY_TERMS`` for the
day-level regression), because that split is a modelling decision rather than a
data one.

Design notes
------------
* The target is ``log(posted_rate / distance)`` (log rate-per-mile).  The rate
  was generated from the *recorded* distance, not from a true one: replacing
  ``distance`` with the per-lane median makes the fit strictly worse (residual
  sd 0.0493 against 0.0456), so distance is used exactly as reported.
* ~12% of validation rows use cities that never appear in training, so every
  geographic signal is expressed in coordinates (lat/lon, midpoint, bearing,
  great-circle distance) as well as in the raw city names.  The coordinates let
  the model interpolate to the unseen markets; the city categoricals only
  sharpen lanes it has actually seen.
* The two market series are each split into a *daily level* and the row's
  *deviation* from it.  Both halves are real -- the level moves the whole day,
  a load quoted above its own day runs dearer -- but they are consumed by
  different parts of the model, so they are separated here.
* Calendar features are restricted to *repeating* cycles (day-of-week,
  day-of-month).  Month, day-of-year and the time index are deliberately kept
  out of the tree stage: November and December lie outside the training range
  and a tree cannot extrapolate.  The drift over calendar time is carried by
  an explicit linear term instead (see ``model.py``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_MI = 3958.8
ORIGIN = pd.Timestamp("2025-01-01")

CATEGORICAL = ["equipment", "pickup", "delivery"]


def haversine_miles(lat1, lon1, lat2, lon2):
    rad = np.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bearing(lat1, lon1, lat2, lon2):
    rad = np.pi / 180.0
    dlon = (lon2 - lon1) * rad
    y = np.sin(dlon) * np.cos(lat2 * rad)
    x = np.cos(lat1 * rad) * np.sin(lat2 * rad) - np.sin(lat1 * rad) * np.cos(lat2 * rad) * np.cos(dlon)
    return np.arctan2(y, x)


def build_reference(frames: list[pd.DataFrame]) -> dict:
    """Reference tables computed from feature columns only (no label involved)."""
    pooled = pd.concat(frames, ignore_index=True)
    city_coords = pd.concat([
        pooled[["pickup", "pickup_lat", "pickup_lon"]]
        .rename(columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}),
        pooled[["delivery", "delivery_lat", "delivery_lon"]]
        .rename(columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}),
    ]).groupby("city")[["lat", "lon"]].first()
    return {
        "lane_count": pooled.groupby(["pickup", "delivery"]).size().rename("lane_count"),
        # the *mean*, not the median: with ~150 loads a day the mean is the
        # lower-variance estimate of the day's quote level, and that level
        # carries a real coefficient on log rate-per-mile (+0.047, t = 13), so
        # the extra noise in a median measurably degrades the fit.
        "quote_daily": pooled.groupby("date")["quote_signal"].mean(),
        "city_coords": city_coords,
        "categories": {c: pd.Index(sorted(pooled[c].dropna().unique())) for c in CATEGORICAL},
    }


def transform(frame: pd.DataFrame, ref: dict) -> pd.DataFrame:
    f = frame.copy()

    f["log_distance"] = np.log(f["distance"])
    f["weight_per_mile"] = f["weight"] / f["distance"]

    f["quote_signal_daily"] = f["date"].map(ref["quote_daily"])
    f["quote_signal_daily"] = f["quote_signal_daily"].fillna(f["quote_signal"])
    f["quote_signal_dev"] = f["quote_signal"] - f["quote_signal_daily"]
    f["quote_signal_abs_dev"] = f["quote_signal_dev"].abs()

    f["mid_lat"] = (f["pickup_lat"] + f["delivery_lat"]) / 2
    f["mid_lon"] = (f["pickup_lon"] + f["delivery_lon"]) / 2
    f["d_lat"] = f["delivery_lat"] - f["pickup_lat"]
    f["d_lon"] = f["delivery_lon"] - f["pickup_lon"]
    f["haversine"] = haversine_miles(f["pickup_lat"], f["pickup_lon"], f["delivery_lat"], f["delivery_lon"])
    f["detour_ratio"] = f["distance"] / f["haversine"].replace(0, np.nan)
    brg = _bearing(f["pickup_lat"], f["pickup_lon"], f["delivery_lat"], f["delivery_lon"])
    f["bearing_sin"], f["bearing_cos"] = np.sin(brg), np.cos(brg)

    dow = f["date"].dt.dayofweek
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    f["day_of_month"] = f["date"].dt.day
    f["is_weekend"] = (dow >= 5).astype(np.int8)

    f = f.merge(ref["lane_count"], on=["pickup", "delivery"], how="left")
    f["lane_count"] = f["lane_count"].fillna(0)

    for col in CATEGORICAL:
        f[col] = pd.Categorical(f[col], categories=ref["categories"][col])

    return f
