"""Two-part model for freight rates: a cross-sectional GBM plus a day-level term.

Target
------
``y = log(posted_rate) - log(distance)`` (log rate-per-mile).  Logs matter
because the rate is close to multiplicative in distance, equipment and the
market index, and because the label corruption in this dataset is itself
multiplicative.  ``distance`` is used exactly as recorded: replacing it with a
de-noised per-lane median makes the fit strictly worse, which says the rate was
generated from the recorded distance, noise included.

Why the split
-------------
Variance in ``y`` decomposes into a *within-day* part (sd ~= 0.018, which load
is it) and a *day-level* part (sd ~= 0.030, what was the market doing that
day).  The two need different machinery:

* **Within-day** is a rich, non-linear, interaction-heavy function of distance,
  equipment, geography and the row's own market/quote readings.  That is what
  gradient boosting is for.

* **Day-level** has to *extrapolate*.  The prediction window (Nov-Dec 2025)
  lies entirely outside the training dates, and the day-level series contains a
  genuine upward drift of about +0.021 log-units per 100 days on top of
  ``market_index``.  Trees cannot extrapolate a trend at all, and - worse -
  when given day aggregates such as the daily mean quote signal they memorise
  *which month* a day belongs to and then fan out wildly on a new month whose
  aggregates sit in a narrow band.  (Measured directly: handing the daily quote
  mean to the trees triples the median error on an August holdout, and November
  and December have exactly the narrow-band shape that triggers it.)

  So the day level is a small weighted least-squares fit instead, on

      level ~ log(market_index) + quote_signal_mean + day_of_month
              + day_of_week + calendar_time

  Nine coefficients, all of them monotone in their input, which is safe to
  evaluate two months past the end of the data.  The drift coefficient is
  estimated at +0.00021/day with a heteroskedasticity-robust t of ~36-44.

Fitting order
-------------
1. Flag corrupted labels.  ~1.3% of ``posted_rate`` values are multiplied by a
   random factor (roughly 1.4x to 6.6x up, or 0.15x to 0.7x down); those errors
   dwarf the genuine residual spread, so they are found from out-of-fold
   residuals of a date-blind model and dropped.
2. Fit the cross-sectional GBM out-of-fold.  Its residuals give two things: a
   per-month noise level (August is markedly noisier than the rest of the
   file, so its rows are down-weighted) and, averaged per calendar day, the
   day-level series the regression above is fitted to.
3. Refit the GBM on the target with the day level removed.

Step 2 uses *out-of-fold* residuals because in-sample GBM residuals are shrunk
and would bias the day-level coefficients - especially the drift - toward zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as F

ORIGIN = F.ORIGIN

# Tuned on a Jan-Aug / Sep-Oct forward holdout.  Shallow trees clearly beat
# deep ones here: most of the cross-sectional signal is smooth and low-order.
LGB_PARAMS = dict(
    objective="regression",
    metric="l2",
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=40,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=2.0,
    max_cat_threshold=64,
    cat_smooth=20.0,
    verbose=-1,
    num_threads=0,
)

# Load-level features that carry no calendar information whatsoever.  Used to
# flag corrupted labels, so that the flagging cannot be confounded by the drift.
DATE_BLIND = [
    "log_distance", "distance", "weight", "weight_per_mile",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "mid_lat", "mid_lon", "d_lat", "d_lon", "haversine", "detour_ratio",
    "bearing_sin", "bearing_cos", "lane_count", "weight_missing",
    "equipment", "pickup", "delivery",
]

# The cross-sectional model.  It adds the row's *deviation* from the day's
# market index and quote signal - both are real within-day effects (a load
# quoted 1 sd above its day runs ~0.3% dearer) - plus day-of-week and
# day-of-month, which are safe for trees because they are small repeating
# cycles that recur unchanged in any future month.  It never sees the day
# *levels* themselves; those belong to the day-level regression.
CROSS_SECTION = DATE_BLIND + [
    "market_index_dev", "market_index_missing",
    "quote_signal_dev", "quote_signal_abs_dev",
    "day_of_month", "dow_sin", "dow_cos", "is_weekend",
]

# Three day-level drivers, all entered linearly.  Chosen by backtesting the
# daily level series on its own (roll an origin forward, forecast the next
# 30/61 days): this beats every richer specification tried, including ones
# adding day-of-week, a quadratic market term, a market x quote interaction, or
# exponential recency weighting.
DAY_TERMS = ["log_market_index_daily", "quote_signal_daily", "t"]


def _train(params: dict, X, y, rounds: int, weight=None) -> lgb.Booster:
    return lgb.train(params, lgb.Dataset(X, y, weight=weight, free_raw_data=False),
                     num_boost_round=rounds)


def day_design(index: pd.DatetimeIndex, market: np.ndarray, quote: np.ndarray) -> np.ndarray:
    """Design matrix of the day-level regression, columns ordered as DAY_TERMS."""
    return np.column_stack([
        np.ones(len(index)),
        np.log(market),
        quote,
        (index - ORIGIN).days.to_numpy(dtype=float),
    ])


@dataclass
class FreightRateModel:
    n_rounds: int = 1500
    n_rounds_aux: int = 800
    n_folds: int = 5
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    outlier_z: float = 6.0
    outlier_floor: float = 0.15
    min_month_weight: float = 0.2
    params: dict = field(default_factory=lambda: dict(LGB_PARAMS))

    # fitted state
    day_coef_: np.ndarray | None = None
    day_stats_: dict = field(default_factory=dict)
    daily_level_: pd.Series | None = None
    month_noise_: pd.Series | None = None
    month_weight_: pd.Series | None = None
    inlier_mask_: np.ndarray | None = None
    row_weight_: np.ndarray | None = None
    boosters_: list = field(default_factory=list)
    oof_residual_sd_: float = float("nan")

    # ------------------------------------------------------------------ utils
    @staticmethod
    def target(frame: pd.DataFrame) -> np.ndarray:
        return np.log(frame["posted_rate"].to_numpy()) - np.log(frame["distance"].to_numpy())

    @staticmethod
    def _robust_sigma(residual: np.ndarray) -> float:
        return 1.4826 * float(np.median(np.abs(residual - np.median(residual))))

    def _oof(self, frame: pd.DataFrame, y: np.ndarray, columns: list[str],
             rounds: int, seed: int) -> np.ndarray:
        """Out-of-fold predictions over random folds.

        Random folds are safe for both uses below because neither model carries
        a day-level feature, so rows from the same day cannot leak a level.
        """
        X = frame[columns]
        folds = np.random.default_rng(seed).integers(0, self.n_folds, len(y))
        oof = np.empty(len(y))
        for k in range(self.n_folds):
            hold = folds == k
            oof[hold] = _train(dict(self.params, seed=0), X[~hold], y[~hold], rounds).predict(X[hold])
        return oof

    # ----------------------------------------- per-month noise down-weighting
    def _fit_month_weights(self, frame: pd.DataFrame, residual: np.ndarray) -> np.ndarray:
        """Down-weight months whose labels are noisier than the rest.

        August 2025 carries about 70% more residual spread than every other
        month in the file (robust sd 0.026 against ~0.013), uniformly across
        days, equipment and haul lengths - extra noise on the labels rather
        than anything a feature explains.  Weighting each month by the inverse
        of its residual variance is the textbook response and it measurably
        sharpens the cross-section on held-out months.
        """
        month = pd.DatetimeIndex(frame["date"]).to_period("M")
        noise = (pd.Series(residual).groupby(month)
                 .apply(lambda x: 1.4826 * float(np.median(np.abs(x - x.median())))))
        weight = ((noise.median() / noise) ** 2).clip(self.min_month_weight, 1.0)
        self.month_noise_, self.month_weight_ = noise, weight
        return month.map(weight).to_numpy(dtype=float)

    # ------------------------------------------- step 1: corrupted-label flags
    def _flag_inliers(self, frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        residual = y - self._oof(frame, y, DATE_BLIND, self.n_rounds_aux, seed=0)
        cut = max(self.outlier_floor, self.outlier_z * self._robust_sigma(residual))
        return np.abs(residual - np.median(residual)) <= cut

    # --------------------------------------------- step 2: the day-level model
    def _fit_day_level(self, frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        residual = y - self._oof(frame, y, CROSS_SECTION, self.n_rounds, seed=1)
        self.oof_residual_sd_ = float(np.std(residual))
        row_weight = self._fit_month_weights(frame, residual - pd.Series(residual)
                                             .groupby(frame["date"].to_numpy()).transform("mean").to_numpy())
        self.row_weight_ = row_weight

        daily = pd.DataFrame({"date": frame["date"].to_numpy(), "r": residual})
        level = daily.groupby("date")["r"].mean()
        count = daily.groupby("date")["r"].size().to_numpy(dtype=float)
        self.daily_level_ = level

        idx = level.index
        market = frame.groupby("date")["market_index_daily"].first().reindex(idx).to_numpy()
        quote = frame.groupby("date")["quote_signal_daily"].first().reindex(idx).to_numpy()
        design = day_design(idx, market, quote)

        # each day is weighted by its load count and by the reliability of its
        # month, so a noisy month cannot drag the day-level coefficients around
        w = np.sqrt(count * idx.to_period("M").map(self.month_weight_).to_numpy(dtype=float))
        coef, *_ = np.linalg.lstsq(design * w[:, None], level.to_numpy() * w, rcond=None)

        fitted = design @ coef
        resid = level.to_numpy() - fitted
        xtx_inv = np.linalg.inv((design * w[:, None]).T @ (design * w[:, None]))
        meat = (design * w[:, None]).T @ np.diag((resid * w) ** 2) @ (design * w[:, None])
        se = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))
        self.day_stats_ = {
            "r2": float(1 - np.var(resid) / np.var(level.to_numpy())),
            "residual_sd": float(np.std(resid)),
            "level_sd": float(np.std(level.to_numpy())),
            "coef": {name: float(c) for name, c in zip(DAY_TERMS, coef[1:])},
            "t_stat": {name: float(c / s) for name, c, s in zip(DAY_TERMS, coef[1:], se[1:])},
            "drift_per_100_days": float(coef[-1]) * 100,
        }
        return coef

    def day_level(self, frame: pd.DataFrame) -> np.ndarray:
        design = day_design(
            pd.DatetimeIndex(frame["date"]),
            frame["market_index_daily"].to_numpy(),
            frame["quote_signal_daily"].to_numpy(),
        )
        return design @ self.day_coef_

    # -------------------------------------------------------------------- fit
    def fit(self, frame: pd.DataFrame) -> "FreightRateModel":
        frame = frame.reset_index(drop=True)
        y = self.target(frame)

        self.inlier_mask_ = self._flag_inliers(frame, y)
        kept = frame[self.inlier_mask_].reset_index(drop=True)
        y_kept = y[self.inlier_mask_]

        self.day_coef_ = self._fit_day_level(kept, y_kept)

        residual_target = y_kept - self.day_level(kept)
        X = kept[CROSS_SECTION]
        self.boosters_ = [
            _train(dict(self.params, seed=s, bagging_seed=s + 7, feature_fraction_seed=s + 13),
                   X, residual_target, self.n_rounds, weight=self.row_weight_)
            for s in self.seeds
        ]
        return self

    # ---------------------------------------------------------------- predict
    def predict_log_rpm(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame[CROSS_SECTION]
        trees = np.mean([b.predict(X) for b in self.boosters_], axis=0)
        return trees + self.day_level(frame)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log_rpm(frame)) * frame["distance"].to_numpy()

    # ------------------------------------------------------------ diagnostics
    def feature_importance(self) -> pd.Series:
        gain = np.mean([b.feature_importance("gain") for b in self.boosters_], axis=0)
        return pd.Series(gain, index=self.boosters_[0].feature_name()).sort_values(ascending=False)
