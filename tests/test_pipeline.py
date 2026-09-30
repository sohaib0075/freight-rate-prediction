"""Fast checks on the pipeline.  Run with `python -m pytest tests -q`.

These are smoke and invariant tests, not a substitute for `evaluate.py`: they
fit the model on a small subsample so the suite finishes in well under a
minute.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data as D
from src import features as F
from src.model import CROSS_SECTION, FreightRateModel, day_design
from train_predict import build_december_frame

DATA = Path("data")


@pytest.fixture(scope="module")
def prepared():
    train_raw, valid_raw = D.prepare(DATA)
    ref = F.build_reference([train_raw, valid_raw])
    return train_raw, valid_raw, ref


def test_cleaning_repairs_weight_and_market_index(prepared):
    train_raw, valid_raw, _ = prepared
    raw = pd.read_csv(DATA / "train_test.csv")
    for cleaned in (train_raw, valid_raw):
        assert (cleaned["weight"] > 0).all()
        assert cleaned["weight"].notna().all()
        assert cleaned["market_index"].notna().all()
    # the flags record what was repaired rather than silently hiding it
    assert train_raw["weight_sign_flipped"].sum() == (raw["weight"] < 0).sum()
    assert train_raw["weight_missing"].sum() == raw["weight"].isna().sum()
    # |w| is used, not a re-imputation: a known sign-flipped row keeps its size
    flipped = raw.index[raw["weight"] < 0][0]
    assert train_raw.loc[flipped, "weight"] == pytest.approx(-raw.loc[flipped, "weight"])


def test_market_index_imputation_uses_the_same_day(prepared):
    train_raw, _, _ = prepared
    raw = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    gap = raw.index[raw["market_index"].isna()][0]
    same_day = raw.loc[raw["date"] == raw.loc[gap, "date"], "market_index"].mean()
    assert train_raw.loc[gap, "market_index"] == pytest.approx(same_day)


def test_every_model_feature_exists_on_every_frame(prepared):
    train_raw, valid_raw, ref = prepared
    december = build_december_frame(
        pd.read_csv(DATA / "december_chart_inputs.csv"), valid_raw, ref)
    for raw in (train_raw, valid_raw, december):
        frame = F.transform(raw, ref)
        missing = [c for c in CROSS_SECTION if c not in frame.columns]
        assert not missing, missing
        assert frame[[c for c in CROSS_SECTION if frame[c].dtype.kind == "f"]].notna().all().all()


def test_categories_are_shared_so_unseen_cities_do_not_shift_codes(prepared):
    train_raw, valid_raw, ref = prepared
    train, valid = F.transform(train_raw, ref), F.transform(valid_raw, ref)
    assert list(train["pickup"].cat.categories) == list(valid["pickup"].cat.categories)
    assert set(valid_raw["pickup"]) - set(train_raw["pickup"]), "expected unseen cities in validation"


def test_day_level_is_linear_and_monotone_in_calendar_time(prepared):
    """The day term must extrapolate smoothly - that is the whole point of it."""
    dates = pd.DatetimeIndex(pd.date_range("2025-11-01", "2025-12-31"))
    design = day_design(dates, np.full(len(dates), 1.0), np.full(len(dates), 2.05))
    assert design.shape == (len(dates), 4)
    model = FreightRateModel()
    model.day_coef_ = np.array([0.0, 0.15, 0.05, 0.0002])
    level = design @ model.day_coef_
    steps = np.diff(level)
    assert np.allclose(steps, steps[0])          # constant slope
    assert steps[0] > 0


def test_model_fits_and_predicts_positive_finite_rates(prepared):
    train_raw, valid_raw, ref = prepared
    train = F.transform(train_raw, ref)
    valid = F.transform(valid_raw, ref)
    small = FreightRateModel(n_rounds=60, n_rounds_aux=60, n_folds=3, seeds=(0,))
    small.fit(train.sample(6000, random_state=0))

    pred = small.predict(valid.head(500))
    assert np.isfinite(pred).all()
    assert (pred > 0).all()
    # a sane model prices a load somewhere near the going rate per mile
    rpm = pred / valid.head(500)["distance"].to_numpy()
    assert 1.0 < float(np.median(rpm)) < 4.0

    assert 0 < 100 * (1 - small.inlier_mask_.mean()) < 6      # corrupt-label share
    assert set(small.day_stats_["coef"]) == {"log_market_index_daily",
                                             "quote_signal_daily", "t"}


def test_december_frame_is_the_fixed_load_on_every_december_day(prepared):
    _, valid_raw, ref = prepared
    december = build_december_frame(
        pd.read_csv(DATA / "december_chart_inputs.csv"), valid_raw, ref)
    assert len(december) == 31
    assert december["market_index"].notna().all()
    assert december["quote_signal"].notna().all()
    # market conditions must actually vary across the month, or the chart is flat
    assert december["market_index"].nunique() == 31
    # coordinates come from the city table, so the lane must resolve
    assert december[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].notna().all().all()


def test_submission_files_match_the_scorer_contract():
    predictions = pd.read_csv("validation_predictions.csv")
    assert list(predictions.columns) == ["load_id", "predicted_rate"]
    assert len(predictions) == 12_000
    assert not predictions["load_id"].duplicated().any()
    assert (predictions["predicted_rate"] > 0).all()
    assert set(predictions["load_id"]) == {f"TE-{i:06d}" for i in range(1, 12_001)}

    december = pd.read_csv(DATA / "december_chart_inputs.csv")
    assert list(december.columns) == ["pickup", "delivery", "distance", "equipment",
                                      "weight", "date", "predicted_rate"]
    assert len(december) == 31
    assert (december["predicted_rate"] > 0).all()
