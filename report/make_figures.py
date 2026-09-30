"""Figures for the written report.

    python report/make_figures.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D                                     # noqa: E402
from src import features as F                                 # noqa: E402
from src.model import CROSS_SECTION, FreightRateModel, day_design   # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
INK, ACCENT, GREY = "#064A56", "#C2703D", "#8FA3A8"
plt.rcParams.update({"figure.dpi": 170, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#E3EAEC", "grid.linewidth": 0.7,
                     "axes.edgecolor": "#9DAFB3"})


def fig_quality(train_raw: pd.DataFrame) -> None:
    """Weight sign flips, and the multiplicative corruption in posted_rate."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.1))

    w = train_raw["weight"]
    bins = np.linspace(5000, 47500, 45)
    a.hist(w[w > 0], bins=bins, density=True, color=INK, alpha=0.75,
           label=f"positive (n={int((w > 0).sum()):,})")
    a.hist(-w[w < 0], bins=bins, density=True, histtype="step", color=ACCENT, linewidth=1.8,
           label=f"negated negatives (n={int((w < 0).sum()):,})")
    a.set_title("weight: the negatives are sign flips", loc="left", fontweight="bold")
    a.set_xlabel("pounds"); a.set_ylabel("density"); a.legend(frameon=False, fontsize=7.5)

    lrpm = np.log(train_raw["posted_rate"] / train_raw["distance"])
    centred = lrpm - lrpm.median()
    b.hist(centred, bins=240, color=INK)
    b.set_yscale("log")
    b.axvspan(-0.28, 0.28, color=GREY, alpha=0.25)
    b.set_title("log rate-per-mile, centred: corruption sits far outside the body",
                loc="left", fontweight="bold")
    b.set_xlabel("log deviation from the median"); b.set_ylabel("loads (log scale)")
    b.text(0.02, 0.86, "shaded = kept", transform=b.transAxes, color="#455A60", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "data_quality.png", bbox_inches="tight"); plt.close(fig)


def fig_month_noise(model: FreightRateModel) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    noise = model.month_noise_
    labels = [str(p)[-2:] for p in noise.index]
    colors = [ACCENT if v > 1.3 * noise.median() else INK for v in noise]
    ax.bar(labels, noise.to_numpy(), color=colors)
    ax.axhline(float(noise.median()), color=GREY, linestyle="--", linewidth=1)
    ax.set_title("August labels are ~70% noisier than every other month",
                 loc="left", fontweight="bold")
    ax.set_xlabel("month of 2025"); ax.set_ylabel("robust residual sd (log)")
    fig.tight_layout(); fig.savefig(OUT / "month_noise.png", bbox_inches="tight"); plt.close(fig)


def fig_day_level(model: FreightRateModel, train: pd.DataFrame, valid: pd.DataFrame) -> None:
    """Observed daily level, the fitted day model, and the Nov-Dec extension."""
    level = model.daily_level_
    idx = level.index
    market = train.groupby("date")["market_index_daily"].first().reindex(idx).to_numpy()
    quote = train.groupby("date")["quote_signal_daily"].first().reindex(idx).to_numpy()
    fitted = day_design(idx, market, quote) @ model.day_coef_

    vidx = pd.DatetimeIndex(sorted(valid["date"].unique()))
    vmarket = valid.groupby("date")["market_index_daily"].first().reindex(vidx).to_numpy()
    vquote = valid.groupby("date")["quote_signal_daily"].first().reindex(vidx).to_numpy()
    vfitted = day_design(vidx, vmarket, vquote) @ model.day_coef_

    drift = model.day_coef_[-1]
    t_all = np.r_[(idx - F.ORIGIN).days.to_numpy(), (vidx - F.ORIGIN).days.to_numpy()]
    baseline = np.r_[fitted, vfitted].mean() + drift * (t_all - t_all.mean())

    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    ax.plot(idx, level.to_numpy(), color=GREY, linewidth=0.8, label="observed daily level")
    ax.plot(idx, fitted, color=INK, linewidth=1.9, label="day model (fitted, Jan–Oct)")
    ax.plot(vidx, vfitted, color=ACCENT, linewidth=1.9, label="day model (applied, Nov–Dec)")
    ax.plot(np.r_[idx, vidx], baseline, color="#455A60", linewidth=1.0, linestyle=":",
            label=f"drift only ({drift * 100:+.3f}/100 days)")
    ax.axvline(pd.Timestamp("2025-11-01"), color="#455A60", linewidth=0.9, linestyle="--")
    ax.text(pd.Timestamp("2025-11-02"), ax.get_ylim()[1] * 0.92, "training ends",
            fontsize=8, color="#455A60")
    ax.set_title("The day-level term: what the model has to extrapolate",
                 loc="left", fontweight="bold")
    ax.set_ylabel("log rate-per-mile, level")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "day_level.png", bbox_inches="tight"); plt.close(fig)


def fig_holdout(metrics: dict) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.0))
    roll = metrics["rolling"]
    months = list(roll)
    labels = [pd.Period(m).strftime("%b") for m in months]
    x = np.arange(len(months))

    a.bar(x - 0.19, [roll[m]["MedAPE_%"] for m in months], 0.36, color=INK, label="MedAPE")
    a.bar(x + 0.19, [roll[m]["MAPE_%"] for m in months], 0.36, color=INK, alpha=0.32, label="MAPE")
    a.set_title("Rolling-origin holdouts", loc="left", fontweight="bold")
    a.set_ylabel("% error"); a.set_xticks(x); a.set_xticklabels(labels)
    a.legend(frameon=False, fontsize=8)

    b.bar(x - 0.19, [roll[m]["day_level_rmse"] for m in months], 0.36, color=ACCENT,
          label="day-level RMSE")
    b.bar(x + 0.19, [roll[m]["within_day_MedAE"] for m in months], 0.36, color=INK,
          label="within-day MedAE")
    b.set_title("Where the error lives (log units)", loc="left", fontweight="bold")
    b.set_xticks(x); b.set_xticklabels(labels)
    b.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "holdouts.png", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-model", action="store_true",
                        help="only rebuild the figures that need no model fit")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fig_quality(pd.read_csv("data/train_test.csv", parse_dates=["date"]))

    metrics_path = Path("report/validation_metrics.json")
    if metrics_path.is_file():
        fig_holdout(json.loads(metrics_path.read_text()))
    else:
        print("report/validation_metrics.json missing - run evaluate.py first")

    if not args.skip_model:
        train_raw, valid_raw = D.prepare(Path("data"))
        ref = F.build_reference([train_raw, valid_raw])
        train, valid = F.transform(train_raw, ref), F.transform(valid_raw, ref)
        model = FreightRateModel().fit(train)
        fig_month_noise(model)
        fig_day_level(model, train[model.inlier_mask_], valid)

    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
