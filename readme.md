# Freight Rate Prediction — ML Engineer Assessment

Predicts `posted_rate` for 12,000 freight loads (Nov–Dec 2025) from 48,000
labelled loads (Jan–Oct 2025), plus a fixed-load December rate curve.

## Results

Forward-in-time holdout — train Jan–Aug, score the unseen Sep–Oct block
(9,523 loads). The two rows differ only in whether the ~1.3% of rows with
corrupted labels are included; those labels are unpredictable by construction
and dominate any squared error.

| Sep–Oct holdout | MAE | RMSE | MAPE | MedAPE | R² |
|---|---|---|---|---|---|
| all 9,523 rows | $86.25 | $628.91 | 3.79% | 1.06% | 0.830 |
| labels passing a corruption screen (9,248) | **$30.13** | **$44.54** | **1.27%** | **1.05%** | **0.9990** |

Rolling-origin backtest (each month trained only on what precedes it):

| holdout | MedAPE | MedAE | median bias | day-level RMSE | within-day MedAE |
|---|---|---|---|---|---|
| July | 1.68% | $32.16 | +0.26% | 0.0238 | 0.0093 |
| August¹ | 3.72% | $66.68 | +2.22% | 0.0242 | 0.0315 |
| September | 1.01% | $18.68 | −0.30% | 0.0079 | 0.0087 |
| October | 1.09% | $20.56 | −0.57% | 0.0091 | 0.0088 |

¹ August labels carry ~70% more noise than any other month in the file — see
the report. It is a property of the data, not of the model.

Full numbers: [`report/validation_metrics.json`](report/validation_metrics.json),
[`report/screened_metrics.json`](report/screened_metrics.json).
Write-up: **[`report/freight_rate_report.pdf`](report/freight_rate_report.pdf)**.

## Approach in one paragraph

The target is log rate-per-mile. Variance splits into a *within-day* part
(which load is it?) and a *day-level* part (what was the market doing?), and
the two get different machinery. A LightGBM model handles the cross-section.
The day level is a four-coefficient weighted least-squares fit on
`log(market_index) + quote_signal (daily mean) + calendar time`, because it is
the only part that has to extrapolate: Nov–Dec lies entirely past the last
training date, the series drifts upward by +0.0147 log-units per 100 days, and
trees cannot extrapolate a trend — worse, given day-level aggregates they
memorise *which month* a day belongs to and then fan out on a new month.
~1.3% of `posted_rate` labels are corrupted by a large multiplicative factor
and are detected out-of-fold and dropped before the final fit.

## Quick start

```bash
python -m pip install -r requirements.txt
```

```bash
python train_predict.py
```

That writes `validation_predictions.csv` and fills `predicted_rate` in
`data/december_chart_inputs.csv` (~5 minutes on a laptop). Then run the
provided scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

which validates both files and writes `scorer_results/candidate_december.png`.

To reproduce the validation numbers (~25 minutes — it fits the model five
times):

```bash
python evaluate.py && python report/screened_metrics.py
```

To reproduce the day-level specification choice, and the report:

```bash
python experiments/day_level_backtest.py
```

```bash
python report/make_figures.py && python report/make_report.py
```

## Layout

```
src/data.py                        loading and the data-quality repairs
src/features.py                    feature engineering
src/model.py                       the model: cross-sectional GBM + day-level term
train_predict.py                   end-to-end run, writes both submission files
evaluate.py                        forward-holdout and rolling-origin validation
experiments/day_level_backtest.py  how the day-level specification was chosen
report/                            figures, metrics and the written report
tests/test_pipeline.py             smoke and invariant checks (`pytest tests -q`)
docs/original_task_readme.md       the readme supplied with the assessment
data/                              the supplied CSVs, under the task's own names
```

`train_predict.py` reads `data/december_chart_inputs.csv` and writes the same
file back with `predicted_rate` filled, so re-running it is safe.

## Requirements

Python 3.11+, `lightgbm`, `scikit-learn`, `pandas`, `numpy`, `scipy`,
`matplotlib`; `reportlab` and `pypdf` only for rebuilding the report PDF.
`requirements.txt` pins them.
