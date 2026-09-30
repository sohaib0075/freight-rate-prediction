"""Holdout metrics split by label quality.

About 1.3% of ``posted_rate`` values are corrupted by a large multiplicative
factor.  Those rows are unpredictable by construction, and on a 9,500-row
holdout they dominate MAE, RMSE and R2 while saying nothing about the model.

This script scores the Sep-Oct forward holdout twice: on every row, and on the
rows whose label passes a corruption screen.  The screen is built *only* from
the holdout's own loads and labels - an out-of-fold, date-blind model fitted
inside the holdout, exactly the procedure the trainer uses on training data -
so it never consults the model being scored, and cannot flatter it by throwing
away rows the model happens to miss.

    python report/screened_metrics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D                                  # noqa: E402
from src import features as F                              # noqa: E402
from src.model import FreightRateModel                     # noqa: E402
from evaluate import metrics                               # noqa: E402


def main() -> None:
    train_raw, valid_raw = D.prepare(Path("data"))
    ref = F.build_reference([train_raw, valid_raw])
    dev = F.transform(train_raw, ref)

    cut = pd.Timestamp("2025-09-01")
    train, test = dev[dev.date < cut], dev[dev.date >= cut].reset_index(drop=True)

    model = FreightRateModel().fit(train)
    pred = model.predict(test)
    truth = test["posted_rate"].to_numpy()

    # label screen built inside the holdout, independent of `model`
    screen = FreightRateModel()._flag_inliers(test, FreightRateModel.target(test))

    out = {
        "all_rows": metrics(truth, pred),
        "labels_passing_the_screen": metrics(truth[screen], pred[screen]),
        "screen_rejected_%": 100 * (1 - float(screen.mean())),
        "share_of_rows_with_abs_log_error_over_0.2_%":
            100 * float(np.mean(np.abs(np.log(pred) - np.log(truth)) > 0.2)),
    }
    print(json.dumps(out, indent=2))
    Path("report/screened_metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
