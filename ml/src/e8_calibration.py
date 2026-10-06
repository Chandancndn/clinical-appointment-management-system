"""E8: is the best model calibrated, and does calibration help?  python -m ml.src.e8_calibration

Takes the model E3 chose by cross-validation. Platt scaling (sigmoid) and isotonic regression are compared with the
uncalibrated model on out-of-fold predictions from PATIENT-GROUPED cross-validation of the training part (calibrators
are cross-fitted inside each training fold). The method with the lowest out-of-fold Brier score is chosen, then fitted
on the whole training part and scored ONCE on the holdout.

The holdout row also carries the uncalibrated model's Brier and expected calibration error for the before/after
comparison (the same single scoring event). Writes results/e8_calibration.csv, results/e8_reliability.csv and the
reliability figure results/figures/e8_reliability.png (equal-frequency bins, 95% Wilson intervals).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import calibrate, evaluate, experiment, train
from .data import DEFAULT_SEED, RESULTS_DIR, ROOT
from .evaluate import N_BOOT, HoldoutGuard
from .experiment import ResultWriter, ordered
from .plotting import VERSION_COLORS, line_figure

SCRIPT = "ml/src/e8_calibration.py"
BINS = 10


def reliability_figure(reliability: pd.DataFrame, method: str, writer: ResultWriter) -> None:
    series = []
    for version, label in (("uncalibrated", "Uncalibrated"), ("calibrated", f"Calibrated ({method})")):
        part = reliability[reliability["version"] == version]
        series.append({"label": label, "x": part["mean_predicted"].to_numpy(), "y": part["observed_rate"].to_numpy(),
                       "color": VERSION_COLORS[version], "markers": True,
                       "yerr": [(part["observed_rate"] - part["ci_low"]).to_numpy(), (part["ci_high"] - part["observed_rate"]).to_numpy()]})
    top = float(np.ceil(max(reliability["mean_predicted"].max(), reliability["ci_high"].max()) * 20) / 20)
    path = writer.figure_path("e8_reliability.png")
    line_figure(path, "Reliability on the Kaggle holdout",
                f"Predicted risk against observed no-show rate, {BINS} equal-frequency bins. On the diagonal = calibrated.",
                "Calibrator chosen by out-of-fold Brier on the training part. Whiskers: 95% Wilson interval.\n"
                "Numbers: results/e8_reliability.csv", series, "Mean predicted probability", "Observed no-show rate",
                {"x": [0, top], "y": [0, top], "label": "perfect calibration"}, limits=((0, top), (0, top)))
    writer.record_figure(path)


def run(prepared=None, results_dir=RESULTS_DIR, n_boot: int = N_BOOT, seed: int = DEFAULT_SEED, grids=None,
        repo_root=ROOT) -> dict:
    prep = prepared or experiment.prepare_kaggle(seed=seed)
    selected = experiment.load_selected(results_dir)
    model, params, columns = selected["model"], selected["params"], selected["columns"]

    def make():
        return train.build_pipeline(model, columns, params, seed)

    X, y, groups = prep.X_train[columns], prep.y_train, prep.groups_train
    cv_rows, briers = [], {}
    for method in calibrate.METHODS:
        oof = calibrate.oof_predictions(make, X, y, groups, prep.folds, method, seed=seed)
        auc = roc_auc_score(y, oof)
        evaluate.check_auc_plausible(auc, f"out-of-fold predictions ({method})")
        briers[method] = brier_score_loss(y, oof)
        cv_rows.append({"stage": "grouped_cv", "method": method, "model": model, "brier_cv": briers[method],
                        "ece_cv": calibrate.expected_calibration_error(y, oof, BINS), "auc_cv": auc,
                        "ap_cv": average_precision_score(y, oof), "mean_predicted_cv": float(oof.mean())})
    chosen = calibrate.choose_method(briers)  # by cross-validated Brier, never by the holdout

    raw = make().fit(X, y).predict_proba(prep.X_holdout[columns])[:, 1]
    if chosen == "uncalibrated":
        calibrated = raw
    else:
        calibrated = calibrate.fit_calibrated(make, X, y, groups, chosen, seed=seed).predict_proba(prep.X_holdout[columns])[:, 1]
    guard = HoldoutGuard(n_boot, seed)
    result = guard.score(f"{model}/{chosen}", prep.y_holdout, calibrated, prep.groups_holdout)  # the one holdout look
    holdout_row = {"stage": "holdout", "method": chosen, "model": model, **result,
                   "ece": calibrate.expected_calibration_error(prep.y_holdout, calibrated, BINS),
                   "brier_uncalibrated": brier_score_loss(prep.y_holdout, raw),
                   "ece_uncalibrated": calibrate.expected_calibration_error(prep.y_holdout, raw, BINS),
                   "mean_predicted": float(calibrated.mean()), "prevalence": float(prep.y_holdout.mean())}
    table = pd.concat([pd.DataFrame(cv_rows), pd.DataFrame([holdout_row])], ignore_index=True)
    table = ordered(table, ["stage", "method", "model"])

    reliability = pd.concat([
        calibrate.reliability_table(prep.y_holdout, raw, BINS, "quantile").assign(version="uncalibrated"),
        calibrate.reliability_table(prep.y_holdout, calibrated, BINS, "quantile").assign(version="calibrated"),
    ], ignore_index=True)
    reliability = reliability[["version", "bin", "n", "mean_predicted", "observed_rate", "ci_low", "ci_high"]]

    writer = ResultWriter(results_dir, SCRIPT, seed, prep.source_files, repo_root)
    writer.table(table, "e8_calibration.csv")
    writer.table(reliability, "e8_reliability.csv")
    reliability_figure(reliability, chosen, writer)
    return {"table": table, "reliability": reliability, "chosen": chosen, "scored": guard.scored}


def main() -> None:
    result = run()
    cols = ["stage", "method", "brier_cv", "ece_cv", "auc_cv", "brier", "brier_lo", "brier_hi", "ece", "brier_uncalibrated", "ece_uncalibrated"]
    print(result["table"][[c for c in cols if c in result["table"]]].round(4).to_string(index=False))
    print("chosen:", result["chosen"])


if __name__ == "__main__":
    main()
