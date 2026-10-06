"""E3: model comparison on Kaggle with all features.  python -m ml.src.e3_kaggle_models

Logistic regression (balanced class weights), random forest and HistGradientBoostingClassifier, each tuned over a
small grid with PATIENT-GROUPED cross-validation on the training part only, by cross-validated average precision.
Each final model is scored ONCE on the locked holdout (AUC-ROC, average precision, sensitivity at 30/40/50%
precision, Brier; 95% intervals from 1,000 patient-level bootstrap resamples).

Also written: a shuffled-label control (must score about 0.5), the tuning grid, which model won, the ladder check
("every step must beat the one before, or the report says it did not") and ROC / precision-recall figures.

Stops with LeakageSuspected, writing nothing, if any validation or holdout AUC is above 0.85.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from . import experiment, features, train
from .data import DEFAULT_SEED, RESULTS_DIR, ROOT
from .evaluate import N_BOOT, HoldoutGuard, paired_differences
from .experiment import ResultWriter, ordered
from .plotting import MODEL_COLORS, MODEL_LABELS, line_figure

SCRIPT = "ml/src/e3_kaggle_models.py"
CHOSEN_BY = "cross-validated average precision on the training part"
LADDER_METRICS = ["auc_roc", "avg_precision", "sens_at_prec_40"]


def ladder_table(prep, scores: dict, n_boot: int, seed: int) -> pd.DataFrame:
    """Does each step beat the one before? Paired bootstrap: both models on the same resampled patients."""
    rule = prep.X_holdout["prev_no_show_rate"].fillna(0).to_numpy()
    chain = [("earlier_no_show_rate_rule", rule), ("logistic_regression", scores["logistic_regression"]),
             ("random_forest", scores["random_forest"]), ("hist_gradient_boosting", scores["hist_gradient_boosting"])]
    rows = []
    for (previous, previous_scores), (current, current_scores) in zip(chain, chain[1:]):
        diffs = paired_differences(prep.y_holdout, current_scores, previous_scores, prep.groups_holdout, LADDER_METRICS,
                                   n_boot, seed)
        row = {"step": f"{previous} -> {current}"}
        for metric, (diff, low, high) in diffs.items():
            row.update({f"delta_{metric}": diff, f"delta_{metric}_lo": low, f"delta_{metric}_hi": high})
        row["beats_previous_step"] = bool(row["delta_auc_roc_lo"] > 0 and row["delta_avg_precision_lo"] > 0)
        row["beats_by_point_estimate"] = bool(row["delta_auc_roc"] > 0 and row["delta_avg_precision"] > 0)
        row["note"] = ("difference = next minus previous; 'beats' means both the AUC and average-precision intervals "
                       "are above zero. Brier is not compared here: class-weighted probabilities are not calibrated (E8)")
        rows.append(row)
    return pd.DataFrame(rows)


def draw_figures(prep, scores: dict, writer: ResultWriter) -> None:
    y = prep.y_holdout
    roc, pr = [], []
    for model, score in scores.items():
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        color = MODEL_COLORS[model]
        roc.append({"label": MODEL_LABELS[model], "x": fpr, "y": tpr, "color": color})
        pr.append({"label": MODEL_LABELS[model], "x": recall, "y": precision, "color": color})
    note = "Locked holdout (latest 20% of appointment dates), each model scored once. Numbers: results/e3_kaggle_models.csv"
    path = writer.figure_path("e3_roc.png")
    line_figure(path, "ROC curves on the Kaggle holdout", "All features. Higher and further left is better.", note, roc,
                "False positive rate", "True positive rate (sensitivity)", {"x": [0, 1], "y": [0, 1], "label": "chance"})
    writer.record_figure(path)
    path = writer.figure_path("e3_pr.png")
    line_figure(path, "Precision-recall curves on the Kaggle holdout", "All features. The grey line is the no-show rate "
                "(what always-show achieves).", note, pr, "Recall (sensitivity)", "Precision", {
                    "x": [0, 1], "y": [y.mean(), y.mean()], "label": "no-show rate"}, limits=((0, 1), (0, 1)))
    writer.record_figure(path)


def run(prepared=None, results_dir=RESULTS_DIR, n_boot: int = N_BOOT, seed: int = DEFAULT_SEED, grids=None,
        repo_root=ROOT) -> dict:
    prep = prepared or experiment.prepare_kaggle(seed=seed)
    columns = features.all_columns("kaggle")
    guard = HoldoutGuard(n_boot, seed)
    rows, tuning, scores, best_params = [], [], {}, {}

    for model in train.MODEL_NAMES:
        tuned, fitted = experiment.tune_and_fit(model, prep, columns, grids, seed)
        selected = tuned.results[tuned.results["selected"]].iloc[0]
        scores[model] = experiment.holdout_probabilities(fitted, prep, columns)
        result = guard.score(model, prep.y_holdout, scores[model], prep.groups_holdout)  # the one holdout look
        rows.append({"dataset": "kaggle", "model": model, "params": json.dumps(tuned.best_params, sort_keys=True),
                     "cv_ap_mean": selected["cv_ap_mean"], "cv_auc_mean": selected["cv_auc_mean"], **result})
        tuning.append(tuned.results)
        best_params[model] = tuned.best_params

    # control: the best-performing recipe trained on shuffled labels must be a coin flip
    shuffled = np.random.default_rng(seed).permutation(prep.y_train)
    control = train.fit_final("hist_gradient_boosting", prep.X_train, shuffled, columns, best_params["hist_gradient_boosting"], seed)
    result = guard.score("control_shuffled_labels", prep.y_holdout, experiment.holdout_probabilities(control, prep, columns),
                         prep.groups_holdout)
    rows.append({"dataset": "kaggle", "model": "control_shuffled_labels",
                 "params": json.dumps(best_params["hist_gradient_boosting"], sort_keys=True), **result})

    table = ordered(pd.DataFrame(rows), ["dataset", "model", "params", "cv_ap_mean", "cv_auc_mean"])
    tuning = pd.concat(tuning, ignore_index=True)
    ladder = ladder_table(prep, scores, n_boot, seed)
    models = table[table["model"].isin(train.MODEL_NAMES)]
    winner = models.loc[models["cv_ap_mean"].idxmax(), "model"]  # chosen by cross-validation, never by the holdout
    selected_info = {"model": winner, "params": best_params[winner], "chosen_by": CHOSEN_BY,
                     "cv_ap_mean": float(models.loc[models["model"] == winner, "cv_ap_mean"].iloc[0]),
                     "all_models": best_params, "columns": columns, "seed": seed}

    writer = ResultWriter(results_dir, SCRIPT, seed, prep.source_files, repo_root)
    writer.table(table, "e3_kaggle_models.csv")
    writer.table(tuning, "e3_tuning.csv")
    writer.table(ladder, "e3_ladder.csv")
    writer.json(selected_info, "e3_selected.json")
    draw_figures(prep, scores, writer)
    return {"table": table, "tuning": tuning, "ladder": ladder, "selected": selected_info, "scored": guard.scored}


def main() -> None:
    result = run()
    show = ["model", "cv_ap_mean", "auc_roc", "avg_precision", "brier", "sens_at_prec_40"]
    print(result["table"][show].round(4).to_string(index=False))
    print("\nladder:"); print(result["ladder"][["step", "delta_auc_roc", "delta_avg_precision", "beats_previous_step"]].round(4).to_string(index=False))
    print("\nselected by cross-validation:", result["selected"]["model"], result["selected"]["params"])


if __name__ == "__main__":
    main()
