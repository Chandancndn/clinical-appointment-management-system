"""E5: the same ladder on OpenML.  python -m ml.src.e5_openml_models

Always-show, then logistic regression (balanced class weights), random forest and gradient boosting on every OpenML
feature (lead time, weekday, hour, booking weekday and hour, age, sex, channel, appointment type, specialty).
OpenML has no patient id, so there is no history feature and no rule baseline; tuning uses STRATIFIED cross-validation
on months 1-3; the holdout is the last month, scored once per final model; intervals resample rows (no patients).

Writes results/e5_openml_models.csv and results/e5_tuning.csv.
"""
from __future__ import annotations

import json

import pandas as pd

from . import experiment, features, train
from .data import DEFAULT_SEED, RESULTS_DIR, ROOT
from .evaluate import N_BOOT, HoldoutGuard, accuracy_at, always_show_scores
from .experiment import ResultWriter, ordered

SCRIPT = "ml/src/e5_openml_models.py"


def run(prepared=None, results_dir=RESULTS_DIR, n_boot: int = N_BOOT, seed: int = DEFAULT_SEED, grids=None,
        repo_root=ROOT) -> dict:
    prep = prepared or experiment.prepare_openml(seed=seed)
    columns = features.all_columns("openml")
    guard = HoldoutGuard(n_boot, seed)
    zeros = always_show_scores(len(prep.y_holdout))
    rows = [{"dataset": "openml", "model": "always_show", **guard.score("always_show", prep.y_holdout, zeros, None),
             "prevalence": prep.y_holdout.mean(), "accuracy": accuracy_at(prep.y_holdout, zeros, 0.5)}]
    tuning = []
    for model in train.MODEL_NAMES:
        tuned, fitted = experiment.tune_and_fit(model, prep, columns, grids, seed)
        selected = tuned.results[tuned.results["selected"]].iloc[0]
        scores = experiment.holdout_probabilities(fitted, prep, columns)
        rows.append({"dataset": "openml", "model": model, "params": json.dumps(tuned.best_params, sort_keys=True),
                     "cv_ap_mean": selected["cv_ap_mean"], "cv_auc_mean": selected["cv_auc_mean"],
                     **guard.score(model, prep.y_holdout, scores, None), "prevalence": prep.y_holdout.mean()})
        tuning.append(tuned.results)

    table = ordered(pd.DataFrame(rows), ["dataset", "model", "params", "cv_ap_mean", "cv_auc_mean"])
    writer = ResultWriter(results_dir, SCRIPT, seed, prep.source_files, repo_root)
    writer.table(table, "e5_openml_models.csv")
    writer.table(pd.concat(tuning, ignore_index=True), "e5_tuning.csv")
    return {"table": table, "scored": guard.scored}


def main() -> None:
    result = run()
    show = ["model", "cv_ap_mean", "cv_auc_mean", "auc_roc", "avg_precision", "brier", "sens_at_prec_40"]
    print(result["table"][show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
