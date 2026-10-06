"""E7: imbalance handling, as an ABLATION only.  python -m ml.src.e7_imbalance

Compares class weights, threshold tuning and three resamplers (random under-sampling, SMOTE, NearMiss) against no
handling, for logistic regression and gradient boosting, on Kaggle with all features. Hyper-parameters are the ones
E3 tuned; only the imbalance strategy changes. Resampling happens inside each training fold only (a pipeline step
that acts in fit and not in predict), so no validation or holdout row is ever resampled.

Why it is an ablation: resampling distorts probabilities (mean_predicted vs prevalence shows it), and the simulation
needs real probabilities, so it never uses a resampled model. The cut-off for "at the cross-validated threshold" is
chosen on the out-of-fold predictions of the TRAINING part (best F1), never on the holdout.

threshold_tuning is the unweighted model at that tuned threshold; its ranking metrics equal 'none' by construction.
Writes results/e7_imbalance.csv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from sklearn.metrics import precision_recall_curve

from . import experiment, features, train
from .data import DEFAULT_SEED, RESULTS_DIR, ROOT
from .e2_baselines import operating_point
from .evaluate import N_BOOT, HoldoutGuard
from .experiment import ResultWriter, ordered

SCRIPT = "ml/src/e7_imbalance.py"
BASES = ("logistic_regression", "hist_gradient_boosting")
FITTED_STRATEGIES = ("none", "class_weights", "random_under_sampling", "smote", "nearmiss")
ALL_STRATEGIES = ("none", "class_weights", "threshold_tuning", "random_under_sampling", "smote", "nearmiss")


def sampler_for(strategy: str, seed: int):
    return {"random_under_sampling": lambda: RandomUnderSampler(random_state=seed),
            "smote": lambda: SMOTE(random_state=seed),  # interpolates all columns; not SMOTENC (a documented limit)
            "nearmiss": lambda: NearMiss(version=1)}.get(strategy, lambda: None)()


def best_f1_threshold(y, probabilities) -> float:
    """The probability threshold with the best F1 on out-of-fold predictions."""
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def run(prepared=None, results_dir=RESULTS_DIR, n_boot: int = N_BOOT, seed: int = DEFAULT_SEED, grids=None,
        repo_root=ROOT) -> dict:
    prep = prepared or experiment.prepare_kaggle(seed=seed)
    selected = experiment.load_selected(results_dir)
    columns = features.all_columns("kaggle")
    guard = HoldoutGuard(n_boot, seed)
    X, y = prep.X_train[columns], prep.y_train
    rows = []

    for base in BASES:
        tuned = selected["all_models"][base]  # E3's cross-validated choice for this model
        base_rows = {}
        for strategy in FITTED_STRATEGIES:
            params = {**tuned, "class_weight": "balanced" if strategy == "class_weights" else None}

            def make(p=params, s=strategy):
                return train.build_pipeline(base, columns, p, seed, sampler_for(s, seed))

            oof = train.cross_val_predict_proba(make, X, y, prep.folds)  # training part only
            threshold = best_f1_threshold(y, oof)
            final = make().fit(X, y)
            probabilities = final.predict_proba(prep.X_holdout[columns])[:, 1]
            result = guard.score(f"{base}/{strategy}", prep.y_holdout, probabilities, prep.groups_holdout)
            default_point = operating_point(prep.y_holdout, probabilities, 0.5)
            tuned_point = operating_point(prep.y_holdout, probabilities, threshold)
            base_rows[strategy] = {
                "dataset": "kaggle", "base_model": base, "strategy": strategy, "params": json.dumps(params, sort_keys=True),
                "mean_predicted": float(probabilities.mean()), "prevalence": float(prep.y_holdout.mean()),
                "cv_threshold": threshold,
                "precision_at_default_threshold": default_point["precision"], "recall_at_default_threshold": default_point["recall"],
                "f1_at_default_threshold": default_point["f1"],
                "precision_at_cv_threshold": tuned_point["precision"], "recall_at_cv_threshold": tuned_point["recall"],
                "f1_at_cv_threshold": tuned_point["f1"], **result}
        base_rows["threshold_tuning"] = {**base_rows["none"], "strategy": "threshold_tuning"}  # same model, tuned cut-off
        rows += [base_rows[s] for s in ALL_STRATEGIES]

    table = ordered(pd.DataFrame(rows), ["dataset", "base_model", "strategy"])
    ResultWriter(results_dir, SCRIPT, seed, prep.source_files, repo_root).table(table, "e7_imbalance.csv")
    return {"table": table, "scored": guard.scored}


def main() -> None:
    result = run()
    show = ["base_model", "strategy", "auc_roc", "avg_precision", "brier", "mean_predicted", "f1_at_default_threshold", "f1_at_cv_threshold"]
    print(result["table"][show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
