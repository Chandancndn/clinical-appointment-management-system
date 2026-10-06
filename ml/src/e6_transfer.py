"""E6: how much AUC is lost when training on one dataset and testing on the other?  python -m ml.src.e6_transfer

Only age, sex, lead time (and the same-day flag) and appointment weekday exist, with the same meaning, in both
datasets, so both directions use exactly those features, for logistic regression and gradient boosting.

  transfer       trained on the SOURCE training part, scored on the TARGET holdout
  within-target  trained on the TARGET training part with the same features, scored on the same target holdout
  auc_drop       within-target minus transfer, with a paired bootstrap 95% interval (same resamples for both)

Training uses only each dataset's training part, so no holdout is used for fitting and every final model is scored
once. Writes results/e6_transfer.csv.
"""
from __future__ import annotations

import json

import pandas as pd

from . import experiment, features
from .data import DEFAULT_SEED, RESULTS_DIR, ROOT
from .evaluate import N_BOOT, HoldoutGuard, paired_differences
from .experiment import ResultWriter

SCRIPT = "ml/src/e6_transfer.py"
MODELS = ("logistic_regression", "hist_gradient_boosting")
DIRECTIONS = (("kaggle", "openml"), ("openml", "kaggle"))


def run(kaggle=None, openml=None, results_dir=RESULTS_DIR, n_boot: int = N_BOOT, seed: int = DEFAULT_SEED, grids=None,
        repo_root=ROOT) -> dict:
    sets = {"kaggle": kaggle or experiment.prepare_kaggle(seed=seed), "openml": openml or experiment.prepare_openml(seed=seed)}
    columns = features.COMMON_COLUMNS
    guard = HoldoutGuard(n_boot, seed)

    fitted, tuned_params = {}, {}
    for name, prep in sets.items():  # each (dataset, model) is tuned by cross-validation on its own training part
        for model in MODELS:
            tuned, fitted[(name, model)] = experiment.tune_and_fit(model, prep, columns, grids, seed)
            tuned_params[(name, model)] = tuned.best_params

    rows = []
    for source, target in DIRECTIONS:
        t = sets[target]
        for model in MODELS:
            direction = f"{source}_to_{target}"
            within_scores = experiment.holdout_probabilities(fitted[(target, model)], t, columns)
            transfer_scores = experiment.holdout_probabilities(fitted[(source, model)], t, columns)
            within = guard.score(f"{direction}/{model}/within_target", t.y_holdout, within_scores, t.groups_holdout)
            transfer = guard.score(f"{direction}/{model}/transfer", t.y_holdout, transfer_scores, t.groups_holdout)
            drops = paired_differences(t.y_holdout, within_scores, transfer_scores, t.groups_holdout,
                                       ["auc_roc", "avg_precision"], n_boot, seed)
            row = {"direction": direction, "source": source, "target": target, "model": model,
                   "features": ",".join(columns), "n_train_source": len(sets[source].y_train),
                   "n_holdout_target": int(within["n"]), "prevalence_target_holdout": float(t.y_holdout.mean()),
                   "prevalence_source_train": float(sets[source].y_train.mean()),
                   "params_source": json.dumps(tuned_params[(source, model)], sort_keys=True),
                   "params_within": json.dumps(tuned_params[(target, model)], sort_keys=True)}
            for label, result in (("within_target", within), ("transfer", transfer)):
                suffix = "within_target" if label == "within_target" else "transfer"
                row.update({f"auc_{suffix}": result["auc_roc"], f"auc_{suffix}_lo": result["auc_roc_lo"],
                            f"auc_{suffix}_hi": result["auc_roc_hi"], f"ap_{suffix}": result["avg_precision"],
                            f"brier_{suffix}": result["brier"]})
            for metric, short in (("auc_roc", "auc"), ("avg_precision", "ap")):
                diff, low, high = drops[metric]
                row.update({f"{short}_drop": diff, f"{short}_drop_lo": low, f"{short}_drop_hi": high})
            rows.append(row)

    table = pd.DataFrame(rows)
    inputs = sets["kaggle"].source_files + sets["openml"].source_files
    ResultWriter(results_dir, SCRIPT, seed, inputs, repo_root).table(table, "e6_transfer.csv")
    return {"table": table, "scored": guard.scored}


def main() -> None:
    result = run()
    show = ["direction", "model", "auc_within_target", "auc_transfer", "auc_drop", "auc_drop_lo", "auc_drop_hi"]
    print(result["table"][show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
