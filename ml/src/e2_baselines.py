"""E2: the two baselines.  python -m ml.src.e2_baselines

  always_show               predict "attends" for everybody (probability 0). About 80% accurate and useless.
  earlier_no_show_rate_rule predict no-show when the patient's earlier no-show rate reaches a cut-off. The cut-off is
                            chosen on the TRAINING part (best F1), then scored once on the holdout. Kaggle only:
                            OpenML has no patient id, so it has no earlier-appointment history.

Writes results/e2_baselines.csv. Accuracy appears here only to show why it is not a headline metric.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import experiment
from .data import DEFAULT_SEED, RESULTS_DIR, ROOT
from .evaluate import N_BOOT, HoldoutGuard, accuracy_at, always_show_scores
from .experiment import ResultWriter, ordered

SCRIPT = "ml/src/e2_baselines.py"
CUTOFFS = [round(c, 2) for c in np.arange(0.1, 1.0001, 0.1)]


def operating_point(y, score, cutoff: float) -> dict:
    predicted = np.asarray(score) >= cutoff
    y = np.asarray(y)
    tp = int((predicted & (y == 1)).sum())
    precision = tp / predicted.sum() if predicted.sum() else 0.0
    recall = tp / y.sum() if y.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def best_cutoff(y, score) -> float:
    """The cut-off with the best F1 on the training part (first one on ties)."""
    return max(CUTOFFS, key=lambda c: operating_point(y, score, c)["f1"])


def run(kaggle=None, openml=None, results_dir=RESULTS_DIR, n_boot: int = N_BOOT, seed: int = DEFAULT_SEED,
        grids=None, repo_root=ROOT) -> dict:
    kaggle = kaggle or experiment.prepare_kaggle(seed=seed)
    openml = openml or experiment.prepare_openml(seed=seed)
    guard = HoldoutGuard(n_boot, seed)
    rows = []

    scores = always_show_scores(len(kaggle.y_holdout))
    rows.append({"dataset": "kaggle", "model": "always_show", **guard.score("kaggle/always_show", kaggle.y_holdout, scores,
                 kaggle.groups_holdout), "prevalence": kaggle.y_holdout.mean(),
                 "accuracy": accuracy_at(kaggle.y_holdout, scores, 0.5)})

    rate_train = kaggle.X_train["prev_no_show_rate"].fillna(0).to_numpy()
    cutoff = best_cutoff(kaggle.y_train, rate_train)
    rate_holdout = kaggle.X_holdout["prev_no_show_rate"].fillna(0).to_numpy()  # no history -> rate 0 -> predicted show
    point = operating_point(kaggle.y_holdout, rate_holdout, cutoff)
    rows.append({"dataset": "kaggle", "model": "earlier_no_show_rate_rule",
                 **guard.score("kaggle/earlier_no_show_rate_rule", kaggle.y_holdout, rate_holdout, kaggle.groups_holdout),
                 "prevalence": kaggle.y_holdout.mean(), "accuracy": accuracy_at(kaggle.y_holdout, rate_holdout, cutoff),
                 "cutoff": cutoff, "cutoff_chosen_on": "training part, best F1",
                 "precision_at_cutoff": point["precision"], "recall_at_cutoff": point["recall"], "f1_at_cutoff": point["f1"]})

    scores = always_show_scores(len(openml.y_holdout))
    rows.append({"dataset": "openml", "model": "always_show",
                 **guard.score("openml/always_show", openml.y_holdout, scores, None),
                 "prevalence": openml.y_holdout.mean(), "accuracy": accuracy_at(openml.y_holdout, scores, 0.5)})

    frame = ordered(pd.DataFrame(rows), ["dataset", "model"])
    writer = ResultWriter(results_dir, SCRIPT, seed, kaggle.source_files + openml.source_files, repo_root)
    writer.table(frame, "e2_baselines.csv")
    return {"table": frame, "scored": guard.scored}


def main() -> None:
    result = run()
    show = ["dataset", "model", "auc_roc", "avg_precision", "brier", "accuracy", "cutoff"]
    print(result["table"][show].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
