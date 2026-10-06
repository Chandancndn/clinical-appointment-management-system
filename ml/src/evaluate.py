"""Metrics, patient-level bootstrap intervals, paired comparisons and the score-the-holdout-once guard.

Headline metrics (CLAUDE.md): AUC-ROC, average precision, sensitivity at fixed precision (30%, 40%, 50%) and the
Brier score. Accuracy is never a headline (always-show scores about 80%), only shown for the baseline.

Uncertainty: 1,000 bootstrap resamples of the holdout that resample PATIENTS, not rows, because one patient's
appointments are correlated. A resample is represented by a weight per row (a patient drawn k times gives all of
that patient's rows weight k), which gives exactly the metrics of the resampled data without copying it.
OpenML has no patient id, so there rows are resampled directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score

N_BOOT = 1000
DEFAULT_SEED = 42
PRECISION_TARGETS = (0.30, 0.40, 0.50)
METRIC_NAMES = ["auc_roc", "avg_precision", "brier", "sens_at_prec_30", "sens_at_prec_40", "sens_at_prec_50"]
AUC_LIMIT = 0.85  # above this, stop and look for leakage (PLAN section 4: expect roughly 0.65 to 0.80)


class LeakageSuspected(RuntimeError):
    """An AUC above the plausibility limit: investigate leakage before reporting anything."""


class HoldoutAlreadyScored(RuntimeError):
    """A final model was scored on the locked holdout a second time."""


def check_auc_plausible(auc: float, where: str, limit: float = AUC_LIMIT) -> None:
    if auc is not None and not np.isnan(auc) and auc > limit:
        raise LeakageSuspected(f"AUC {auc:.3f} in {where} is above {limit}: stop and look for leakage "
                               "before going on (expected roughly 0.65 to 0.80)")


# ---- point metrics ----------------------------------------------------------------------------------
def always_show_scores(n: int) -> np.ndarray:
    """The always-show baseline: probability 0 of a no-show for everyone."""
    return np.zeros(n)


def _recall_at_precision(precision: np.ndarray, recall: np.ndarray, target: float) -> float:
    reachable = precision >= target - 1e-12
    return float(recall[reachable].max()) if reachable.any() else 0.0


def sensitivity_at_precision(y, score, target: float, sample_weight=None) -> float:
    """Highest recall among thresholds whose precision is at least `target`; 0 if no threshold reaches it."""
    precision, recall, _ = precision_recall_curve(y, score, sample_weight=sample_weight)
    return _recall_at_precision(precision, recall, target)


def point_metrics(y, score, sample_weight=None) -> dict:
    """The six headline metrics on (optionally weighted) data."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    if len(np.unique(y)) < 2:
        return {name: np.nan for name in METRIC_NAMES}
    precision, recall, _ = precision_recall_curve(y, score, sample_weight=sample_weight)
    result = {
        "auc_roc": roc_auc_score(y, score, sample_weight=sample_weight),
        "avg_precision": average_precision_score(y, score, sample_weight=sample_weight),
        "brier": brier_score_loss(y, score, sample_weight=sample_weight),
    }
    for target in PRECISION_TARGETS:
        result[f"sens_at_prec_{round(target * 100)}"] = _recall_at_precision(precision, recall, target)
    return result


def accuracy_at(y, score, threshold: float = 0.5) -> float:
    return float(np.mean((np.asarray(score) >= threshold).astype(int) == np.asarray(y)))


# ---- bootstrap --------------------------------------------------------------------------------------
def _weight_stream(groups, n_boot: int, seed: int, n_rows: int | None):
    """Yield one weight vector per bootstrap replicate: patients (or rows) drawn with replacement."""
    rng = np.random.default_rng(seed)
    if groups is None:
        for _ in range(n_boot):
            yield rng.multinomial(n_rows, np.full(n_rows, 1.0 / n_rows)).astype("int32")
        return
    codes, uniques = pd.factorize(np.asarray(groups))
    patients = len(uniques)
    for _ in range(n_boot):
        drawn = np.bincount(rng.integers(0, patients, patients), minlength=patients)
        yield drawn[codes].astype("int32")


def bootstrap_weights(groups, n_boot: int = N_BOOT, seed: int = DEFAULT_SEED, n_rows: int | None = None) -> np.ndarray:
    """(n_boot, n_rows) integer weights; all of a patient's rows share one weight."""
    return np.vstack(list(_weight_stream(groups, n_boot, seed, n_rows if groups is None else None)))


def _interval(values) -> tuple[float, float]:
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return float(low), float(high)


def metrics_with_ci(y, score, groups=None, n_boot: int = N_BOOT, seed: int = DEFAULT_SEED) -> dict:
    """Point metrics plus a 95% percentile interval from `n_boot` patient-level bootstrap resamples."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    point = point_metrics(y, score)
    draws = {name: [] for name in METRIC_NAMES}
    for weights in _weight_stream(groups, n_boot, seed, len(y)):
        keep = weights > 0
        replicate = point_metrics(y[keep], score[keep], sample_weight=weights[keep])
        for name in METRIC_NAMES:
            draws[name].append(replicate[name])
    result = {"n": int(len(y)), "n_positive": int(y.sum()), "n_boot": int(n_boot)}
    for name in METRIC_NAMES:
        result[name] = float(point[name])
        result[f"{name}_lo"], result[f"{name}_hi"] = _interval(draws[name])
    return result


def paired_differences(y, score_a, score_b, groups, metrics, n_boot: int = N_BOOT, seed: int = DEFAULT_SEED) -> dict:
    """metric(a) - metric(b) with a 95% interval, using the SAME resamples for both scores. {metric: (diff, lo, hi)}."""
    y, score_a, score_b = np.asarray(y), np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
    point_a, point_b = point_metrics(y, score_a), point_metrics(y, score_b)
    diffs = {m: [] for m in metrics}
    for weights in _weight_stream(groups, n_boot, seed, len(y)):
        keep = weights > 0
        a = point_metrics(y[keep], score_a[keep], sample_weight=weights[keep])
        b = point_metrics(y[keep], score_b[keep], sample_weight=weights[keep])
        for m in metrics:
            diffs[m].append(a[m] - b[m])
    result = {}
    for m in metrics:
        low, high = _interval(diffs[m])
        result[m] = (float(point_a[m] - point_b[m]), low, high)
    return result


def paired_difference(y, score_a, score_b, groups, metric: str, n_boot: int = N_BOOT, seed: int = DEFAULT_SEED):
    return paired_differences(y, score_a, score_b, groups, [metric], n_boot, seed)[metric]


# ---- the holdout guard ------------------------------------------------------------------------------
class HoldoutGuard:
    """Scores final models on the locked holdout, refusing a second look at the same model."""

    def __init__(self, n_boot: int = N_BOOT, seed: int = DEFAULT_SEED) -> None:
        self.n_boot, self.seed = n_boot, seed
        self.scored: list[str] = []

    def score(self, key: str, y, score, groups=None) -> dict:
        if key in self.scored:
            raise HoldoutAlreadyScored(f"'{key}' was already scored on the holdout; a final model is scored once")
        point = point_metrics(y, score)
        check_auc_plausible(point["auc_roc"], f"holdout score of {key}")
        self.scored.append(key)
        return metrics_with_ci(y, score, groups, self.n_boot, self.seed)
