"""Calibration: reliability tables, expected calibration error, and out-of-fold calibrated predictions.

Platt scaling ("sigmoid") and isotonic regression are compared on the TRAINING part with patient-grouped
cross-validation; the method with the lowest out-of-fold Brier score is the one used (PLAN section 4, step 6).
Within each outer training fold the calibrator itself is fitted by an inner patient-grouped cross-fit, so no
calibrator ever sees a validation row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from . import train
from .data import DEFAULT_SEED, grouped_folds
from .e1_profile import wilson_interval

METHODS = ("uncalibrated", "sigmoid", "isotonic")  # simplest first: ties keep the earlier method


def choose_method(briers: dict) -> str:
    """The method with the lowest Brier score; on a tie the simpler (earlier in METHODS) one."""
    best = None
    for method in METHODS:
        if method in briers and (best is None or briers[method] < briers[best]):
            best = method
    return best


def _bin_index(p: np.ndarray, bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.minimum((p * bins).astype(int), bins - 1)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    return np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)


def reliability_table(y, p, bins: int = 10, strategy: str = "uniform") -> pd.DataFrame:
    """One row per non-empty bin: n, mean predicted probability, observed no-show rate and its Wilson interval."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    index = _bin_index(p, bins, strategy)
    rows = []
    for b in np.unique(index):
        mask = index == b
        low, high = wilson_interval(int(y[mask].sum()), int(mask.sum()))
        rows.append({"bin": int(len(rows) + 1), "n": int(mask.sum()), "mean_predicted": float(p[mask].mean()),
                     "observed_rate": float(y[mask].mean()), "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Sum over equal-width probability bins of (share of rows) * |observed rate - mean predicted|."""
    table = reliability_table(y, p, bins, "uniform")
    return float((table["n"] / table["n"].sum() * (table["observed_rate"] - table["mean_predicted"]).abs()).sum())


def oof_predictions(make_base, X, y, groups, folds, method: str, inner_splits: int = 3,
                    seed: int = DEFAULT_SEED) -> np.ndarray:
    """Out-of-fold probabilities for `method`. Calibrators are cross-fitted inside each outer training fold."""
    if method == "uncalibrated":
        return train.cross_val_predict_proba(make_base, X, y, folds)
    y, groups = np.asarray(y), np.asarray(groups)
    predictions = np.full(len(y), np.nan)
    for train_positions, valid_positions in folds:
        inner = grouped_folds(groups[train_positions], n_splits=inner_splits, seed=seed)
        calibrated = CalibratedClassifierCV(make_base(), method=method, cv=inner)
        calibrated.fit(train._rows(X, train_positions), y[train_positions])
        predictions[valid_positions] = calibrated.predict_proba(train._rows(X, valid_positions))[:, 1]
    return predictions


def fit_calibrated(make_base, X, y, groups, method: str, inner_splits: int = 3, seed: int = DEFAULT_SEED):
    """The final calibrated model on all training rows (inner patient-grouped cross-fit)."""
    inner = grouped_folds(np.asarray(groups), n_splits=inner_splits, seed=seed)
    return CalibratedClassifierCV(make_base(), method=method, cv=inner).fit(X, np.asarray(y))
