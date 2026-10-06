"""Model pipelines, fold-by-fold prediction and tuning (patient-grouped CV on the training part only).

Everything that learns from data (imputers, scalers, one-hot and target encoders, resamplers, the model itself) is a
step of ONE sklearn pipeline. cross_val_predict_proba() builds a fresh pipeline for every fold and fits it on that
fold's training rows only, so nothing is ever fitted on a validation row (leakage test 4).
"""
from __future__ import annotations

import json
from typing import Callable, Optional

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from . import evaluate, features
from .data import DEFAULT_SEED

MODEL_NAMES = ("logistic_regression", "random_forest", "hist_gradient_boosting")

# Small grids, tuned by cross-validated average precision on the training part (PLAN section 4).
GRIDS = {
    "logistic_regression": {"C": [0.01, 0.1, 1.0, 10.0]},
    "random_forest": {"n_estimators": [300], "min_samples_leaf": [20, 80], "max_features": ["sqrt", 0.5]},
    "hist_gradient_boosting": {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31], "max_iter": [200],
                               "l2_regularization": [0.0, 1.0]},
}
DEFAULT_PARAMS = {
    "logistic_regression": {"C": 1.0, "class_weight": "balanced"},  # balanced class weights, as the plan says
    "random_forest": {"n_estimators": 300, "min_samples_leaf": 20, "max_features": "sqrt", "class_weight": None},
    "hist_gradient_boosting": {"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 200,
                               "l2_regularization": 0.0, "class_weight": None},
}


def _target_encoder(seed: int) -> TargetEncoder:
    """Target encoder with internal 5-fold cross-fitting (shuffled, seeded). It is a pipeline step, so it is fitted
    on the training fold only. scikit-learn 1.9 moved shuffling into the splitter passed as `cv`."""
    if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 9):
        return TargetEncoder(target_type="binary", cv=KFold(5, shuffle=True, random_state=seed))
    return TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=seed)


def _preprocessor(model: str, columns, seed: int, impute_nan: bool = False) -> ColumnTransformer:
    spec = features.feature_spec(columns)
    parts = []
    if model == "logistic_regression":
        if spec.numeric:  # NaN only occurs for "no earlier appointments" in prev_no_show_rate: 0 together with has_history
            parts.append(("numeric", Pipeline([("fill", SimpleImputer(strategy="constant", fill_value=0)),
                                               ("scale", StandardScaler())]), spec.numeric))
        if spec.binary:
            parts.append(("binary", "passthrough", spec.binary))
        if spec.category_small:
            parts.append(("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), spec.category_small))
        if spec.target_encoded:
            parts.append(("target", Pipeline([("encode", _target_encoder(seed)), ("scale", StandardScaler())]),
                          spec.target_encoded))
    else:
        if spec.numeric:
            # the histogram booster handles NaN natively, unless a resampler sits after the preprocessing (E7)
            fill = (SimpleImputer(strategy="constant", fill_value=-1) if model == "random_forest" or impute_nan
                    else "passthrough")
            parts.append(("numeric", fill, spec.numeric))
        if spec.binary + spec.category_small:
            parts.append(("plain", "passthrough", spec.binary + spec.category_small))
        if spec.target_encoded:
            parts.append(("target", _target_encoder(seed), spec.target_encoded))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


def _estimator(model: str, params: dict, seed: int):
    merged = {**DEFAULT_PARAMS[model], **params}
    if model == "logistic_regression":
        return LogisticRegression(max_iter=2000, solver="lbfgs", **merged)
    if model == "random_forest":
        return RandomForestClassifier(n_jobs=-1, random_state=seed, **merged)
    if model == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **merged)
    raise KeyError(f"unknown model '{model}'")


def build_pipeline(model: str, columns, params: Optional[dict] = None, seed: int = DEFAULT_SEED, sampler=None):
    """An unfitted pipeline: preprocessing -> [resampler] -> model. `sampler` (imbalanced-learn) is for E7 only."""
    prep = _preprocessor(model, columns, seed, impute_nan=sampler is not None)
    estimator = _estimator(model, params or {}, seed)
    if sampler is None:
        return Pipeline([("prep", prep), ("model", estimator)])
    from imblearn.pipeline import Pipeline as SamplingPipeline  # resampling happens in fit only, never in predict

    return SamplingPipeline([("prep", prep), ("sampler", sampler), ("model", estimator)])


def fit_final(model: str, X, y, columns, params: Optional[dict] = None, seed: int = DEFAULT_SEED, sampler=None):
    """Fit a pipeline on all of the given rows (the whole training part)."""
    return build_pipeline(model, columns, params, seed, sampler).fit(X[list(columns)], np.asarray(y))


# ---- cross-validation -------------------------------------------------------------------------------
def _rows(X, positions):
    return X.iloc[positions] if hasattr(X, "iloc") else X[positions]


def cross_val_predict_proba(make_estimator: Callable, X, y, folds) -> np.ndarray:
    """Out-of-fold P(no-show). Each fold gets a FRESH estimator fitted on that fold's training rows only."""
    y = np.asarray(y)
    predictions = np.full(len(y), np.nan)
    for train_positions, valid_positions in folds:
        estimator = make_estimator()
        estimator.fit(_rows(X, train_positions), y[train_positions])
        predictions[valid_positions] = estimator.predict_proba(_rows(X, valid_positions))[:, 1]
    return predictions


def fold_scores(y, oof, folds) -> pd.DataFrame:
    """Per-fold AUC, average precision and Brier from out-of-fold predictions."""
    y = np.asarray(y)
    rows = []
    for _, valid in folds:
        rows.append({"auc": roc_auc_score(y[valid], oof[valid]), "ap": average_precision_score(y[valid], oof[valid]),
                     "brier": brier_score_loss(y[valid], oof[valid])})
    return pd.DataFrame(rows)


class TuneResult:
    def __init__(self, best_params: dict, results: pd.DataFrame) -> None:
        self.best_params, self.results = best_params, results


def tune(model: str, X, y, folds, columns, grid: Optional[dict] = None, seed: int = DEFAULT_SEED,
         metric: str = "ap") -> TuneResult:
    """Pick hyper-parameters by cross-validated average precision on the TRAINING part. The holdout is not involved.

    Stops (LeakageSuspected) if any validation fold scores an AUC above the plausibility limit.
    """
    grid = GRIDS[model] if grid is None else grid
    Xc = X[list(columns)]
    records = []
    for params in ParameterGrid(grid):
        oof = cross_val_predict_proba(lambda p=params: build_pipeline(model, columns, p, seed), Xc, y, folds)
        scores = fold_scores(y, oof, folds)
        for auc in scores["auc"]:
            evaluate.check_auc_plausible(auc, f"a validation fold of {model} {params}")
        records.append({"model": model, "params": json.dumps(params, sort_keys=True),
                        "cv_ap_mean": scores["ap"].mean(), "cv_ap_std": scores["ap"].std(ddof=1),
                        "cv_auc_mean": scores["auc"].mean(), "cv_auc_std": scores["auc"].std(ddof=1),
                        "cv_brier_mean": scores["brier"].mean()})
    results = pd.DataFrame(records)
    key = {"ap": "cv_ap_mean", "auc": "cv_auc_mean"}[metric]
    best = int(results[key].idxmax())
    results["selected"] = [i == best for i in range(len(results))]
    return TuneResult(json.loads(results.loc[best, "params"]), results)
