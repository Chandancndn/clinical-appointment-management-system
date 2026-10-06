"""Calibration helpers: ECE, reliability tables and choosing Platt vs isotonic by out-of-fold Brier."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from helpers_ml import synthetic_kaggle
from ml.src import calibrate, experiment, features, train


def test_a_perfectly_calibrated_forecast_has_zero_ece():
    p = np.repeat([0.1, 0.5, 0.9], 1000)
    y = np.concatenate([np.r_[np.ones(100), np.zeros(900)], np.r_[np.ones(500), np.zeros(500)], np.r_[np.ones(900), np.zeros(100)]])
    assert calibrate.expected_calibration_error(y, p, bins=10) == pytest.approx(0.0, abs=1e-9)


def test_an_overconfident_forecast_has_the_ece_you_can_compute_by_hand():
    p = np.full(1000, 0.8)
    y = np.r_[np.ones(500), np.zeros(500)]  # says 80%, happens 50%
    assert calibrate.expected_calibration_error(y, p, bins=10) == pytest.approx(0.3)


def test_the_reliability_table_covers_every_row_once():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    table = calibrate.reliability_table(y, p, bins=10)
    assert table["n"].sum() == 1000 and len(table) == 10
    assert table["mean_predicted"].is_monotonic_increasing
    assert (table["observed_rate"] - table["mean_predicted"]).abs().max() < 0.12  # a calibrated forecast hugs the diagonal


def test_the_method_with_the_lowest_brier_is_chosen_and_ties_keep_the_simpler_one():
    assert calibrate.choose_method({"uncalibrated": 0.20, "sigmoid": 0.15, "isotonic": 0.14}) == "isotonic"
    assert calibrate.choose_method({"uncalibrated": 0.20, "sigmoid": 0.14, "isotonic": 0.15}) == "sigmoid"
    assert calibrate.choose_method({"uncalibrated": 0.14, "sigmoid": 0.14, "isotonic": 0.14}) == "uncalibrated"


def test_calibrating_a_class_weighted_model_lowers_the_out_of_fold_brier_score():
    """Balanced class weights inflate every probability; Platt scaling should pull them back."""
    prep = experiment.prepare_kaggle(clean=synthetic_kaggle(n=6000, seed=11))
    columns = features.all_columns("kaggle")
    make = lambda: train.build_pipeline("logistic_regression", columns, params={})
    X, y, groups = prep.X_train[columns], prep.y_train, prep.groups_train
    raw = calibrate.oof_predictions(make, X, y, groups, prep.folds, "uncalibrated", seed=1)
    platt = calibrate.oof_predictions(make, X, y, groups, prep.folds, "sigmoid", seed=1)
    assert brier_score_loss(y, platt) < brier_score_loss(y, raw)
    assert raw.mean() > y.mean() + 0.05  # the class-weighted model really does over-predict
    assert abs(platt.mean() - y.mean()) < 0.02  # and calibration brings the average back
