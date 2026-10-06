"""Metrics, the patient-level bootstrap, paired comparisons and the score-the-holdout-once guard."""
from __future__ import annotations

import numpy as np
import pytest

from ml.src import evaluate

METRICS = ["auc_roc", "avg_precision", "brier", "sens_at_prec_30", "sens_at_prec_40", "sens_at_prec_50"]


def test_known_values_for_auc_average_precision_and_brier():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.4, 0.35, 0.8])
    got = evaluate.point_metrics(y, score)
    assert got["auc_roc"] == pytest.approx(0.75)
    assert got["avg_precision"] == pytest.approx(5 / 6)
    assert got["brier"] == pytest.approx((0.1 ** 2 + 0.4 ** 2 + 0.65 ** 2 + 0.2 ** 2) / 4)


def test_sensitivity_at_a_fixed_precision_on_a_worked_example():
    # ranked: 1 0 1 0 0 1 0 0 0 0   precision after each step: 1, .5, .67, .5, .4, .5, .43, .38, .33, .3
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    score = np.linspace(0.9, 0.0, 10)
    assert evaluate.sensitivity_at_precision(y, score, 0.6) == pytest.approx(2 / 3)  # best reachable: top 3, recall 2/3
    assert evaluate.sensitivity_at_precision(y, score, 1.0) == pytest.approx(1 / 3)  # only the first case is pure
    assert evaluate.sensitivity_at_precision(y, score, 0.35) == pytest.approx(1.0)  # top 7 is 43% precise and finds all


def test_a_precision_that_no_threshold_reaches_gives_zero_sensitivity():
    y = np.array([1, 0, 0, 0, 0] * 20)  # 20% prevalence
    assert evaluate.sensitivity_at_precision(y, np.zeros(100), 0.30) == 0.0  # a constant score is never 30% precise


def test_always_show_is_useless_even_though_it_is_80_percent_accurate():
    y = np.array([1, 0, 0, 0, 0] * 200)
    scores = evaluate.always_show_scores(len(y))
    got = evaluate.point_metrics(y, scores)
    assert got["auc_roc"] == 0.5
    assert got["avg_precision"] == pytest.approx(y.mean())
    assert got["brier"] == pytest.approx(y.mean())  # predicting probability 0 for everyone
    assert got["sens_at_prec_30"] == got["sens_at_prec_40"] == got["sens_at_prec_50"] == 0.0
    assert evaluate.accuracy_at(y, scores, threshold=0.5) == pytest.approx(0.8)


def test_weights_are_the_same_as_repeating_rows():
    rng = np.random.default_rng(0)
    y = (rng.random(60) < 0.3).astype(int)
    score = np.where(y == 1, rng.beta(3, 2, 60), rng.beta(2, 3, 60))
    weights = rng.integers(0, 4, 60)
    repeated = np.repeat(np.arange(60), weights)
    a = evaluate.point_metrics(y, score, sample_weight=weights)
    b = evaluate.point_metrics(y[repeated], score[repeated])
    for name in METRICS:
        assert a[name] == pytest.approx(b[name]), name


# ---- bootstrap --------------------------------------------------------------------------------------
def make_scored(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    score = np.clip(0.2 + 0.15 * (y - 0.2) + rng.normal(0, 0.18, n), 0, 1)  # AUC about 0.72
    return y, score


def test_default_number_of_resamples_is_1000():
    assert evaluate.N_BOOT == 1000


def test_the_interval_contains_the_point_estimate_and_is_reproducible():
    y, score = make_scored()
    groups = np.arange(len(y)) // 2  # two rows per patient
    a = evaluate.metrics_with_ci(y, score, groups, n_boot=200, seed=5)
    b = evaluate.metrics_with_ci(y, score, groups, n_boot=200, seed=5)
    assert a == b
    for name in METRICS:
        assert a[f"{name}_lo"] <= a[name] <= a[f"{name}_hi"], name
    assert a["n"] == len(y) and a["n_positive"] == int(y.sum()) and a["n_boot"] == 200
    assert evaluate.metrics_with_ci(y, score, groups, n_boot=200, seed=6)["auc_roc_lo"] != a["auc_roc_lo"]


def test_resampling_patients_gives_wider_intervals_than_resampling_rows_when_rows_cluster():
    """20 rows per patient with identical outcome and score: the effective sample size is the patient count."""
    rng = np.random.default_rng(1)
    patients = 120
    y_p = (rng.random(patients) < 0.25).astype(int)
    s_p = np.clip(0.25 + 0.2 * (y_p - 0.25) + rng.normal(0, 0.15, patients), 0, 1)
    y, score, groups = np.repeat(y_p, 20), np.repeat(s_p, 20), np.repeat(np.arange(patients), 20)
    by_patient = evaluate.metrics_with_ci(y, score, groups, n_boot=300, seed=2)
    by_row = evaluate.metrics_with_ci(y, score, None, n_boot=300, seed=2)
    width = lambda r: r["auc_roc_hi"] - r["auc_roc_lo"]
    assert width(by_patient) > 2 * width(by_row)


def test_a_patients_rows_always_travel_together_in_a_resample():
    y, score = make_scored(600)
    groups = np.repeat(np.arange(200), 3)
    replicates = evaluate.bootstrap_weights(groups, n_boot=50, seed=1)
    assert replicates.shape == (50, 600)
    for weights in replicates:
        assert (weights.reshape(200, 3) == weights.reshape(200, 3)[:, :1]).all()  # same weight for each of a patient's rows
        assert weights.sum() % 3 == 0 and weights.sum() == 3 * 200  # 200 patients drawn, with replacement


def test_without_a_patient_id_rows_are_resampled_directly():
    replicates = evaluate.bootstrap_weights(None, n_boot=20, seed=1, n_rows=500)
    assert replicates.shape == (20, 500) and (replicates.sum(axis=1) == 500).all()


def test_paired_difference_is_zero_for_identical_scores_and_positive_for_a_better_model():
    y, good = make_scored(3000, seed=3)
    rng = np.random.default_rng(9)
    weak = np.clip(good + rng.normal(0, 0.3, len(good)), 0, 1)
    groups = np.arange(len(y)) // 2
    same = evaluate.paired_difference(y, good, good, groups, "auc_roc", n_boot=100, seed=1)
    assert same == (0.0, 0.0, 0.0)
    diff, lo, hi = evaluate.paired_difference(y, good, weak, groups, "auc_roc", n_boot=200, seed=1)
    assert lo <= diff <= hi and lo > 0  # the better model's gain is clearly above zero


# ---- the guard --------------------------------------------------------------------------------------
def test_a_final_model_can_be_scored_on_the_holdout_only_once():
    y, score = make_scored(400)
    guard = evaluate.HoldoutGuard(n_boot=20, seed=1)
    result = guard.score("random_forest", y, score)
    assert result["auc_roc"] == pytest.approx(evaluate.point_metrics(y, score)["auc_roc"])
    with pytest.raises(evaluate.HoldoutAlreadyScored, match="random_forest"):
        guard.score("random_forest", y, score)
    guard.score("hist_gradient_boosting", y, score)  # another model is fine
    assert guard.scored == ["random_forest", "hist_gradient_boosting"]


def test_a_suspicious_holdout_auc_is_stopped_at_scoring_time():
    y = np.array([0, 1] * 200)
    guard = evaluate.HoldoutGuard(n_boot=10, seed=1)
    with pytest.raises(evaluate.LeakageSuspected):
        guard.score("leaky", y, y.astype(float))  # a perfect score is a leak, not a result
