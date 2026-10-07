"""E9 (the deployable model, its artifact and model card) and E10 (risk thresholds), on small synthetic data."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest

from helpers_ml import synthetic_kaggle
from ml.src import (data, e3_kaggle_models, e8_calibration, e9_deployable, e10_thresholds, evaluate, experiment,
                    features)

FAST_GRIDS = {
    "logistic_regression": {"C": [1.0]},
    "random_forest": {"n_estimators": [25], "min_samples_leaf": [30]},
    "hist_gradient_boosting": {"learning_rate": [0.1], "max_iter": [40], "max_leaf_nodes": [8]},
}
N_BOOT = 30
SIX = features.DEPLOYABLE_FEATURES


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    base = tmp_path_factory.mktemp("m5")
    (base / "results").mkdir()
    (base / "raw.csv").write_text("stand-in for the raw Kaggle file\n")
    return base


@pytest.fixture(scope="module")
def kaggle(workdir):
    return experiment.prepare_kaggle(clean=synthetic_kaggle(n=7000, n_patients=1200, seed=41, signal=2.0),
                                     source_files=[workdir / "raw.csv"])


@pytest.fixture(scope="module")
def prerequisites(kaggle, workdir):
    """E3 (which model family won) and E8 (which calibration) must exist before E9, as on the real data."""
    out = workdir / "results"
    e3_kaggle_models.run(prepared=kaggle, results_dir=out, n_boot=N_BOOT, grids=FAST_GRIDS, repo_root=workdir)
    e8_calibration.run(prepared=kaggle, results_dir=out, n_boot=N_BOOT, grids=FAST_GRIDS, repo_root=workdir)
    return out


@pytest.fixture(scope="module")
def e9(kaggle, workdir, prerequisites):
    return e9_deployable.run(prepared=kaggle, results_dir=prerequisites, artifacts_dir=workdir / "artifacts",
                             n_boot=N_BOOT, grids=FAST_GRIDS, repo_root=workdir)


@pytest.fixture(scope="module")
def e10(kaggle, workdir, e9):
    return e10_thresholds.run(prepared=kaggle, results_dir=workdir / "results", artifacts_dir=workdir / "artifacts",
                              n_boot=N_BOOT, repo_root=workdir)


def card(workdir):
    return json.loads((workdir / "artifacts" / "model_card.json").read_text())


# ---- the data the deployable model trains on -----------------------------------------------------------
def test_the_training_matrix_comes_from_the_serving_function_and_is_checked_against_the_vectorised_path(kaggle):
    X_train, X_holdout = experiment.deployable_data(kaggle)
    assert list(X_train.columns) == list(X_holdout.columns) == SIX
    assert len(X_train) == len(kaggle.y_train) and len(X_holdout) == len(kaggle.y_holdout)
    assert not {"sms_received", "neighbourhood", "lead_same_day", "has_history"} & set(X_train.columns)


def test_a_difference_between_the_two_feature_paths_stops_the_run(kaggle, monkeypatch):
    real = features.deployable_matrix

    def drifting(frame):
        out = real(frame).copy()
        out.iloc[0, 0] += 1  # one age off by a year: exactly the kind of drift this check exists to catch
        return out

    monkeypatch.setattr(features, "deployable_matrix", drifting)
    with pytest.raises(AssertionError, match="training and serving"):
        experiment.deployable_data(kaggle)


# ---- E9 ---------------------------------------------------------------------------------------------
def test_e9_saves_a_model_and_a_card_and_uses_the_calibration_e8_chose(e9, workdir, prerequisites):
    assert (workdir / "artifacts" / "risk_model.joblib").stat().st_size > 0
    e8 = pd.read_csv(prerequisites / "e8_calibration.csv")
    chosen_in_e8 = e8.loc[e8["stage"] == "holdout", "method"].iloc[0]
    assert card(workdir)["calibration"]["method"] == chosen_in_e8
    assert card(workdir)["calibration"]["chosen_in"] == "E8"


def test_the_saved_model_takes_exactly_the_six_features_and_returns_probabilities(e9, workdir):
    model = joblib.load(workdir / "artifacts" / "risk_model.joblib")
    assert list(model.feature_names_in_) == SIX
    one_booking = pd.DataFrame([features.deployable_features(
        age=34, sex="F", booked_on=pd.Timestamp("2016-05-02").date(), appointment_date=pd.Timestamp("2016-05-09").date(),
        history=[(pd.Timestamp("2016-04-20").date(), 1)])])[SIX]
    p = model.predict_proba(one_booking)[:, 1]
    assert p.shape == (1,) and 0 < p[0] < 1


def test_a_booking_served_through_the_app_function_gets_the_same_risk_as_in_batch(e9, kaggle, workdir):
    """End to end: the serving path (plain values -> function -> model) equals the batch path for real bookings."""
    model = joblib.load(workdir / "artifacts" / "risk_model.joblib")
    X_train, X_holdout = experiment.deployable_data(kaggle)
    batch = model.predict_proba(X_holdout)[:, 1]
    frame = pd.concat([kaggle.train_frame, kaggle.holdout_frame], ignore_index=True)
    history = {p: list(zip(g["appointment_date"].dt.date, g["no_show"])) for p, g in frame.groupby("patient_id")}
    for position in np.random.default_rng(0).choice(len(kaggle.holdout_frame), 25, replace=False):
        r = kaggle.holdout_frame.iloc[position]
        served = features.deployable_features(age=int(r["age"]), sex=r["sex"], booked_on=r["scheduled_at"].date(),
                                              appointment_date=r["appointment_date"].date(), history=history[r["patient_id"]])
        p = model.predict_proba(pd.DataFrame([served])[SIX])[:, 1][0]
        assert p == pytest.approx(batch[position], abs=1e-12)


def test_the_model_card_has_everything_the_plan_asks_for(e9, workdir):
    c = card(workdir)
    assert c["version"] and c["seed"] == data.DEFAULT_SEED and c["created"]
    assert [f["name"] for f in c["features"]] == SIX and all(f["description"] for f in c["features"])
    assert c["target"].startswith("no_show = 1")
    training = c["training_data"]
    assert training["sha256"] == hashlib.sha256((workdir / "raw.csv").read_bytes()).hexdigest()
    assert pd.Timestamp(training["training_appointment_dates"]["last"]) < pd.Timestamp(training["holdout_appointment_dates"]["first"])
    assert training["rows_train"] > 0 and training["rows_holdout"] > 0
    assert c["model"]["family"] in {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    assert c["model"]["params"] and "grouped" in c["model"]["tuned_by"]
    assert set(c["calibration"]["cv_brier_check"]) == {"uncalibrated", "sigmoid", "isotonic"}
    for key in ("auc_roc", "avg_precision", "brier", "sens_at_prec_30", "sens_at_prec_40", "sens_at_prec_50"):
        assert c["metrics"]["holdout"][key]["lo"] <= c["metrics"]["holdout"][key]["value"] <= c["metrics"]["holdout"][key]["hi"]
    assert c["metrics"]["holdout"]["n_boot"] == N_BOOT
    assert len(c["limitations"]) >= 6 and all(isinstance(t, str) and t for t in c["limitations"])
    assert "advisory" in c["intended_use"].lower()
    assert {"python", "scikit-learn", "numpy", "pandas"} <= set(c["environment"])
    assert c["artifact_sha256"] == hashlib.sha256((workdir / "artifacts" / "risk_model.joblib").read_bytes()).hexdigest()
    assert c["thresholds"] is None  # E10 fills this in


def test_the_card_row_counts_match_the_data(e9, kaggle, workdir):
    t = card(workdir)["training_data"]
    assert t["rows_train"] == len(kaggle.y_train) and t["rows_holdout"] == len(kaggle.y_holdout)


def test_e9_scores_the_final_model_on_the_holdout_once_and_compares_it_with_the_research_model(e9, workdir):
    assert e9["scored"] == ["deployable"]
    table = pd.read_csv(workdir / "results" / "e9_deployable.csv")
    holdout = table[table["stage"] == "holdout"].set_index("model")
    assert set(holdout.index) == {"deployable", "research_all_features"}
    deployable = holdout.loc["deployable"]
    for name in evaluate.METRIC_NAMES:
        assert deployable[f"{name}_lo"] <= deployable[name] <= deployable[f"{name}_hi"]
    assert {"delta_auc_roc_vs_research", "delta_auc_roc_vs_research_lo", "delta_auc_roc_vs_research_hi",
            "delta_avg_precision_vs_research"} <= set(table.columns)
    assert abs(deployable["delta_auc_roc_vs_research"]) < 0.1  # the six-feature model is close to the full one
    check = table[table["stage"] == "grouped_cv_calibration_check"]
    assert set(check["method"]) == {"uncalibrated", "sigmoid", "isotonic"}


def test_every_m5_output_is_in_the_manifest(e9, workdir):
    stored = json.loads((workdir / "results" / "manifest.json").read_text())
    for key in ("results/e9_deployable.csv", "artifacts/risk_model.joblib", "artifacts/model_card.json"):
        assert key in stored, key
        assert stored[key]["seed"] == data.DEFAULT_SEED and stored[key]["data_files"]


def test_a_sigmoid_calibrated_model_also_round_trips(kaggle, workdir, prerequisites, monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "load_calibration_choice", lambda results_dir: "sigmoid")
    result = e9_deployable.run(prepared=kaggle, results_dir=prerequisites, artifacts_dir=tmp_path, n_boot=N_BOOT,
                               grids=FAST_GRIDS, repo_root=workdir)
    model = joblib.load(tmp_path / "risk_model.joblib")
    p = model.predict_proba(experiment.deployable_data(kaggle)[1])[:, 1]
    assert ((p > 0) & (p < 1)).all()
    assert json.loads((tmp_path / "model_card.json").read_text())["calibration"]["method"] == "sigmoid"
    assert result["scored"] == ["deployable"]


# ---- E10: choosing a threshold ------------------------------------------------------------------------
def calibrated_scores(n=60000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(2, 7, n)
    return (rng.random(n) < p).astype(int), p


def test_the_threshold_for_a_precision_target_reaches_it_and_stops_where_it_stops_holding():
    y, p = calibrated_scores()
    t30 = e10_thresholds.threshold_for_precision(y, p, 0.30)
    t40 = e10_thresholds.threshold_for_precision(y, p, 0.40)
    assert t40 > t30 > 0  # a stricter precision target needs a higher cut-off
    for t, target in ((t30, 0.30), (t40, 0.40)):
        assert y[p >= t].mean() >= target
        assert y[p >= round(t - 0.01, 2)].mean() < target  # one step lower and the target is lost
    assert t30 == round(t30, 2)  # a clean two-decimal number, as the app will show it


def test_an_unreachable_precision_target_gives_none_rather_than_a_lucky_tiny_slice():
    y, p = calibrated_scores()
    assert e10_thresholds.threshold_for_precision(y, p, 0.95) is None
    # a handful of top-scored rows can be 100% precise by chance; the minimum share keeps those from counting
    y2 = np.r_[np.ones(5), np.zeros(5000)]
    p2 = np.r_[np.full(5, 0.99), np.full(5000, 0.1)]
    assert e10_thresholds.threshold_for_precision(y2, p2, 0.9, min_share=0.01) is None
    unrestricted = e10_thresholds.threshold_for_precision(y2, p2, 0.9, min_share=0.0)  # lowest cut-off still flagging just those 5
    assert (p2 >= unrestricted).sum() == 5 and y2[p2 >= unrestricted].mean() == 1.0


# ---- E10 ---------------------------------------------------------------------------------------------
def test_e10_tabulates_fixed_thresholds_percentiles_and_the_chosen_ones(e10, workdir):
    table = pd.read_csv(workdir / "results" / "e10_thresholds.csv")
    fixed = table[table["kind"] == "fixed"]
    assert list(fixed["threshold"].round(2)) == [round(t, 2) for t in np.arange(0.20, 0.7001, 0.05)]
    pct = table[table["kind"] == "percentile"]
    assert list(pct["percentile"]) == [50, 70, 80, 90, 95]
    assert pct["threshold"].is_monotonic_increasing
    assert set(table[table["kind"] == "chosen"]["label"]) == {"medium", "high"}
    needed = {"threshold", "flagged_n", "share_flagged", "share_flagged_lo", "share_flagged_hi", "precision", "precision_lo",
              "precision_hi", "sensitivity", "sensitivity_lo", "sensitivity_hi", "oof_share_flagged", "oof_precision",
              "oof_sensitivity"}
    assert needed <= set(table.columns)


def test_a_higher_threshold_flags_fewer_bookings(e10, workdir):
    table = pd.read_csv(workdir / "results" / "e10_thresholds.csv")
    fixed = table[table["kind"] == "fixed"].sort_values("threshold")
    assert fixed["share_flagged"].is_monotonic_decreasing and fixed["oof_share_flagged"].is_monotonic_decreasing
    pct = table[table["kind"] == "percentile"]
    assert (pct["oof_share_flagged"] - (1 - pct["percentile"] / 100)).abs().max() < 0.02  # p-th percentile flags the rest


def test_the_card_gets_medium_and_high_thresholds_with_their_reasoning(e10, workdir):
    t = card(workdir)["thresholds"]
    assert 0 < t["medium"]["threshold"] < t["high"]["threshold"] < 1
    assert t["medium"]["target_precision"] == 0.30 and t["high"]["target_precision"] == 0.40
    assert t["medium"]["oof_precision"] >= 0.30 and t["high"]["oof_precision"] >= 0.40  # met where they were chosen
    assert "training" in t["chosen_on"] and "holdout" in t["chosen_on"]
    assert len(t["reasoning"]) > 100 and "advisory" in t["reasoning"].lower()
    for level in ("medium", "high"):
        assert {"holdout_precision", "holdout_sensitivity", "holdout_share_flagged"} <= set(t[level])


def test_the_thresholds_in_the_card_deliver_what_the_card_says_on_the_holdout(e10, kaggle, workdir):
    model = joblib.load(workdir / "artifacts" / "risk_model.joblib")
    _, X_holdout = experiment.deployable_data(kaggle)
    p, y = model.predict_proba(X_holdout)[:, 1], kaggle.y_holdout
    t = card(workdir)["thresholds"]
    for level in ("medium", "high"):
        flagged = p >= t[level]["threshold"]
        assert t[level]["holdout_share_flagged"] == pytest.approx(flagged.mean())
        assert t[level]["holdout_precision"] == pytest.approx(y[flagged].mean())
        assert t[level]["holdout_sensitivity"] == pytest.approx(y[flagged].sum() / y.sum())


def test_the_thresholds_are_chosen_without_looking_at_the_holdout(kaggle, workdir, e10, tmp_path):
    """Flip every holdout label: thresholds must not move (they come from training predictions only), while the
    holdout precision and sensitivity reported for them must."""
    flipped = dataclasses.replace(kaggle, y_holdout=1 - kaggle.y_holdout)
    shutil.copytree(workdir / "artifacts", tmp_path / "artifacts")
    (tmp_path / "results").mkdir()
    e10_thresholds.run(prepared=flipped, results_dir=tmp_path / "results", artifacts_dir=tmp_path / "artifacts",
                       n_boot=N_BOOT, repo_root=workdir)
    original, other = card(workdir)["thresholds"], json.loads((tmp_path / "artifacts" / "model_card.json").read_text())["thresholds"]
    for level in ("medium", "high"):
        assert original[level]["threshold"] == other[level]["threshold"]
        assert original[level]["holdout_precision"] != other[level]["holdout_precision"]
    a = pd.read_csv(workdir / "results" / "e10_thresholds.csv")
    b = pd.read_csv(tmp_path / "results" / "e10_thresholds.csv")
    assert (a["threshold"] == b["threshold"]).all()  # the percentile thresholds too


def test_e10_stops_if_the_holdout_auc_looks_leaky(kaggle, workdir, e10, tmp_path):
    """A holdout whose labels the model's own score separates perfectly would give an implausible AUC."""
    model = joblib.load(workdir / "artifacts" / "risk_model.joblib")
    p = model.predict_proba(experiment.deployable_data(kaggle)[1])[:, 1]
    leaky = dataclasses.replace(kaggle, y_holdout=(p >= np.quantile(p, 0.8)).astype(int))
    shutil.copytree(workdir / "artifacts", tmp_path / "artifacts")
    (tmp_path / "results").mkdir()
    with pytest.raises(evaluate.LeakageSuspected):
        e10_thresholds.run(prepared=leaky, results_dir=tmp_path / "results", artifacts_dir=tmp_path / "artifacts",
                           n_boot=N_BOOT, repo_root=workdir)
