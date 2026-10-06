"""E2 to E8 end to end on small synthetic data: every script writes its files and manifest entries, scores the
holdout once per final model, and stops if an AUC looks leaky. The real numbers come from running the scripts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from helpers_ml import synthetic_kaggle, synthetic_openml
from ml.src import (data, e2_baselines, e3_kaggle_models, e4_ablation, e5_openml_models, e6_transfer, e7_imbalance,
                    e8_calibration, experiment)

FAST_GRIDS = {
    "logistic_regression": {"C": [0.1, 1.0]},
    "random_forest": {"n_estimators": [25], "min_samples_leaf": [30]},
    "hist_gradient_boosting": {"learning_rate": [0.1], "max_iter": [30], "max_leaf_nodes": [8]},
}
N_BOOT = 25
METRICS = ["auc_roc", "avg_precision", "brier", "sens_at_prec_30", "sens_at_prec_40", "sens_at_prec_50"]


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    base = tmp_path_factory.mktemp("work")
    (base / "results").mkdir()
    (base / "raw.csv").write_text("stand-in for a raw data file, so the manifest has something to hash\n")
    return base


@pytest.fixture(scope="module")
def out(workdir):
    return workdir / "results"


@pytest.fixture(scope="module")
def kaggle(workdir):
    return experiment.prepare_kaggle(clean=synthetic_kaggle(n=6000, n_patients=1000, seed=21),
                                     source_files=[workdir / "raw.csv"])


@pytest.fixture(scope="module")
def openml(workdir):
    return experiment.prepare_openml(clean=synthetic_openml(n=5000, seed=22), source_files=[workdir / "raw.csv"])


def table(out, name):
    return pd.read_csv(out / name)


def manifest(out):
    return json.loads((out / "manifest.json").read_text())


def check_metric_columns(frame):
    for name in METRICS:
        assert {name, f"{name}_lo", f"{name}_hi"} <= set(frame.columns), name
        assert (frame[f"{name}_lo"] <= frame[name] + 1e-12).all() and (frame[name] <= frame[f"{name}_hi"] + 1e-12).all(), name


# ---- E2 ---------------------------------------------------------------------------------------------
def test_e2_scores_the_two_baselines(kaggle, openml, out):
    e2_baselines.run(kaggle=kaggle, openml=openml, results_dir=out, n_boot=N_BOOT, repo_root=out.parent)
    frame = table(out, "e2_baselines.csv")
    assert list(zip(frame["dataset"], frame["model"])) == [
        ("kaggle", "always_show"), ("kaggle", "earlier_no_show_rate_rule"), ("openml", "always_show")]
    check_metric_columns(frame)
    always = frame.iloc[0]
    assert always["auc_roc"] == 0.5 and always["sens_at_prec_30"] == 0.0
    assert always["accuracy"] == pytest.approx(1 - kaggle.y_holdout.mean())
    rule = frame.iloc[1]
    assert rule["auc_roc"] > 0.5  # a patient's earlier no-shows really are informative in the synthetic data
    assert 0 < rule["cutoff"] <= 1 and rule["precision_at_cutoff"] >= 0
    assert "results/e2_baselines.csv" in manifest(out)


# ---- E3 ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def e3(kaggle, out):
    return e3_kaggle_models.run(prepared=kaggle, results_dir=out, n_boot=N_BOOT, grids=FAST_GRIDS, repo_root=out.parent)


def test_e3_scores_three_models_plus_a_shuffled_label_control(e3, out):
    frame = table(out, "e3_kaggle_models.csv")
    assert list(frame["model"]) == ["logistic_regression", "random_forest", "hist_gradient_boosting", "control_shuffled_labels"]
    check_metric_columns(frame)
    control = frame.iloc[3]
    assert 0.44 < control["auc_roc"] < 0.56  # a model trained on shuffled labels is a coin flip
    assert (frame.iloc[:3]["auc_roc"] > 0.6).all()


def test_e3_records_the_tuning_grid_the_selection_and_the_ladder_comparisons(e3, out):
    tuning = table(out, "e3_tuning.csv")
    assert {"model", "params", "cv_ap_mean", "cv_ap_std", "cv_auc_mean", "selected"} <= set(tuning.columns)
    assert tuning.groupby("model")["selected"].sum().eq(1).all()  # exactly one winner per model
    selected = json.loads((out / "e3_selected.json").read_text())
    assert selected["model"] in {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    assert selected["chosen_by"] == "cross-validated average precision on the training part"
    ladder = table(out, "e3_ladder.csv")
    assert list(ladder["step"]) == ["earlier_no_show_rate_rule -> logistic_regression",
                                    "logistic_regression -> random_forest", "random_forest -> hist_gradient_boosting"]
    assert {"delta_auc_roc", "delta_auc_roc_lo", "delta_auc_roc_hi", "delta_avg_precision", "beats_previous_step"} <= set(ladder.columns)


def test_e3_draws_the_roc_and_precision_recall_figures(e3, out):
    for name in ("e3_roc.png", "e3_pr.png"):
        assert (out / "figures" / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_e3_scored_each_final_model_on_the_holdout_exactly_once(e3):
    assert sorted(e3["scored"]) == sorted(set(e3["scored"]))  # no key twice
    assert len(e3["scored"]) == 4


# ---- E4 ---------------------------------------------------------------------------------------------
def test_e4_ablates_each_feature_group_with_cross_validation_only(kaggle, e3, out):
    e4_ablation.run(prepared=kaggle, results_dir=out, n_boot=N_BOOT, grids=FAST_GRIDS, repo_root=out.parent)
    frame = table(out, "e4_ablation.csv")
    variants = set(frame["variant"])
    groups = ["history", "lead_calendar", "demographics", "flags", "neighbourhood"]
    assert {"all_features", "no_sms_received"} | {f"drop_{g}" for g in groups} | {f"only_{g}" for g in groups} == variants
    assert {"cv_auc_mean", "cv_auc_std", "cv_ap_mean", "cv_ap_std", "delta_auc_vs_all", "delta_ap_vs_all", "n_features"} <= set(frame.columns)
    assert frame.loc[frame["variant"] == "all_features", "delta_auc_vs_all"].iloc[0] == 0.0
    assert not any(c.startswith("holdout") for c in frame.columns)  # the holdout is untouched by the ablation


def test_e4_cross_tabulates_sms_against_lead_time(kaggle, e3, out):
    e4_ablation.run(prepared=kaggle, results_dir=out, n_boot=N_BOOT, grids=FAST_GRIDS, repo_root=out.parent)
    crosstab = table(out, "e4_sms_crosstab.csv")
    assert {"lead_time_band", "sms_received", "n", "no_show_n", "rate", "ci_low", "ci_high", "share_of_band_with_sms"} <= set(crosstab.columns)
    assert set(crosstab["sms_received"]) <= {0, 1}
    assert crosstab["n"].sum() == len(kaggle.y_train)  # training part only


# ---- E5 ---------------------------------------------------------------------------------------------
def test_e5_runs_the_same_ladder_on_openml(openml, out):
    e5_openml_models.run(prepared=openml, results_dir=out, n_boot=N_BOOT, grids=FAST_GRIDS, repo_root=out.parent)
    frame = table(out, "e5_openml_models.csv")
    assert list(frame["model"]) == ["always_show", "logistic_regression", "random_forest", "hist_gradient_boosting"]
    assert (frame["dataset"] == "openml").all()
    check_metric_columns(frame)
    assert (frame.iloc[1:]["auc_roc"] > 0.55).all()
    assert (out / "e5_tuning.csv").exists()


# ---- E6 ---------------------------------------------------------------------------------------------
def test_e6_trains_on_one_dataset_and_tests_on_the_other_with_common_features_only(kaggle, openml, out):
    e6_transfer.run(kaggle=kaggle, openml=openml, results_dir=out, n_boot=N_BOOT, grids=FAST_GRIDS, repo_root=out.parent)
    frame = table(out, "e6_transfer.csv")
    assert set(frame["direction"]) == {"kaggle_to_openml", "openml_to_kaggle"}
    assert set(frame["model"]) == {"logistic_regression", "hist_gradient_boosting"}
    assert {"auc_within_target", "auc_transfer", "auc_drop", "auc_drop_lo", "auc_drop_hi", "features"} <= set(frame.columns)
    assert np.allclose(frame["auc_drop"], frame["auc_within_target"] - frame["auc_transfer"])
    assert all(f == "age,is_female,lead_days,lead_same_day,appt_weekday" for f in frame["features"])


# ---- E7 ---------------------------------------------------------------------------------------------
def test_e7_compares_imbalance_strategies_as_an_ablation(kaggle, e3, out):
    e7_imbalance.run(prepared=kaggle, results_dir=out, n_boot=N_BOOT, grids=FAST_GRIDS, repo_root=out.parent)
    frame = table(out, "e7_imbalance.csv")
    strategies = {"none", "class_weights", "threshold_tuning", "random_under_sampling", "smote", "nearmiss"}
    assert set(frame["strategy"]) == strategies and set(frame["base_model"]) == {"logistic_regression", "hist_gradient_boosting"}
    assert {"mean_predicted", "prevalence", "f1_at_cv_threshold", "precision_at_cv_threshold", "recall_at_cv_threshold"} <= set(frame.columns)
    check_metric_columns(frame)
    resampled = frame[frame["strategy"].isin(["random_under_sampling", "smote", "nearmiss"]) & (frame["base_model"] == "logistic_regression")]
    plain = frame[(frame["strategy"] == "none") & (frame["base_model"] == "logistic_regression")]
    assert (resampled["mean_predicted"] > plain["mean_predicted"].iloc[0] + 0.05).all()  # resampling inflates probabilities


# ---- E8 ---------------------------------------------------------------------------------------------
def test_e8_chooses_a_calibration_method_by_cross_validated_brier_and_draws_the_reliability_figure(kaggle, e3, out):
    e8_calibration.run(prepared=kaggle, results_dir=out, n_boot=N_BOOT, grids=FAST_GRIDS, repo_root=out.parent)
    frame = table(out, "e8_calibration.csv")
    cv = frame[frame["stage"] == "grouped_cv"]
    assert set(cv["method"]) == {"uncalibrated", "sigmoid", "isotonic"}
    chosen = frame[frame["stage"] == "holdout"]
    assert len(chosen) == 1 and chosen["method"].iloc[0] == cv.loc[cv["brier_cv"].idxmin(), "method"]
    assert {"brier", "brier_lo", "brier_hi", "ece", "brier_uncalibrated", "ece_uncalibrated"} <= set(chosen.columns)
    reliability = table(out, "e8_reliability.csv")
    assert {"version", "bin", "n", "mean_predicted", "observed_rate"} <= set(reliability.columns)
    assert set(reliability["version"]) == {"uncalibrated", "calibrated"}
    assert (out / "figures" / "e8_reliability.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ---- every output is in the manifest -------------------------------------------------------------------
def test_every_result_file_has_a_manifest_entry_with_seed_and_data_hash(out):
    stored = manifest(out)
    written = {f"results/{p.relative_to(out).as_posix()}" for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"
               and p.suffix in {".csv", ".png"}}
    assert written and written <= set(stored), sorted(written - set(stored))
    for entry in stored.values():
        assert entry["seed"] == data.DEFAULT_SEED and entry["data_files"]
