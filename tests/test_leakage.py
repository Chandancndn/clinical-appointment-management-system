"""Leakage and correctness tests for the research models (CLAUDE.md "Leakage and correctness tests").

1. history features use only appointments whose date is before the current BOOKING date
2. no patient is in both a training fold and its validation fold
3. holdout dates come after training dates
4. preprocessing and encoders are fitted inside each training fold only
5. a model trained on shuffled labels scores an AUC near 0.5
plus: a row's own outcome never reaches its features, identifiers never reach the model, and a suspiciously
high AUC stops the run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from helpers_ml import synthetic_kaggle
from ml.src import data, evaluate, experiment, features, train


# ---- 1. history uses only outcomes known at booking time -------------------------------------------
def appt_rows(*rows) -> pd.DataFrame:
    """(patient, booked date, appointment date, no_show) -> a frame shaped like the cleaned Kaggle data."""
    return pd.DataFrame({
        "patient_id": [r[0] for r in rows],
        "scheduled_at": pd.to_datetime([f"{r[1]} 10:30" for r in rows]),
        "appointment_date": pd.to_datetime([r[2] for r in rows]),
        "no_show": np.array([r[3] for r in rows], dtype="int8"),
    })


def test_history_counts_only_appointments_that_took_place_before_the_booking_date():
    frame = features.add_history(appt_rows(
        (1, "2016-05-01", "2016-05-02", 1),   # A: first appointment, no history yet
        (1, "2016-05-03", "2016-05-10", 0),   # B: booked 3 May. A (2 May) already happened and was a no-show
        (1, "2016-05-10", "2016-05-12", 1),   # C: booked 10 May. B is ON 10 May: its outcome is not known yet
        (2, "2016-05-04", "2016-05-05", 0),   # a different patient never counts for patient 1
    ))
    assert list(frame["prev_appointments"]) == [0, 1, 1, 0]
    assert list(frame["prev_no_shows"]) == [0, 1, 1, 0]
    assert frame["prev_no_show_rate"].tolist()[1:3] == [1.0, 1.0]
    assert np.isnan(frame["prev_no_show_rate"].iloc[0]) and np.isnan(frame["prev_no_show_rate"].iloc[3])
    assert list(frame["has_history"]) == [0, 1, 1, 0]


def test_history_is_stricter_than_before_the_current_appointment():
    """D happened (10 May) before C's appointment (20 May) but after C was booked (3 May): not usable for C."""
    frame = features.add_history(appt_rows(
        (1, "2016-05-01", "2016-05-02", 0),   # A
        (1, "2016-05-03", "2016-05-20", 0),   # C: booked 3 May for 20 May
        (1, "2016-05-04", "2016-05-10", 1),   # D: booked 4 May for 10 May, a no-show
    ))
    c = frame.iloc[1]
    assert c["prev_appointments"] == 1 and c["prev_no_shows"] == 0  # only A; D is "before the appointment" but not before the booking
    assert frame.iloc[2]["prev_appointments"] == 1  # D is booked 4 May: A counts, C (20 May) does not


def brute_force_history(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        booked = row["scheduled_at"].normalize()
        earlier = frame[(frame["patient_id"] == row["patient_id"]) & (frame["appointment_date"] < booked)]
        rows.append((len(earlier), int(earlier["no_show"].sum())))
    return pd.DataFrame(rows, columns=["prev_appointments", "prev_no_shows"], index=frame.index)


def test_history_matches_a_slow_loop_on_random_data():
    frame = synthetic_kaggle(n=400, n_patients=60, seed=3)
    got, want = features.add_history(frame), brute_force_history(frame)
    assert got["prev_appointments"].tolist() == want["prev_appointments"].tolist()
    assert got["prev_no_shows"].tolist() == want["prev_no_shows"].tolist()
    assert (got["prev_appointments"] > 0).sum() > 50  # the comparison is not trivially all zeros


def test_changing_outcomes_on_or_after_the_booking_date_cannot_change_a_rows_history():
    frame = synthetic_kaggle(n=500, n_patients=80, seed=4)
    base = features.add_history(frame)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(frame), 40, replace=False):
        booked = frame.loc[i, "scheduled_at"].normalize()
        # flip every outcome of this patient's appointments dated on or after the booking date, including its own
        mask = (frame["patient_id"] == frame.loc[i, "patient_id"]) & (frame["appointment_date"] >= booked)
        flipped = frame.copy()
        flipped.loc[mask, "no_show"] = 1 - flipped.loc[mask, "no_show"]
        again = features.add_history(flipped)
        for column in ("prev_appointments", "prev_no_shows", "prev_no_show_rate", "has_history"):
            a, b = base.loc[i, column], again.loc[i, column]
            assert (a == b) or (np.isnan(a) and np.isnan(b)), (i, column)


def test_a_rows_own_outcome_never_reaches_any_of_its_features():
    frame = synthetic_kaggle(n=500, n_patients=80, seed=5)
    before = features.kaggle_features(features.add_history(frame))
    for row in (100, 250, 400):
        flipped = frame.copy()
        flipped.loc[row, "no_show"] = 1 - flipped.loc[row, "no_show"]
        after = features.kaggle_features(features.add_history(flipped))
        pd.testing.assert_series_equal(before.loc[row], after.loc[row])  # its own features did not move


def test_identifiers_and_the_target_never_reach_the_feature_matrix():
    matrix = features.kaggle_features(features.add_history(synthetic_kaggle(n=300, seed=6)))
    forbidden = {"no_show", "patient_id", "appointment_id", "scheduled_at", "appointment_date"}
    assert not forbidden & set(matrix.columns)
    assert set(matrix.columns) == set(features.all_columns("kaggle"))


# ---- 2 and 3. the splits the experiments actually use ----------------------------------------------
@pytest.fixture(scope="module")
def prepared():
    return experiment.prepare_kaggle(clean=synthetic_kaggle(n=8000, n_patients=2500, seed=7))


def test_no_patient_is_in_both_a_training_fold_and_its_validation_fold(prepared):
    assert len(prepared.folds) == 5
    patients = prepared.groups_train
    for train_pos, valid_pos in prepared.folds:
        assert not set(patients[train_pos]) & set(patients[valid_pos])
        assert max(train_pos.max(), valid_pos.max()) < len(prepared.y_train)  # folds index the training part only


def test_every_training_row_is_validated_exactly_once(prepared):
    validated = np.concatenate([valid for _, valid in prepared.folds])
    assert sorted(validated) == list(range(len(prepared.y_train)))


def test_holdout_dates_come_after_training_dates(prepared):
    assert prepared.holdout_dates.min() > prepared.train_dates.max()
    assert prepared.holdout_dates.min() == prepared.cut
    assert len(prepared.y_train) + len(prepared.y_holdout) == len(synthetic_kaggle(n=8000, n_patients=2500, seed=7))


def test_the_same_patient_may_appear_in_training_and_holdout_as_in_real_use(prepared):
    # a time-based holdout keeps returning patients: their history is the point of the history features
    assert set(prepared.groups_train) & set(prepared.groups_holdout)


def test_the_holdout_labels_never_enter_a_training_row_through_history(prepared):
    """History of a training row uses outcomes dated before its booking date, so never a holdout outcome."""
    frame = experiment.prepare_kaggle(clean=synthetic_kaggle(n=3000, seed=8))
    train_frame = frame.train_frame
    assert train_frame["appointment_date"].max() < frame.cut
    # recompute each training row's history from training-period appointments only: identical
    only_train = features.add_history(train_frame.drop(columns=[c for c in features.HISTORY_COLUMNS if c in train_frame]))
    for column in ("prev_appointments", "prev_no_shows"):
        assert only_train[column].tolist() == train_frame[column].tolist()


# ---- 4. preprocessing is fitted inside each training fold only -------------------------------------
class SpyTransformer(BaseEstimator, TransformerMixin):
    """Remembers which rows it was fitted on."""

    fitted_on: list = []

    def fit(self, X, y=None):
        SpyTransformer.fitted_on.append(set(X["row_id"]))
        return self

    def transform(self, X):
        return X[["x"]]


def test_the_pipeline_is_refitted_from_scratch_on_each_training_fold_only():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"row_id": np.arange(300), "x": rng.normal(size=300)})
    y = (X["x"] + rng.normal(size=300) > 0).astype(int).to_numpy()
    folds = data.stratified_folds(y, n_splits=4, seed=1)
    SpyTransformer.fitted_on = []
    train.cross_val_predict_proba(lambda: Pipeline([("spy", SpyTransformer()), ("lr", LogisticRegression())]), X, y, folds)
    assert len(SpyTransformer.fitted_on) == 4
    for seen, (train_pos, valid_pos) in zip(SpyTransformer.fitted_on, folds):
        assert seen == set(train_pos)  # fitted on the training fold...
        assert not seen & set(valid_pos)  # ...and never saw a validation row


def test_tuning_goes_through_the_same_fold_by_fold_path(monkeypatch):
    X = pd.DataFrame({"age": np.arange(200) % 80, "lead_days": np.arange(200) % 9})
    y = (np.arange(200) % 3 == 0).astype(int)
    folds = data.stratified_folds(y, n_splits=3, seed=1)
    seen_folds = []
    original = train.cross_val_predict_proba

    def recorder(make_estimator, X_, y_, folds_, **kwargs):
        seen_folds.append(folds_)
        return original(make_estimator, X_, y_, folds_, **kwargs)

    monkeypatch.setattr(train, "cross_val_predict_proba", recorder)
    train.tune("logistic_regression", X, y, folds, columns=["age", "lead_days"], grid={"C": [0.1, 1.0]})
    assert len(seen_folds) == 2 and all(f is folds for f in seen_folds)  # one call per grid point, same folds


def test_target_encoding_of_neighbourhood_is_fitted_on_the_training_fold_only():
    X_train = pd.DataFrame({"neighbourhood": ["A"] * 100 + ["B"] * 100})
    y_train = np.array([0] * 100 + [1] * 100)  # in training: A never no-shows, B always does
    pipeline = train.build_pipeline("logistic_regression", ["neighbourhood"], params={})
    pipeline.fit(X_train, y_train)
    X_valid = pd.DataFrame({"neighbourhood": ["A", "B", "C"]})  # C was never seen in training
    encoded = np.asarray(pipeline.named_steps["prep"].transform(X_valid)).ravel()
    a, b, c = encoded
    # The encoding comes from training labels only: A low, B high, and the unseen C falls back to the training
    # average, strictly between them. If validation rows had been used to fit it, C could not sit in the middle.
    assert a < c < b


def test_no_preprocessing_object_is_fitted_before_cross_validation_starts():
    pipeline = train.build_pipeline("hist_gradient_boosting", ["age", "neighbourhood", "lead_days"], params={})
    prep = pipeline.named_steps["prep"]
    assert not hasattr(prep, "transformers_")  # fitting happens inside .fit(), per fold


# ---- 5. shuffled labels score an AUC near 0.5 -------------------------------------------------------
@pytest.mark.parametrize("model", ["logistic_regression", "hist_gradient_boosting"])
def test_a_model_trained_on_shuffled_labels_scores_an_auc_near_half(prepared, model):
    """One shuffle on this small holdout (about 1,600 rows) can land 0.05 from 0.5 by chance alone: the standard
    error is 0.02 and a model that overfits noise spreads it to about 0.03. The mean over several shuffles is the
    meaningful check."""
    columns = features.all_columns("kaggle")
    real = train.fit_final(model, prepared.X_train, prepared.y_train, columns, params={})
    auc_real = roc_auc_score(prepared.y_holdout, real.predict_proba(prepared.X_holdout[columns])[:, 1])
    assert auc_real > 0.62, auc_real  # the control is meaningful: the real data does carry signal

    shuffled_aucs = []
    for seed in range(6):
        shuffled = train.fit_final(model, prepared.X_train, np.random.default_rng(seed).permutation(prepared.y_train),
                                   columns, params={})
        shuffled_aucs.append(roc_auc_score(prepared.y_holdout, shuffled.predict_proba(prepared.X_holdout[columns])[:, 1]))
    assert 0.47 < np.mean(shuffled_aucs) < 0.53, shuffled_aucs


@pytest.mark.skipif(not data.KAGGLE_PATH.exists(), reason="raw Kaggle file not in data/raw (see data/README.md)")
def test_on_the_real_kaggle_data_shuffled_labels_also_score_near_half():
    prep = experiment.prepare_kaggle()
    columns = features.all_columns("kaggle")
    model = train.fit_final("hist_gradient_boosting", prep.X_train, np.random.default_rng(0).permutation(prep.y_train),
                            columns, params={})
    auc = roc_auc_score(prep.y_holdout, model.predict_proba(prep.X_holdout[columns])[:, 1])
    assert 0.48 < auc < 0.52, auc


# ---- the stop sign ----------------------------------------------------------------------------------
def test_an_auc_above_085_stops_the_run_with_a_leakage_warning():
    evaluate.check_auc_plausible(0.80, "holdout")  # fine
    evaluate.check_auc_plausible(0.85, "holdout")  # the limit itself is allowed
    with pytest.raises(evaluate.LeakageSuspected, match="leakage"):
        evaluate.check_auc_plausible(0.851, "validation of random_forest")
    with pytest.raises(evaluate.LeakageSuspected):
        evaluate.check_auc_plausible(0.97, "x")
    evaluate.check_auc_plausible(float("nan"), "empty fold")  # a NaN is not a leak
