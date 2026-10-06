"""Splits: a locked time-based holdout, patient-grouped folds (Kaggle) and stratified folds (OpenML)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.src import data


def appointments(n_dates=30, per_date=20, n_patients=60, seed=0, skip_weekends=True) -> pd.DataFrame:
    """Shuffled synthetic appointments with repeat patients and (like the real data) no weekend dates."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-04-29", periods=n_dates) if skip_weekends else pd.date_range("2016-04-29", periods=n_dates)
    frame = pd.DataFrame({
        "appointment_date": np.repeat(dates.values, per_date),
        "patient_id": rng.integers(1000, 1000 + n_patients, n_dates * per_date),
        "no_show": (rng.random(n_dates * per_date) < 0.2).astype("int8"),
    })
    return frame.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---- time-based holdout --------------------------------------------------------------------------
def test_holdout_dates_are_all_later_than_training_dates():
    split = data.split_holdout_by_date(appointments())
    assert split.holdout["appointment_date"].min() > split.train["appointment_date"].max()


def test_the_cut_is_at_a_date_boundary_never_mid_day():
    split = data.split_holdout_by_date(appointments())
    train_dates, holdout_dates = set(split.train["appointment_date"]), set(split.holdout["appointment_date"])
    assert not train_dates & holdout_dates  # no calendar day is shared
    assert split.cut == split.holdout["appointment_date"].min()


def test_the_holdout_is_the_latest_20_percent_of_appointment_dates():
    frame = appointments(n_dates=30)
    split = data.split_holdout_by_date(frame)
    assert split.holdout["appointment_date"].nunique() == 6  # round(0.2 * 30)
    assert len(split.train) + len(split.holdout) == len(frame)
    assert set(split.train.index) | set(split.holdout.index) == set(frame.index)  # nothing lost, nothing twice
    assert not set(split.train.index) & set(split.holdout.index)


def test_uneven_rows_per_date_still_cut_by_whole_dates():
    frame = appointments(n_dates=10, per_date=10)
    frame = pd.concat([frame, frame[frame["appointment_date"] == frame["appointment_date"].max()].head(5)])  # last day busier
    split = data.split_holdout_by_date(frame, fraction=0.2)
    assert split.holdout["appointment_date"].nunique() == 2
    assert len(split.holdout) == 10 + 15  # the two latest dates, all their rows


def test_input_order_does_not_matter():
    frame = appointments()
    ordered = frame.sort_values("appointment_date")
    a, b = data.split_holdout_by_date(frame), data.split_holdout_by_date(ordered)

    def rows(part):
        return sorted(zip(part["appointment_date"], part["patient_id"], part["no_show"]))

    assert a.cut == b.cut and rows(a.holdout) == rows(b.holdout) and rows(a.train) == rows(b.train)


def test_at_least_one_date_is_held_out_and_one_is_kept():
    tiny = appointments(n_dates=3, per_date=4)
    split = data.split_holdout_by_date(tiny, fraction=0.01)
    assert split.holdout["appointment_date"].nunique() == 1 and split.train["appointment_date"].nunique() == 2
    with pytest.raises(ValueError):
        data.split_holdout_by_date(appointments(n_dates=1), fraction=0.2)  # nothing left to train on
    for bad in (0, 1, -0.1, 1.5):
        with pytest.raises(ValueError):
            data.split_holdout_by_date(appointments(), fraction=bad)


def test_openml_holdout_is_the_last_month():
    frame = pd.DataFrame({"appt_month": [1, 1, 2, 2, 3, 3, 4, 4, 4], "no_show": 0})
    split = data.split_holdout_last_month(frame)
    assert set(split.holdout["appt_month"]) == {4} == {split.cut}
    assert split.train["appt_month"].max() < split.holdout["appt_month"].min()
    assert len(split.train) + len(split.holdout) == len(frame)


# ---- patient-grouped folds (Kaggle) --------------------------------------------------------------
def test_no_patient_appears_in_both_a_training_fold_and_its_validation_fold():
    frame = appointments(n_patients=80)
    folds = data.grouped_folds(frame["patient_id"], n_splits=5)
    assert len(folds) == 5
    for train_pos, valid_pos in folds:
        train_patients = set(frame["patient_id"].iloc[train_pos])
        valid_patients = set(frame["patient_id"].iloc[valid_pos])
        assert train_patients and valid_patients
        assert not train_patients & valid_patients


def test_grouped_folds_validate_every_row_exactly_once():
    frame = appointments()
    folds = data.grouped_folds(frame["patient_id"], n_splits=5)
    validated = np.concatenate([valid for _, valid in folds])
    assert sorted(validated) == list(range(len(frame)))
    for train_pos, valid_pos in folds:
        assert not set(train_pos) & set(valid_pos)
        assert len(train_pos) + len(valid_pos) == len(frame)


def test_grouped_folds_are_reproducible_and_depend_on_the_seed():
    groups = appointments()["patient_id"]
    same_a, same_b = data.grouped_folds(groups, seed=7), data.grouped_folds(groups, seed=7)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(same_a, same_b))
    other = data.grouped_folds(groups, seed=8)
    assert not all(np.array_equal(a[1], b[1]) for a, b in zip(same_a, other))


def test_folds_built_on_the_training_part_never_touch_the_holdout():
    frame = appointments(n_patients=40)
    split = data.split_holdout_by_date(frame)
    train = split.train.reset_index(drop=True)
    for train_pos, valid_pos in data.grouped_folds(train["patient_id"], n_splits=4):
        assert max(train_pos.max(), valid_pos.max()) < len(train)  # positions index the training part only
        assert train["appointment_date"].iloc[valid_pos].max() < split.cut


def test_too_few_patients_for_the_folds_is_an_error():
    with pytest.raises(ValueError):
        data.grouped_folds(pd.Series([1, 1, 2, 2]), n_splits=5)


# ---- stratified folds (OpenML) -------------------------------------------------------------------
def test_stratified_folds_cover_each_row_once_and_keep_the_no_show_rate():
    rng = np.random.default_rng(1)
    y = pd.Series((rng.random(1000) < 0.2).astype("int8"))
    folds = data.stratified_folds(y, n_splits=5)
    assert sorted(np.concatenate([v for _, v in folds])) == list(range(1000))
    for train_pos, valid_pos in folds:
        assert not set(train_pos) & set(valid_pos)
        assert abs(y.iloc[valid_pos].mean() - y.mean()) < 0.01  # stratified: every fold has about 20%


def test_stratified_folds_are_reproducible():
    y = pd.Series([0, 1] * 50)
    a, b = data.stratified_folds(y, seed=3), data.stratified_folds(y, seed=3)
    assert all(np.array_equal(x[1], z[1]) for x, z in zip(a, b))
