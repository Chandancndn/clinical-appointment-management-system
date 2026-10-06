"""Feature groups, kinds and the Kaggle / OpenML feature builders."""
from __future__ import annotations

from helpers_ml import synthetic_kaggle, synthetic_openml
from ml.src import features

KAGGLE_GROUPS = {"history", "lead_calendar", "demographics", "flags", "neighbourhood"}


def test_kaggle_has_the_five_documented_feature_groups():
    assert set(features.KAGGLE_GROUPS) == KAGGLE_GROUPS
    assert features.KAGGLE_GROUPS["history"] == ["prev_appointments", "prev_no_show_rate", "has_history"]
    assert set(features.KAGGLE_GROUPS["flags"]) >= {"sms_received", "hypertension", "diabetes", "alcoholism", "handicap", "scholarship"}
    assert features.KAGGLE_GROUPS["neighbourhood"] == ["neighbourhood"]


def test_every_feature_column_has_exactly_one_kind_and_each_column_is_in_one_group():
    for dataset, groups in (("kaggle", features.KAGGLE_GROUPS), ("openml", features.OPENML_GROUPS)):
        columns = [c for group in groups.values() for c in group]
        assert len(columns) == len(set(columns))
        assert set(columns) == set(features.all_columns(dataset))
        assert all(features.COLUMN_KIND[c] in {"numeric", "binary", "category_small", "target_encoded"} for c in columns)


def test_the_kaggle_builder_keeps_the_index_and_encodes_sex_and_same_day_bookings():
    frame = features.add_history(synthetic_kaggle(n=500, seed=1))
    matrix = features.kaggle_features(frame)
    assert matrix.index.equals(frame.index)
    assert (matrix["is_female"] == (frame["sex"] == "F").astype(int)).all()
    assert (matrix["lead_same_day"] == (frame["lead_days"] == 0).astype(int)).all()
    assert matrix["prev_no_show_rate"].isna().sum() == (matrix["has_history"] == 0).sum()  # missing exactly when no history
    assert not matrix.drop(columns="prev_no_show_rate").isna().any().any()


def test_the_openml_builder_leaves_out_calendar_position_columns():
    matrix = features.openml_features(synthetic_openml(n=300))
    assert not {"appt_month", "booked_month"} & set(matrix.columns)  # months identify time, and the holdout is the last month
    assert {"specialty", "channel", "appt_hour", "booked_hour", "lead_days"} <= set(matrix.columns)
    assert not {"no_show"} & set(matrix.columns)
    assert set(matrix.columns) == set(features.all_columns("openml"))


def test_the_common_features_mean_the_same_thing_in_both_datasets():
    k = features.kaggle_features(features.add_history(synthetic_kaggle(n=300)))
    o = features.openml_features(synthetic_openml(n=300))
    assert features.COMMON_COLUMNS == ["age", "is_female", "lead_days", "lead_same_day", "appt_weekday"]
    for matrix in (k, o):
        assert set(features.COMMON_COLUMNS) <= set(matrix.columns)
        assert matrix["is_female"].isin([0, 1]).all() and matrix["appt_weekday"].between(0, 6).all()
        assert (matrix["lead_same_day"] == (matrix["lead_days"] == 0)).all()


def test_columns_for_groups_keeps_a_stable_order_and_rejects_unknown_groups():
    cols = features.columns_for("kaggle", ["demographics", "history"])
    assert cols == features.KAGGLE_GROUPS["history"] + features.KAGGLE_GROUPS["demographics"]  # group order, not argument order
    import pytest
    with pytest.raises(KeyError):
        features.columns_for("kaggle", ["astrology"])


def test_a_feature_spec_sorts_columns_by_how_they_are_preprocessed():
    spec = features.feature_spec(["age", "is_female", "appt_weekday", "neighbourhood", "prev_no_show_rate"])
    assert spec.numeric == ["age", "prev_no_show_rate"]
    assert spec.binary == ["is_female"]
    assert spec.category_small == ["appt_weekday"]
    assert spec.target_encoded == ["neighbourhood"]
