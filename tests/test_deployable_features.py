"""The ONE deployable feature function: the training path and the serving path must give identical numbers.

Deployable features (CLAUDE.md): age, sex, lead time in days, weekday, number of earlier appointments and the
earlier no-show rate, where history uses only appointments whose outcome was known when the booking was made.
The function takes plain values and a history list (not a DataFrame row) so the Flask app can import it unchanged.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from helpers_ml import synthetic_kaggle
from ml.src import features

SIX = ["age", "is_female", "lead_days", "appt_weekday", "prev_appointments", "prev_no_show_rate"]


def same(a: dict, b) -> bool:
    return all((x == y) or (math.isnan(x) and math.isnan(y)) for x, y in zip(a.values(), b))


# ---- requirement 2: one booking, two paths, identical numbers ------------------------------------------
def test_the_deployable_features_are_exactly_the_six_documented_ones():
    assert features.DEPLOYABLE_FEATURES == SIX
    assert set(SIX) <= set(features.COLUMN_KIND)  # every one has a defined preprocessing kind


def test_one_booking_gives_identical_features_through_the_training_and_serving_paths():
    dob, appointment, booked = date(1990, 6, 15), date(2016, 5, 12), date(2016, 5, 5)  # a Thursday, 7 days ahead
    history = [
        (date(2016, 4, 30), 1),  # a no-show a week before the booking: usable
        (date(2016, 5, 2), 0),   # attended: usable
        (date(2016, 5, 5), 1),   # ON the booking date: its outcome was not known yet, so not usable
        (date(2016, 5, 9), 1),   # after the booking, before the appointment: not usable either
    ]

    # SERVING path: what the Flask app will do, from plain values read out of the database
    serving = features.deployable_features(
        age=features.age_in_years(dob, appointment), sex="F", booked_on=booked, appointment_date=appointment,
        history=history)

    # TRAINING path: the same booking and its history as rows of a cleaned Kaggle-format frame
    def row(appt, booked_on, no_show, age=25):
        return {"patient_id": 7, "appointment_date": pd.Timestamp(appt), "scheduled_at": pd.Timestamp(booked_on) + pd.Timedelta(hours=10),
                "no_show": no_show, "age": age, "sex": "F"}

    frame = pd.DataFrame([row(d, d - pd.Timedelta(days=3), o) for d, o in history] + [row(appointment, booked, 0)])
    training = features.deployable_matrix(frame).iloc[-1]

    assert list(training.index) == features.DEPLOYABLE_FEATURES == list(serving)
    assert same(serving, training.to_numpy(dtype=float))
    # and both equal the hand-worked answer
    assert serving == {"age": 25, "is_female": 1, "lead_days": 7, "appt_weekday": 3,
                       "prev_appointments": 2, "prev_no_show_rate": 0.5}


def test_the_serving_function_matches_the_training_matrix_on_every_row_of_a_realistic_frame():
    frame = synthetic_kaggle(n=600, n_patients=120, seed=31)
    matrix = features.deployable_matrix(frame)
    by_patient = {p: list(zip(g["appointment_date"].dt.date, g["no_show"])) for p, g in frame.groupby("patient_id")}
    assert (matrix["prev_appointments"] > 0).sum() > 100  # the comparison is not trivially "no history"
    for position in range(len(frame)):
        r = frame.iloc[position]
        served = features.deployable_features(
            age=int(r["age"]), sex=r["sex"], booked_on=r["scheduled_at"].date(),
            appointment_date=r["appointment_date"].date(), history=by_patient[r["patient_id"]])
        assert same(served, matrix.iloc[position].to_numpy(dtype=float)), position


def test_the_training_matrix_also_agrees_with_the_vectorised_history_used_in_m4():
    """Two independent implementations of the strict history rule must agree on every row."""
    frame = synthetic_kaggle(n=800, n_patients=150, seed=32)
    function_path = features.deployable_matrix(frame)
    research_path = features.kaggle_features(features.add_history(frame))[features.DEPLOYABLE_FEATURES]
    pd.testing.assert_frame_equal(function_path.reset_index(drop=True), research_path.reset_index(drop=True),
                                  check_dtype=False)


# ---- the strict history rule inside the function --------------------------------------------------------
def booking(**overrides):
    base = dict(age=30, sex="M", booked_on=date(2016, 5, 10), appointment_date=date(2016, 5, 12), history=[])
    base.update(overrides)
    return features.deployable_features(**base)


def test_only_outcomes_dated_before_the_booking_date_count():
    got = booking(history=[(date(2016, 5, 9), 1), (date(2016, 5, 10), 1), (date(2016, 5, 11), 1), (date(2016, 5, 12), 0)])
    assert got["prev_appointments"] == 1 and got["prev_no_show_rate"] == 1.0


def test_history_order_and_the_bookings_own_row_do_not_matter():
    history = [(date(2016, 4, 1), 1), (date(2016, 4, 5), 0), (date(2016, 4, 9), 0)]
    a = booking(history=history)
    b = booking(history=list(reversed(history)))
    c = booking(history=history + [(date(2016, 5, 12), 1)])  # the booking itself, appended by a careless caller
    assert a == b == c and a["prev_appointments"] == 3 and a["prev_no_show_rate"] == pytest.approx(1 / 3)


def test_no_history_means_zero_earlier_appointments_and_a_missing_rate():
    got = booking()
    assert got["prev_appointments"] == 0 and math.isnan(got["prev_no_show_rate"])


def test_outcomes_can_be_booleans_or_integers():
    assert booking(history=[(date(2016, 4, 1), True), (date(2016, 4, 2), False)])["prev_no_show_rate"] == 0.5


def test_lead_time_weekday_and_sex_are_coded_like_the_training_data():
    got = booking(sex="F", booked_on=date(2016, 5, 12), appointment_date=date(2016, 5, 12))  # same-day booking, a Thursday
    assert got["lead_days"] == 0 and got["appt_weekday"] == 3 and got["is_female"] == 1
    assert booking(sex="M")["is_female"] == 0
    assert booking(appointment_date=date(2016, 5, 16))["appt_weekday"] == 0  # Monday is 0, as in Kaggle's dayofweek
    assert booking(booked_on=date(2016, 5, 1), appointment_date=date(2016, 5, 31))["lead_days"] == 30


@pytest.mark.parametrize("overrides", [
    {"sex": "X"}, {"sex": "female"}, {"age": -1}, {"age": 130},
    {"booked_on": date(2016, 5, 13), "appointment_date": date(2016, 5, 12)},  # booked after the appointment
])
def test_impossible_inputs_are_refused_rather_than_guessed(overrides):
    with pytest.raises(ValueError):
        booking(**overrides)


def test_age_in_years_counts_completed_years_on_the_appointment_date():
    assert features.age_in_years(date(1990, 6, 15), date(2016, 6, 14)) == 25  # the day before the birthday
    assert features.age_in_years(date(1990, 6, 15), date(2016, 6, 15)) == 26  # on it
    assert features.age_in_years(date(2000, 2, 29), date(2001, 2, 28)) == 0
    assert features.age_in_years(date(2000, 2, 29), date(2001, 3, 1)) == 1
    assert features.age_in_years(date(2016, 5, 12), date(2016, 5, 12)) == 0
    with pytest.raises(ValueError):
        features.age_in_years(date(2016, 5, 13), date(2016, 5, 12))  # born after the appointment


def test_the_function_needs_no_dataframe_and_its_output_is_a_plain_ordered_dict():
    got = booking(history=[(date(2016, 4, 1), 1)])
    assert isinstance(got, dict) and list(got) == features.DEPLOYABLE_FEATURES
    assert all(isinstance(v, (int, float)) for v in got.values())
    row = pd.DataFrame([got])[features.DEPLOYABLE_FEATURES]  # what the app will hand to the model
    assert row.shape == (1, 6) and not np.isnan(row.iloc[0, :5].to_numpy(dtype=float)).any()
