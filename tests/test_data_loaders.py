"""Loaders and cleaning rules for the three public datasets, on small hand-made files.

In ALL code the positive class is no_show = 1 (CLAUDE.md, "Target coding trap"). These tests build the
same appointments in Kaggle format and OpenML format and require both loaders to agree.
"""
from __future__ import annotations

import csv
import itertools
from pathlib import Path

import pandas as pd
import pytest

from ml.src import data

ROOT = Path(__file__).resolve().parents[1]

# ---- hand-made files -----------------------------------------------------------------------------
KAGGLE_COLUMNS = ["PatientId", "AppointmentID", "Gender", "ScheduledDay", "AppointmentDay", "Age",
                  "Neighbourhood", "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap",
                  "SMS_received", "No-show"]
_ids = itertools.count(5_000_000)


def k_row(**overrides) -> dict:
    row = {"PatientId": "29872499824296", "AppointmentID": next(_ids), "Gender": "F",
           "ScheduledDay": "2016-04-29T18:38:08Z", "AppointmentDay": "2016-04-29T00:00:00Z", "Age": 62,
           "Neighbourhood": "JARDIM DA PENHA", "Scholarship": 0, "Hipertension": 1, "Diabetes": 0,
           "Alcoholism": 0, "Handcap": 0, "SMS_received": 0, "No-show": "No"}
    row.update(overrides)
    return row


def write_kaggle(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=KAGGLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


OPENML_ATTRIBUTES = ["especialidad", "edad", "sexo", "reserva_mes_d", "reserva_mes_c", "reserva_dia_d",
                     "reserva_dia_c", "reserva_hora_d", "reserva_hora_c", "creacion_mes_d", "creacion_mes_c",
                     "creacion_dia_d", "creacion_dia_c", "creacion_hora_d", "creacion_hora_c", "latencia",
                     "canal", "tipo", "show"]


def o_row(**overrides) -> dict:
    row = {"especialidad": 76, "edad": 40, "sexo": 2, "reserva_mes_d": 1, "reserva_mes_c": 0.87,
           "reserva_dia_d": 3, "reserva_dia_c": 0.0, "reserva_hora_d": 10, "reserva_hora_c": -0.87,
           "creacion_mes_d": 12, "creacion_mes_c": 1.0, "creacion_dia_d": 2, "creacion_dia_c": -0.22,
           "creacion_hora_d": 9, "creacion_hora_c": -0.71, "latencia": 10, "canal": 1, "tipo": 1, "show": 1}
    row.update(overrides)
    return row


def write_openml(path: Path, rows: list[dict]) -> Path:
    lines = ["% hand-made test file", "@RELATION Medical-Appointment", ""]
    lines += [f"@ATTRIBUTE {name} {'REAL' if name.endswith('_c') else 'INTEGER'}" for name in OPENML_ATTRIBUTES]
    lines += ["", "@DATA"]
    lines += [",".join(str(float(row[name])) for name in OPENML_ATTRIBUTES) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


HANGU_COLUMNS = ["ID", "Session", "Month", "DayOfWeek", "WorkingDay", "AM_PM", "Visit.No", "Gender",
                 "M.Cancer", "S.Cancer", "StartTime", "PayTime", "Address", "ServTime"]


def h_row(**overrides) -> dict:
    row = {"ID": "HAA052B7CD", "Session": 1, "Month": "January", "DayOfWeek": "Wednesday", "WorkingDay": "TRUE",
           "AM_PM": "morning", "Visit.No": 7, "Gender": "F", "M.Cancer": "TRUE", "S.Cancer": "FALSE",
           "StartTime": "8:31:40", "PayTime": "8:44:28", "Address": "Out of city", "ServTime": 691}
    row.update(overrides)
    return row


def write_hangu(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HANGU_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---- target coding: the trap ---------------------------------------------------------------------
# (age, sex, missed?) for five hand-made appointments, written once in each dataset's own convention
APPOINTMENTS = [(34, "F", True), (61, "M", False), (7, "F", False), (45, "M", True), (29, "F", False)]


def test_both_loaders_give_no_show_1_for_the_same_hand_made_rows(tmp_path):
    kaggle = write_kaggle(tmp_path / "k.csv", [
        k_row(Age=age, Gender=sex, **{"No-show": "Yes" if missed else "No"}) for age, sex, missed in APPOINTMENTS])
    openml = write_openml(tmp_path / "o.arff", [
        o_row(edad=age, sexo=1 if sex == "M" else 2, show=0 if missed else 1) for age, sex, missed in APPOINTMENTS])

    expected = [1 if missed else 0 for _, _, missed in APPOINTMENTS]
    assert list(data.load_kaggle(kaggle)["no_show"]) == expected  # Kaggle: "Yes" (missed) -> 1
    assert list(data.load_openml(openml)["no_show"]) == expected  # OpenML: show=0 (missed) -> 1 = 1 - show


def test_the_polarity_is_not_inverted_in_either_loader(tmp_path):
    """Spell out the two directions separately, so a swap cannot hide behind a symmetric fixture."""
    kaggle = data.load_kaggle(write_kaggle(tmp_path / "k.csv", [k_row(**{"No-show": "Yes"}), k_row(**{"No-show": "No"})]))
    assert kaggle.loc[0, "no_show"] == 1 and kaggle.loc[1, "no_show"] == 0
    openml = data.load_openml(write_openml(tmp_path / "o.arff", [o_row(show=0), o_row(show=1)]))
    assert openml.loc[0, "no_show"] == 1 and openml.loc[1, "no_show"] == 0
    assert str(kaggle["no_show"].dtype) == str(openml["no_show"].dtype) == "int8"


def test_loaders_drop_the_original_outcome_columns_so_they_cannot_be_used_by_mistake(tmp_path):
    kaggle = data.load_kaggle(write_kaggle(tmp_path / "k.csv", [k_row()]))
    openml = data.load_openml(write_openml(tmp_path / "o.arff", [o_row()]))
    assert "No-show" not in kaggle.columns and "show" not in openml.columns


def test_an_unknown_outcome_label_is_an_error_not_a_silent_zero(tmp_path):
    with pytest.raises(ValueError, match="No-show"):
        data.load_kaggle(write_kaggle(tmp_path / "k.csv", [k_row(**{"No-show": "Maybe"})]))
    with pytest.raises(ValueError, match="show"):
        data.load_openml(write_openml(tmp_path / "o.arff", [o_row(show=2)]))


def test_a_file_with_missing_columns_is_reported(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("PatientId,Age\n1,2\n")
    with pytest.raises(ValueError, match="missing columns"):
        data.load_kaggle(bad)


def test_missing_raw_files_point_to_the_readme(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        data.load_kaggle(tmp_path / "nope.csv")


def test_kaggle_timestamps_keep_their_written_clock_time(tmp_path):
    frame = data.load_kaggle(write_kaggle(tmp_path / "k.csv", [k_row(ScheduledDay="2016-04-29T18:38:08Z")]))
    assert frame.loc[0, "ScheduledDay"] == pd.Timestamp("2016-04-29 18:38:08")  # naive: no timezone shift
    assert frame["ScheduledDay"].dt.tz is None


# ---- real files (skipped if the raw data has not been downloaded; the files are gitignored) ------
needs_kaggle = pytest.mark.skipif(not data.KAGGLE_PATH.exists(), reason="raw Kaggle file not in data/raw (see data/README.md)")
needs_openml = pytest.mark.skipif(not data.OPENML_PATH.exists(), reason="raw OpenML file not in data/raw (see data/README.md)")
needs_hangu = pytest.mark.skipif(not data.HANGU_PATH.exists(), reason="raw Hangu file not in data/raw (see data/README.md)")


@needs_kaggle
def test_real_kaggle_polarity_matches_an_independent_count_of_yes():
    with open(data.KAGGLE_PATH, newline="") as handle:
        yes = sum(1 for row in csv.DictReader(handle) if row["No-show"] == "Yes")
    frame = data.load_kaggle()
    assert int(frame["no_show"].sum()) == yes > 0
    assert 0.15 < frame["no_show"].mean() < 0.25  # about 20% no-shows, as CLAUDE.md says


@needs_openml
def test_real_openml_polarity_matches_an_independent_count_of_show_0():
    last = [line.rsplit(",", 1)[1].strip() for line in data.OPENML_PATH.read_text().splitlines()
            if line and not line.startswith(("%", "@"))]
    missed = sum(1 for value in last if float(value) == 0.0)
    frame = data.load_openml()
    assert len(frame) == len(last)
    assert int(frame["no_show"].sum()) == missed > 0
    assert 0.15 < frame["no_show"].mean() < 0.30


@needs_hangu
def test_real_hangu_service_time_is_in_seconds():
    frame = data.load_hangu()
    assert 300 < frame["ServTime"].median() < 1500  # about 12 minutes, expressed in seconds


# ---- Kaggle cleaning -----------------------------------------------------------------------------
def clean_kaggle(tmp_path, rows):
    log = data.CleaningLog()
    return data.clean_kaggle(data.load_kaggle(write_kaggle(tmp_path / "k.csv", rows)), log), log


def changed(log, rule, dataset="kaggle") -> int:
    return int(log.to_frame().query("dataset == @dataset and rule == @rule")["rows_changed"].iloc[0])


def test_kaggle_column_typos_are_fixed(tmp_path):
    frame, log = clean_kaggle(tmp_path, [k_row()])
    assert {"hypertension", "handicap", "sex", "no_show", "patient_id", "appointment_date"} <= set(frame.columns)
    assert not {"Hipertension", "Handcap", "PatientId", "No-show"} & set(frame.columns)
    assert int(log.to_frame().query("rule == 'rename_columns'")["columns_changed"].iloc[0]) == 13


def test_float_patient_ids_become_integers_and_malformed_ones_are_dropped(tmp_path):
    frame, log = clean_kaggle(tmp_path, [
        k_row(PatientId="29872499824296"), k_row(PatientId="999981631772427"),
        k_row(PatientId="93779.52927"),  # a fractional id: cannot be cast without corrupting it
    ])
    assert str(frame["patient_id"].dtype) == "int64"
    assert sorted(frame["patient_id"]) == [29872499824296, 999981631772427]  # exact, no float rounding
    assert changed(log, "drop_non_integer_patient_id") == 1


def test_negative_and_impossible_ages_are_dropped_but_real_extremes_are_kept(tmp_path):
    frame, log = clean_kaggle(tmp_path, [k_row(Age=-1), k_row(Age=0), k_row(Age=110), k_row(Age=115), k_row(Age=102)])
    assert sorted(frame["age"]) == [0, 102, 110]  # newborns and a 102-year-old are plausible
    assert changed(log, "drop_impossible_age") == 2


def test_appointment_before_scheduled_is_compared_by_date_not_timestamp(tmp_path):
    frame, log = clean_kaggle(tmp_path, [
        # same calendar day: AppointmentDay is stored at 00:00 while ScheduledDay has a time, still valid
        k_row(ScheduledDay="2016-04-29T18:38:08Z", AppointmentDay="2016-04-29T00:00:00Z"),
        k_row(ScheduledDay="2016-05-10T10:51:53Z", AppointmentDay="2016-05-09T00:00:00Z"),  # day before: invalid
        k_row(ScheduledDay="2016-04-20T08:00:00Z", AppointmentDay="2016-04-29T00:00:00Z"),
    ])
    assert len(frame) == 2
    assert changed(log, "drop_appointment_before_scheduled") == 1


def test_lead_time_is_whole_days_ignoring_time_of_day(tmp_path):
    frame, _ = clean_kaggle(tmp_path, [
        k_row(ScheduledDay="2016-04-29T23:59:00Z", AppointmentDay="2016-05-01T00:00:00Z"),  # 2 days, not 1.0007
        k_row(ScheduledDay="2016-04-29T00:01:00Z", AppointmentDay="2016-04-29T00:00:00Z"),  # same day: 0
        k_row(ScheduledDay="2016-04-01T12:00:00Z", AppointmentDay="2016-05-02T00:00:00Z"),  # 31 days
    ])
    assert sorted(frame["lead_days"]) == [0, 2, 31]
    assert str(frame["lead_days"].dtype) == "int64"
    assert set(frame["appt_weekday"]) <= set(range(7))
    assert frame.loc[frame["lead_days"] == 2, "appt_weekday"].iloc[0] == 6  # 2016-05-01 was a Sunday


def test_exact_duplicates_are_dropped_but_rows_differing_only_by_appointment_id_are_kept_and_reported(tmp_path):
    same = k_row(AppointmentID=777)
    frame, log = clean_kaggle(tmp_path, [same, dict(same), k_row(AppointmentID=1), k_row(AppointmentID=2)])
    assert changed(log, "drop_exact_duplicates") == 1
    assert len(frame) == 3
    row = log.to_frame().query("rule == 'report_duplicates_ignoring_appointment_id'").iloc[0]
    assert row["kind"] == "report" and row["rows_changed"] == 2 and row["rows_before"] == row["rows_after"] == 3


def test_cleaning_keeps_the_target_and_logs_a_consistent_chain(tmp_path):
    rows = [k_row(**{"No-show": "Yes"}), k_row(Age=-1), k_row(PatientId="1.5"), k_row(**{"No-show": "No"})]
    frame, log = clean_kaggle(tmp_path, rows)
    assert sorted(frame["no_show"]) == [0, 1]
    table = log.to_frame()
    assert list(table["step"]) == list(range(1, len(table) + 1))
    assert table["rows_before"].iloc[1:].tolist() == table["rows_after"].iloc[:-1].tolist()  # each step starts where the last ended
    assert table["rows_after"].iloc[-1] == len(frame)
    dropped = table.query("kind == 'drop'")
    assert (dropped["rows_before"] - dropped["rows_after"] == dropped["rows_changed"]).all()
    assert table.iloc[0]["rule"] == "load" and "no_show = 1" in table.iloc[0]["note"]


def test_cleaning_does_not_modify_its_input(tmp_path):
    raw = data.load_kaggle(write_kaggle(tmp_path / "k.csv", [k_row(Age=-1), k_row()]))
    before = raw.copy()
    data.clean_kaggle(raw, data.CleaningLog())
    pd.testing.assert_frame_equal(raw, before)


def test_the_cleaning_log_is_written_as_csv(tmp_path):
    frame, log = clean_kaggle(tmp_path, [k_row(Age=-1), k_row()])
    out = log.write(tmp_path / "out" / "cleaning_log.csv")
    table = pd.read_csv(out)
    assert list(table.columns) == ["dataset", "step", "rule", "kind", "rows_before", "rows_after",
                                   "rows_changed", "columns_changed", "note"]
    assert (table["dataset"] == "kaggle").all()


# ---- OpenML cleaning -----------------------------------------------------------------------------
def clean_openml(tmp_path, rows):
    log = data.CleaningLog()
    return data.clean_openml(data.load_openml(write_openml(tmp_path / "o.arff", rows)), log), log


def test_openml_columns_are_renamed_to_english_and_cosine_duplicates_dropped(tmp_path):
    frame, log = clean_openml(tmp_path, [o_row()])
    assert list(frame.columns) == ["specialty", "age", "sex", "appt_month", "appt_weekday", "appt_hour",
                                   "booked_month", "booked_weekday", "booked_hour", "lead_days", "channel",
                                   "appt_type", "no_show"]
    assert int(log.to_frame().query("rule == 'drop_cosine_columns'")["columns_changed"].iloc[0]) == 6
    assert all(str(frame[c].dtype) == "int64" for c in frame.columns if c not in {"sex", "no_show"})
    assert str(frame["no_show"].dtype) == "int8"


def test_openml_sex_and_weekday_use_the_same_coding_as_kaggle(tmp_path):
    frame, _ = clean_openml(tmp_path, [o_row(sexo=1, reserva_dia_d=1), o_row(sexo=2, reserva_dia_d=7)])
    assert list(frame["sex"]) == ["M", "F"]  # sexo: 1 = male, 2 = female
    assert list(frame["appt_weekday"]) == [0, 6]  # 1 = Monday ... 7 = Sunday  ->  0 = Monday ... 6 = Sunday


def test_openml_negative_lead_times_and_impossible_ages_are_dropped(tmp_path):
    frame, log = clean_openml(tmp_path, [o_row(latencia=-3), o_row(latencia=0), o_row(edad=-4), o_row(edad=130), o_row()])
    assert len(frame) == 2
    assert changed(log, "drop_negative_lead_days", "openml") == 1
    assert changed(log, "drop_impossible_age", "openml") == 2


def test_openml_exact_duplicates_are_dropped_and_logged(tmp_path):
    a, b = o_row(edad=30), o_row(edad=31)
    frame, log = clean_openml(tmp_path, [a, dict(a), dict(a), b])
    assert len(frame) == 2
    assert changed(log, "drop_exact_duplicates", "openml") == 2
    assert "patient" in log.to_frame().query("rule == 'drop_exact_duplicates'")["note"].iloc[0]  # explains the caveat


def test_openml_unlikely_hours_are_reported_not_dropped(tmp_path):
    frame, log = clean_openml(tmp_path, [o_row(reserva_hora_d=0), o_row(reserva_hora_d=9, edad=50)])
    assert len(frame) == 2
    row = log.to_frame().query("rule == 'report_hours_outside_clinic_hours'").iloc[0]
    assert row["kind"] == "report" and row["rows_changed"] == 1


# ---- Hangu cleaning ------------------------------------------------------------------------------
def clean_hangu(tmp_path, rows):
    log = data.CleaningLog()
    return data.clean_hangu(data.load_hangu(write_hangu(tmp_path / "h.csv", rows)), log), log


def test_hangu_service_time_is_converted_from_seconds_to_minutes(tmp_path):
    frame, _ = clean_hangu(tmp_path, [h_row(ServTime=720, StartTime="9:00:00"), h_row(ServTime=90, StartTime="9:20:00")])
    assert list(frame["service_seconds"]) == [720, 90]
    assert list(frame["service_minutes"]) == [12.0, 1.5]


def test_hangu_clear_errors_are_dropped_but_the_real_long_tail_is_kept(tmp_path):
    frame, log = clean_hangu(tmp_path, [
        h_row(ServTime=0, StartTime="8:00:00"), h_row(ServTime=-5, StartTime="8:01:00"),
        h_row(ServTime="", StartTime="8:02:00"),  # missing
        h_row(ServTime=3457, StartTime="8:03:00"),  # a long consultation (58 minutes) is real: keep
        h_row(ServTime=23423, StartTime="8:04:00"),  # 6.5 hours is not a consultation: drop
        h_row(ServTime=691, StartTime="8:05:00"),
    ])
    assert sorted(frame["service_seconds"]) == [691, 3457]
    assert changed(log, "drop_missing_or_nonpositive_service_time", "hangu") == 3
    assert changed(log, "drop_service_time_above_limit", "hangu") == 1


def test_hangu_exact_duplicates_are_dropped(tmp_path):
    frame, log = clean_hangu(tmp_path, [h_row(), h_row(), h_row(StartTime="9:30:00")])
    assert len(frame) == 2 and changed(log, "drop_exact_duplicates", "hangu") == 1


def test_all_three_cleaners_log_into_one_shared_table(tmp_path):
    log = data.CleaningLog()
    data.clean_kaggle(data.load_kaggle(write_kaggle(tmp_path / "k.csv", [k_row()])), log)
    data.clean_openml(data.load_openml(write_openml(tmp_path / "o.arff", [o_row()])), log)
    data.clean_hangu(data.load_hangu(write_hangu(tmp_path / "h.csv", [h_row()])), log)
    table = log.to_frame()
    assert set(table["dataset"]) == {"kaggle", "openml", "hangu"}
    for _, steps in table.groupby("dataset"):
        assert list(steps["step"]) == list(range(1, len(steps) + 1))
