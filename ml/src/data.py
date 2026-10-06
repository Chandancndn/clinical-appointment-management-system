"""Data pipeline: loaders, logged cleaning rules and splits for the three public datasets (CLAUDE.md, PLAN section 3).

* In ALL code the positive class is no_show = 1.
    Kaggle:  column "No-show" is "Yes" for a missed appointment  ->  no_show = 1
    OpenML:  column "show" is 1 for an attended appointment      ->  no_show = 1 - show
  Loaders do that conversion and drop the original outcome column so it cannot be used by mistake.
* Loaders only read and parse. Cleaning functions apply the rules, and every rule writes one row to the
  CleaningLog (results/cleaning_log.csv): rows before, rows after, how many rows it changed.
* Raw files live in data/raw/ (gitignored); see data/README.md. Nothing here ever writes to data/raw/.

    python -m ml.src.data      # print the name, size, row count and SHA-256 of each raw file
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
KAGGLE_PATH = RAW_DIR / "kaggle" / "KaggleV2-May-2016.csv"
OPENML_PATH = RAW_DIR / "openml" / "medical_appointment_43617.arff"
HANGU_PATH = RAW_DIR / "hangu" / "Data.csv"  # the file with ServTime in seconds (6,637 consultations)
RESULTS_DIR = ROOT / "results"
CLEANING_LOG_PATH = RESULTS_DIR / "cleaning_log.csv"

TARGET = "no_show"
DEFAULT_SEED = 42
HOLDOUT_FRACTION = 0.2

# Judgement calls, stated here and written into the cleaning log notes.
MAX_PLAUSIBLE_AGE = 110  # a few Kaggle rows say 115; 102 is kept
MAX_SERVICE_SECONDS = 4 * 3600  # a consultation longer than 4 hours is a recording error
CLINIC_HOURS = (7, 21)  # OpenML appointment hours outside this are reported, not dropped

# ---- expected raw columns -------------------------------------------------------------------------
KAGGLE_COLUMNS = ["PatientId", "AppointmentID", "Gender", "ScheduledDay", "AppointmentDay", "Age", "Neighbourhood",
                  "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap", "SMS_received", "No-show"]
KAGGLE_RENAMES = {  # includes the two typos in the source: Hipertension, Handcap
    "PatientId": "patient_id", "AppointmentID": "appointment_id", "Gender": "sex", "ScheduledDay": "scheduled_at",
    "AppointmentDay": "appointment_date", "Age": "age", "Neighbourhood": "neighbourhood",
    "Scholarship": "scholarship", "Hipertension": "hypertension", "Diabetes": "diabetes",
    "Alcoholism": "alcoholism", "Handcap": "handicap", "SMS_received": "sms_received",
}
KAGGLE_ORDER = ["patient_id", "appointment_id", "sex", "scheduled_at", "appointment_date", "age", "neighbourhood",
                "scholarship", "hypertension", "diabetes", "alcoholism", "handicap", "sms_received",
                "lead_days", "appt_weekday", "no_show"]

OPENML_COLUMNS = ["especialidad", "edad", "sexo", "reserva_mes_d", "reserva_mes_c", "reserva_dia_d", "reserva_dia_c",
                  "reserva_hora_d", "reserva_hora_c", "creacion_mes_d", "creacion_mes_c", "creacion_dia_d",
                  "creacion_dia_c", "creacion_hora_d", "creacion_hora_c", "latencia", "canal", "tipo", "show"]
OPENML_RENAMES = {
    "especialidad": "specialty", "edad": "age", "sexo": "sex", "reserva_mes_d": "appt_month",
    "reserva_dia_d": "appt_weekday", "reserva_hora_d": "appt_hour", "creacion_mes_d": "booked_month",
    "creacion_dia_d": "booked_weekday", "creacion_hora_d": "booked_hour", "latencia": "lead_days",
    "canal": "channel", "tipo": "appt_type",
}  # channel: 1 call centre, 2 personal, 3 web.  appt_type: 1 medical, 2 procedures

HANGU_COLUMNS = ["ID", "Session", "Month", "DayOfWeek", "WorkingDay", "AM_PM", "Visit.No", "Gender", "M.Cancer",
                 "S.Cancer", "StartTime", "PayTime", "Address", "ServTime"]
HANGU_RENAMES = {
    "ID": "patient_code", "Session": "session", "Month": "month", "DayOfWeek": "day_of_week",
    "WorkingDay": "working_day", "AM_PM": "am_pm", "Visit.No": "visit_no", "Gender": "sex", "M.Cancer": "m_cancer",
    "S.Cancer": "s_cancer", "StartTime": "start_time", "PayTime": "pay_time", "Address": "address",
    "ServTime": "service_seconds",
}


# ---- the cleaning log -----------------------------------------------------------------------------
LOG_COLUMNS = ["dataset", "step", "rule", "kind", "rows_before", "rows_after", "rows_changed", "columns_changed", "note"]


class CleaningLog:
    """One row per cleaning rule. kind is load, rename, cast, recode, derive, drop, drop_columns or report.

    rows_changed is the number of rows a rule dropped, recoded or flagged; columns_changed counts columns for
    renames, casts and derived columns. A "report" row only counts rows that were noticed and kept.
    """

    def __init__(self) -> None:
        self._records: list[dict] = []

    def add(self, dataset: str, rule: str, kind: str, rows_before: int, rows_after: int,
            rows_changed: int = 0, columns_changed: int = 0, note: str = "") -> None:
        step = 1 + sum(1 for r in self._records if r["dataset"] == dataset)
        self._records.append({"dataset": dataset, "step": step, "rule": rule, "kind": kind,
                              "rows_before": int(rows_before), "rows_after": int(rows_after),
                              "rows_changed": int(rows_changed), "columns_changed": int(columns_changed), "note": note})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._records, columns=LOG_COLUMNS)

    def write(self, path=CLEANING_LOG_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False)
        return path


def _drop(frame: pd.DataFrame, mask, log: CleaningLog, dataset: str, rule: str, note: str) -> pd.DataFrame:
    kept = frame.loc[~np.asarray(mask)].reset_index(drop=True)
    log.add(dataset, rule, "drop", len(frame), len(kept), len(frame) - len(kept), note=note)
    return kept


def _report(frame: pd.DataFrame, flagged: int, log: CleaningLog, dataset: str, rule: str, note: str) -> None:
    log.add(dataset, rule, "report", len(frame), len(frame), int(flagged), note=note)


# ---- loaders --------------------------------------------------------------------------------------
def _require_file(path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Download the raw data as described in data/README.md.")
    return path


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} file is missing columns: {missing}")


def load_kaggle(path=KAGGLE_PATH) -> pd.DataFrame:
    """Kaggle "Medical Appointment No Shows". Original column names; no_show = 1 where "No-show" is "Yes".

    PatientId is read as text and converted afterwards so large ids are never rounded by the float parser.
    Timestamps keep their written clock time (the trailing "Z" is dropped, not converted).
    """
    frame = pd.read_csv(_require_file(path), dtype={"PatientId": str})
    _require_columns(frame, KAGGLE_COLUMNS, "Kaggle")
    labels = frame["No-show"].map({"Yes": 1, "No": 0})
    if labels.isna().any():
        bad = sorted(frame.loc[labels.isna(), "No-show"].astype(str).unique())
        raise ValueError(f"Kaggle 'No-show' has unexpected values {bad}; expected 'Yes' (missed) or 'No' (attended)")
    frame[TARGET] = labels.astype("int8")
    frame = frame.drop(columns="No-show")
    frame["PatientId"] = pd.to_numeric(frame["PatientId"])
    for column in ("ScheduledDay", "AppointmentDay"):
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_localize(None)
    return frame


def load_openml(path=OPENML_PATH) -> pd.DataFrame:
    """OpenML "Medical-Appointment" (id 43617). Original (Spanish) column names; no_show = 1 - show."""
    records, _ = arff.loadarff(str(_require_file(path)))
    frame = pd.DataFrame(records)
    _require_columns(frame, OPENML_COLUMNS, "OpenML")
    if not frame["show"].isin([0.0, 1.0]).all():
        bad = sorted(frame.loc[~frame["show"].isin([0.0, 1.0]), "show"].unique())
        raise ValueError(f"OpenML 'show' has unexpected values {bad}; expected 1 (attended) or 0 (missed)")
    frame[TARGET] = (1 - frame["show"]).astype("int8")
    return frame.drop(columns="show")


def load_hangu(path=HANGU_PATH) -> pd.DataFrame:
    """Hangu consultation data (Data.csv). ServTime is the consultation length in seconds."""
    frame = pd.read_csv(_require_file(path))
    _require_columns(frame, HANGU_COLUMNS, "Hangu")
    frame["ServTime"] = pd.to_numeric(frame["ServTime"], errors="coerce")
    return frame


# ---- cleaning: Kaggle -----------------------------------------------------------------------------
def clean_kaggle(df: pd.DataFrame, log: Optional[CleaningLog] = None) -> pd.DataFrame:
    """Apply the Kaggle cleaning rules in order, logging each. The input frame is not modified."""
    log = CleaningLog() if log is None else log
    frame = df.copy()
    n = len(frame)
    log.add("kaggle", "load", "load", n, n, note=f"positive class no_show = 1 (No-show = 'Yes'): "
            f"{int(frame[TARGET].sum()):,} rows ({frame[TARGET].mean():.2%})")

    frame = frame.rename(columns=KAGGLE_RENAMES)
    log.add("kaggle", "rename_columns", "rename", n, n, columns_changed=len(KAGGLE_RENAMES),
            note="snake_case names; source typos fixed: Hipertension -> hypertension, Handcap -> handicap")

    ids = frame["patient_id"]
    frame = _drop(frame, (ids % 1) != 0, log, "kaggle", "drop_non_integer_patient_id",
                  "a PatientId with a fractional part is malformed and cannot be cast to an integer without corrupting it")
    if frame["patient_id"].abs().max() >= 2 ** 53:
        raise ValueError("PatientId too large to represent exactly as a float; the loader would have rounded it")
    frame["patient_id"] = frame["patient_id"].astype("int64")
    log.add("kaggle", "cast_patient_id_to_int", "cast", len(frame), len(frame), columns_changed=1,
            note="float64 -> int64, lossless (all remaining ids are whole numbers below 2**53)")

    frame = _drop(frame, (frame["age"] < 0) | (frame["age"] > MAX_PLAUSIBLE_AGE), log, "kaggle",
                  "drop_impossible_age", f"age < 0 or age > {MAX_PLAUSIBLE_AGE}; age 0 (newborns) is kept")

    scheduled_date = frame["scheduled_at"].dt.normalize()
    frame["appointment_date"] = frame["appointment_date"].dt.normalize()
    frame = _drop(frame, frame["appointment_date"] < scheduled_date, log, "kaggle",
                  "drop_appointment_before_scheduled",
                  "compared by calendar date: AppointmentDay is stored at 00:00, so a same-day booking is valid")

    frame = _drop(frame, frame.duplicated(), log, "kaggle", "drop_exact_duplicates",
                  "rows identical in every column, including AppointmentID")
    _report(frame, frame.drop(columns="appointment_id").duplicated().sum(), log, "kaggle",
            "report_duplicates_ignoring_appointment_id",
            "rows identical to an earlier row in every column except AppointmentID; kept, because they may be "
            "genuine second appointments")

    frame["lead_days"] = (frame["appointment_date"] - frame["scheduled_at"].dt.normalize()).dt.days.astype("int64")
    frame["appt_weekday"] = frame["appointment_date"].dt.dayofweek.astype("int64")
    log.add("kaggle", "derive_lead_days_and_weekday", "derive", len(frame), len(frame), columns_changed=2,
            note="lead_days = whole days from scheduled date to appointment date, ignoring time of day; "
                 "appt_weekday: 0 = Monday")
    return frame[KAGGLE_ORDER]


# ---- cleaning: OpenML -----------------------------------------------------------------------------
def clean_openml(df: pd.DataFrame, log: Optional[CleaningLog] = None) -> pd.DataFrame:
    """Apply the OpenML cleaning rules in order, logging each. The input frame is not modified."""
    log = CleaningLog() if log is None else log
    frame = df.copy()
    n = len(frame)
    log.add("openml", "load", "load", n, n, note=f"positive class no_show = 1 (no_show = 1 - show): "
            f"{int(frame[TARGET].sum()):,} rows ({frame[TARGET].mean():.2%})")

    cosine = [c for c in frame.columns if c.endswith("_c")]
    frame = frame.drop(columns=cosine)
    log.add("openml", "drop_cosine_columns", "drop_columns", n, n, columns_changed=len(cosine),
            note="cosine encodings of month, weekday and hour duplicate the discrete columns; discrete ones kept")

    frame = frame.rename(columns=OPENML_RENAMES)
    numeric = [c for c in frame.columns if c not in {TARGET}]
    frame[numeric] = frame[numeric].astype("int64")
    log.add("openml", "rename_and_cast_columns", "rename", n, n, columns_changed=len(OPENML_RENAMES),
            note="English names; float64 -> int64 (all values are whole numbers)")

    sex_codes = {1: "M", 2: "F"}
    if not frame["sex"].isin(sex_codes).all():
        raise ValueError("OpenML sexo has values other than 1 (male) and 2 (female)")
    frame["sex"] = frame["sex"].map(sex_codes)
    log.add("openml", "recode_sex", "recode", n, n, rows_changed=n, note="1 -> M, 2 -> F, matching Kaggle's Gender")

    for column in ("appt_weekday", "booked_weekday"):
        frame[column] = frame[column] - 1
    log.add("openml", "recode_weekday", "recode", n, n, rows_changed=n, columns_changed=2,
            note="1 = Monday ... 7 = Sunday  ->  0 = Monday ... 6 = Sunday, matching Kaggle's appt_weekday")

    frame = _drop(frame, frame["lead_days"] < 0, log, "openml", "drop_negative_lead_days",
                  "latencia is the number of days between booking and appointment")
    frame = _drop(frame, (frame["age"] < 0) | (frame["age"] > MAX_PLAUSIBLE_AGE), log, "openml",
                  "drop_impossible_age", f"age < 0 or age > {MAX_PLAUSIBLE_AGE}")
    frame = _drop(frame, frame.duplicated(), log, "openml", "drop_exact_duplicates",
                  "rows identical in every column; the file has no patient ID, so identical rows might be distinct "
                  "appointments by similar patients. Dropped as the plan says; the share is small")
    low, high = CLINIC_HOURS
    _report(frame, ((frame["appt_hour"] < low) | (frame["appt_hour"] > high)).sum(), log, "openml",
            "report_hours_outside_clinic_hours", f"appointment hour outside {low}-{high}; kept")
    return frame[list(OPENML_RENAMES.values()) + [TARGET]]


# ---- cleaning: Hangu ------------------------------------------------------------------------------
def clean_hangu(df: pd.DataFrame, log: Optional[CleaningLog] = None) -> pd.DataFrame:
    """Apply the Hangu cleaning rules in order, logging each. The input frame is not modified."""
    log = CleaningLog() if log is None else log
    frame = df.copy()
    n = len(frame)
    log.add("hangu", "load", "load", n, n, note="consultation records of one physician; ServTime is in seconds")

    frame = frame.rename(columns=HANGU_RENAMES)
    log.add("hangu", "rename_columns", "rename", n, n, columns_changed=len(HANGU_RENAMES),
            note="snake_case names; ServTime -> service_seconds")

    frame = _drop(frame, frame["service_seconds"].isna() | (frame["service_seconds"] <= 0), log, "hangu",
                  "drop_missing_or_nonpositive_service_time", "a consultation cannot take zero or negative time")
    frame = _drop(frame, frame["service_seconds"] > MAX_SERVICE_SECONDS, log, "hangu",
                  "drop_service_time_above_limit",
                  f"over {MAX_SERVICE_SECONDS // 3600} hours is a recording error; the real long tail below it is kept")
    frame = _drop(frame, frame.duplicated(), log, "hangu", "drop_exact_duplicates", "rows identical in every column")

    frame["service_seconds"] = frame["service_seconds"].astype("int64")
    frame["service_minutes"] = frame["service_seconds"] / 60.0
    log.add("hangu", "derive_service_minutes", "derive", len(frame), len(frame), columns_changed=1,
            note="service_minutes = service_seconds / 60")
    return frame


# ---- splits ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HoldoutSplit:
    """A locked time-based holdout. `cut` is the first holdout date (or month); everything before it trains."""

    train: pd.DataFrame
    holdout: pd.DataFrame
    cut: object


def split_holdout_by_date(df: pd.DataFrame, date_col: str = "appointment_date",
                          fraction: float = HOLDOUT_FRACTION) -> HoldoutSplit:
    """Hold out the latest `fraction` of distinct appointment dates, cut at a date boundary, never mid-day.

    With n distinct dates the holdout is the latest round(fraction * n) of them (at least one, and at least one
    date is always left for training). Row order and index are preserved.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    dates = np.sort(df[date_col].unique())
    if len(dates) < 2:
        raise ValueError("need at least two distinct appointment dates to split by date")
    n_holdout = min(max(1, round(fraction * len(dates))), len(dates) - 1)
    cut = pd.Timestamp(dates[-n_holdout])
    return HoldoutSplit(train=df[df[date_col] < cut], holdout=df[df[date_col] >= cut], cut=cut)


def split_holdout_last_month(df: pd.DataFrame, month_col: str = "appt_month") -> HoldoutSplit:
    """OpenML has no calendar dates, only the appointment month (1-4, within one year): hold out the last month."""
    months = df[month_col]
    if months.nunique() < 2:
        raise ValueError("need at least two distinct months to hold the last one out")
    cut = months.max()
    return HoldoutSplit(train=df[months < cut], holdout=df[months == cut], cut=int(cut))


def grouped_folds(groups, n_splits: int = 5, seed: int = DEFAULT_SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Patient-grouped cross-validation (GroupKFold on PatientId): one patient is never on both sides.

    Returns (train_positions, validation_positions) per fold, as positions into `groups`, so build it on the
    training part only. Shuffled with `seed`, so the folds are reproducible.
    """
    groups = np.asarray(groups)
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train, valid) for train, valid in splitter.split(np.zeros(len(groups)), groups=groups)]


def stratified_folds(y, n_splits: int = 5, seed: int = DEFAULT_SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified cross-validation (OpenML has no patient ID): every fold keeps the no-show rate."""
    y = np.asarray(y)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train, valid) for train, valid in splitter.split(np.zeros(len(y)), y)]


# ---- command line: describe the raw files ---------------------------------------------------------
def describe_raw_files() -> None:
    from .manifest import file_sha256

    for name, path, kind in (("Kaggle", KAGGLE_PATH, "csv"), ("OpenML", OPENML_PATH, "arff"), ("Hangu", HANGU_PATH, "csv")):
        if not path.is_file():
            print(f"{name:7s} MISSING  {path.relative_to(ROOT)}   (see data/README.md)")
            continue
        frame = {"csv": lambda p: pd.read_csv(p), "arff": lambda p: pd.DataFrame(arff.loadarff(str(p))[0])}[kind](path)
        print(f"{name:7s} {len(frame):>8,} rows  {len(frame.columns):>2} columns  {path.stat().st_size:>10,} bytes  "
              f"sha256 {file_sha256(path)}  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    describe_raw_files()
