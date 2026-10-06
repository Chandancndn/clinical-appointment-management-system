"""Features for the research models: groups, the leak-free patient history, and the feature builders.

HISTORY RULE (CLAUDE.md): a booking's "earlier appointments" and "earlier no-show rate" use only appointments of
the same patient whose APPOINTMENT DATE is strictly before this booking's BOOKING DATE, because only then was
the outcome known when the booking was made. That is stricter than "before the current appointment": a visit
that happened between the booking and the appointment is not usable, and neither is one on the booking day itself.

The same functions will serve the deployable model and the app (M5, M7), so training and serving cannot drift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HISTORY_COLUMNS = ["prev_appointments", "prev_no_shows", "prev_no_show_rate", "has_history"]

# ---- feature groups (what E4 ablates) -----------------------------------------------------------------
KAGGLE_GROUPS = {
    "history": ["prev_appointments", "prev_no_show_rate", "has_history"],
    "lead_calendar": ["lead_days", "lead_same_day", "appt_weekday"],
    "demographics": ["age", "is_female"],
    "flags": ["sms_received", "scholarship", "hypertension", "diabetes", "alcoholism", "handicap"],
    "neighbourhood": ["neighbourhood"],
}
# OpenML has no patient id (no history) and no health flags. Its month columns are left out: they identify the
# calendar position, and the holdout is the last month, which training never contains.
OPENML_GROUPS = {
    "lead_calendar": ["lead_days", "lead_same_day", "appt_weekday", "appt_hour", "booked_weekday", "booked_hour"],
    "demographics": ["age", "is_female"],
    "booking": ["channel", "is_procedure", "specialty"],
}
GROUPS = {"kaggle": KAGGLE_GROUPS, "openml": OPENML_GROUPS}

# How each column is preprocessed (inside the pipeline, per training fold; see train.build_pipeline)
COLUMN_KIND = {
    "prev_appointments": "numeric", "prev_no_show_rate": "numeric", "has_history": "binary",
    "lead_days": "numeric", "lead_same_day": "binary", "appt_weekday": "category_small",
    "age": "numeric", "is_female": "binary",
    "sms_received": "binary", "scholarship": "binary", "hypertension": "binary", "diabetes": "binary",
    "alcoholism": "binary", "handicap": "numeric",
    "neighbourhood": "target_encoded",
    "appt_hour": "numeric", "booked_weekday": "category_small", "booked_hour": "numeric",
    "channel": "category_small", "is_procedure": "binary", "specialty": "target_encoded",
}

# What exists, with the same meaning, in both datasets (E6 transfer): age, sex, lead time, weekday
COMMON_COLUMNS = ["age", "is_female", "lead_days", "lead_same_day", "appt_weekday"]


@dataclass
class FeatureSpec:
    numeric: list
    binary: list
    category_small: list
    target_encoded: list


def all_columns(dataset: str) -> list[str]:
    return [c for group in GROUPS[dataset].values() for c in group]


def columns_for(dataset: str, groups) -> list[str]:
    """Columns of the named groups, in the canonical group order (not the order asked for)."""
    known = GROUPS[dataset]
    unknown = [g for g in groups if g not in known]
    if unknown:
        raise KeyError(f"unknown feature group(s) for {dataset}: {unknown}")
    return [c for name, columns in known.items() if name in set(groups) for c in columns]


def feature_spec(columns) -> FeatureSpec:
    columns = list(columns)
    kinds = {c: COLUMN_KIND[c] for c in columns}
    return FeatureSpec(
        numeric=[c for c in columns if kinds[c] == "numeric"],
        binary=[c for c in columns if kinds[c] == "binary"],
        category_small=[c for c in columns if kinds[c] == "category_small"],
        target_encoded=[c for c in columns if kinds[c] == "target_encoded"],
    )


# ---- leak-free patient history ------------------------------------------------------------------------
_EPOCH = pd.Timestamp("1970-01-01")
_DAY_SPAN = 1_000_000  # larger than any day number, so (patient, day) packs into one sortable integer


def add_history(df: pd.DataFrame, patient_col: str = "patient_id", booked_col: str = "scheduled_at",
                appointment_col: str = "appointment_date", target: str = "no_show") -> pd.DataFrame:
    """Add prev_appointments, prev_no_shows, prev_no_show_rate and has_history (the strict history rule above).

    prev_no_show_rate is NaN for a patient with no earlier appointment; has_history says so explicitly.
    Run it on cleaned data: a row whose appointment is dated before its own booking would count itself.
    """
    out = df.copy()
    booked = (out[booked_col].dt.normalize() - _EPOCH).dt.days.to_numpy("int64")
    appointment = (out[appointment_col].dt.normalize() - _EPOCH).dt.days.to_numpy("int64")
    if (appointment < booked).any():
        raise ValueError("some appointments are dated before their own booking date; clean the data first")
    codes, _ = pd.factorize(out[patient_col])
    codes = codes.astype("int64")

    # sort every appointment by (patient, appointment day); a booking's usable history is the block of that
    # patient's appointments with day < booking day, found by two binary searches on the packed key
    key = codes * _DAY_SPAN + appointment
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    cumulative_no_shows = np.concatenate([[0], np.cumsum(out[target].to_numpy("int64")[order])])
    start = np.searchsorted(sorted_key, codes * _DAY_SPAN, side="left")
    stop = np.searchsorted(sorted_key, codes * _DAY_SPAN + booked, side="left")  # strictly before the booking day

    previous = stop - start
    previous_no_shows = cumulative_no_shows[stop] - cumulative_no_shows[start]
    out["prev_appointments"] = previous.astype("int64")
    out["prev_no_shows"] = previous_no_shows.astype("int64")
    out["prev_no_show_rate"] = np.where(previous > 0, previous_no_shows / np.maximum(previous, 1), np.nan)
    out["has_history"] = (previous > 0).astype("int64")
    return out


# ---- feature builders ---------------------------------------------------------------------------------
def kaggle_features(df: pd.DataFrame) -> pd.DataFrame:
    """The Kaggle feature matrix, same index as `df`. Needs the history columns (see add_history)."""
    return pd.DataFrame({
        "prev_appointments": df["prev_appointments"], "prev_no_show_rate": df["prev_no_show_rate"],
        "has_history": df["has_history"],
        "lead_days": df["lead_days"], "lead_same_day": (df["lead_days"] == 0).astype("int64"),
        "appt_weekday": df["appt_weekday"],
        "age": df["age"], "is_female": (df["sex"] == "F").astype("int64"),
        "sms_received": df["sms_received"], "scholarship": df["scholarship"], "hypertension": df["hypertension"],
        "diabetes": df["diabetes"], "alcoholism": df["alcoholism"], "handicap": df["handicap"],
        "neighbourhood": df["neighbourhood"],
    }, index=df.index)[all_columns("kaggle")]


def openml_features(df: pd.DataFrame) -> pd.DataFrame:
    """The OpenML feature matrix, same index as `df`."""
    return pd.DataFrame({
        "lead_days": df["lead_days"], "lead_same_day": (df["lead_days"] == 0).astype("int64"),
        "appt_weekday": df["appt_weekday"], "appt_hour": df["appt_hour"],
        "booked_weekday": df["booked_weekday"], "booked_hour": df["booked_hour"],
        "age": df["age"], "is_female": (df["sex"] == "F").astype("int64"),
        "channel": df["channel"], "is_procedure": (df["appt_type"] == 2).astype("int64"),
        "specialty": df["specialty"],
    }, index=df.index)[all_columns("openml")]
