"""Synthetic stand-ins for the cleaned Kaggle and OpenML frames, with real signal so tests are not vacuous."""
from __future__ import annotations

import numpy as np
import pandas as pd

NEIGHBOURHOODS = [f"N{i:02d}" for i in range(12)]


def synthetic_kaggle(n: int = 6000, n_patients: int = 2500, n_dates: int = 30, seed: int = 0,
                     signal: float = 1.0) -> pd.DataFrame:
    """Same columns as data.clean_kaggle. No-show depends on lead time, age, neighbourhood and a per-patient
    propensity (so earlier no-shows are genuinely predictive)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-05-02", periods=n_dates)
    propensity = rng.beta(1.2, 5.0, n_patients)
    neighbourhood_effect = rng.normal(0, 0.4, len(NEIGHBOURHOODS))
    patient = rng.integers(0, n_patients, n)
    appt = dates[rng.integers(0, n_dates, n)]
    lead = rng.choice([0, 0, 0, 1, 2, 4, 7, 12, 20, 40], n)
    age = np.clip(rng.normal(37, 18, n).round(), 0, 100).astype(int)
    hood = rng.integers(0, len(NEIGHBOURHOODS), n)
    logit = (-2.6 + 5.0 * propensity[patient] + neighbourhood_effect[hood]
             + signal * (-1.5 * (lead == 0) + 0.025 * np.minimum(lead, 30) - 0.01 * (age - 37)))  # signal scales lead and age
    no_show = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8")
    scheduled = pd.DatetimeIndex(appt) - pd.to_timedelta(lead, unit="D") + pd.to_timedelta(rng.integers(7, 18, n), unit="h")
    return pd.DataFrame({
        "patient_id": (1_000_000 + patient).astype("int64"),
        "appointment_id": np.arange(n, dtype="int64") + 5_000_000,
        "sex": np.where(rng.random(n) < 0.6, "F", "M"),
        "scheduled_at": scheduled,
        "appointment_date": pd.DatetimeIndex(appt),
        "age": age,
        "neighbourhood": np.array(NEIGHBOURHOODS)[hood],
        "scholarship": (rng.random(n) < 0.1).astype(int),
        "hypertension": (rng.random(n) < 0.2).astype(int),
        "diabetes": (rng.random(n) < 0.07).astype(int),
        "alcoholism": (rng.random(n) < 0.03).astype(int),
        "handicap": rng.choice([0, 0, 0, 0, 1, 2], n),
        "sms_received": ((lead >= 3) & (rng.random(n) < 0.7)).astype(int),
        "lead_days": lead.astype("int64"),
        "appt_weekday": pd.DatetimeIndex(appt).dayofweek.astype("int64"),
        "no_show": no_show,
    }).sort_values(["appointment_date", "appointment_id"]).reset_index(drop=True)


def synthetic_openml(n: int = 5000, seed: int = 1) -> pd.DataFrame:
    """Same columns as data.clean_openml, months 1-4, no patient id."""
    rng = np.random.default_rng(seed)
    month = rng.integers(1, 5, n)
    lead = rng.choice([0, 1, 2, 4, 7, 14, 30], n)
    age = np.clip(rng.normal(35, 20, n).round(), 0, 100).astype(int)
    channel = rng.integers(1, 4, n)
    specialty = rng.integers(1, 30, n)
    effect = rng.normal(0, 0.4, 30)
    hour = rng.integers(8, 19, n)
    logit = -1.6 + 0.02 * np.minimum(lead, 30) - 0.012 * (age - 35) + 0.3 * (channel == 3) + effect[specialty] + 0.03 * (hour - 12)
    no_show = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8")
    return pd.DataFrame({
        "specialty": specialty.astype("int64"), "age": age.astype("int64"),
        "sex": np.where(rng.random(n) < 0.6, "F", "M"), "appt_month": month.astype("int64"),
        "appt_weekday": rng.integers(0, 6, n).astype("int64"), "appt_hour": hour.astype("int64"),
        "booked_month": month.astype("int64"), "booked_weekday": rng.integers(0, 6, n).astype("int64"),
        "booked_hour": rng.integers(8, 19, n).astype("int64"), "lead_days": lead.astype("int64"),
        "channel": channel.astype("int64"), "appt_type": rng.choice([1, 2], n, p=[0.9, 0.1]).astype("int64"),
        "no_show": no_show,
    })
