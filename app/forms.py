"""Parsing and validating submitted form values. Returns (clean values, list of error messages)."""
from __future__ import annotations

import re
from datetime import date, time, timedelta
from typing import Optional

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD = 8
SEXES = ("F", "M")


MAX_DB_ID = 2**31 - 1  # ids are INT columns; anything larger cannot exist and must not reach the database
MAX_PAGE = 1_000_000
EARLIEST, LATEST = date(1900, 1, 1), date(2100, 12, 31)  # dates outside this are typos, and the ends of the calendar overflow


def parse_date(value) -> Optional[date]:
    """An ISO date between 1900 and 2100, else None."""
    try:
        parsed = date.fromisoformat(value.strip())
    except (AttributeError, TypeError, ValueError):
        return None
    return parsed if EARLIEST <= parsed <= LATEST else None


def parse_time(value) -> Optional[time]:
    try:
        return time.fromisoformat(value.strip()).replace(second=0, microsecond=0)
    except (AttributeError, TypeError, ValueError):
        return None


def parse_int(value, low: Optional[int] = None, high: Optional[int] = MAX_DB_ID) -> Optional[int]:
    """A whole number between `low` and `high` (default: the largest database id), else None."""
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if (low is not None and number < low) or (high is not None and number > high):
        return None
    return number


def _check_credentials(form, errors: list) -> tuple[str, str, str]:
    name = (form.get("name") or "").strip()
    email = (form.get("email") or "").strip().lower()
    password = form.get("password") or ""
    if not 1 <= len(name) <= 120:
        errors.append("Enter a name (up to 120 characters).")
    if not EMAIL_RE.match(email) or len(email) > 255:
        errors.append("Enter a valid email address.")
    if not MIN_PASSWORD <= len(password) <= 128:
        errors.append(f"Choose a password of {MIN_PASSWORD} to 128 characters.")
    return name, email, password


def parse_registration(form, today: date):
    errors: list[str] = []
    name, email, password = _check_credentials(form, errors)
    dob = parse_date(form.get("date_of_birth"))
    if dob is None or dob > today or dob < today - timedelta(days=366 * 120):
        errors.append("Enter a valid date of birth.")
    sex = form.get("sex")
    if sex not in SEXES:
        errors.append("Select a sex (F or M).")
    return {"name": name, "email": email, "password": password, "date_of_birth": dob, "sex": sex}, errors


def parse_new_doctor(form):
    errors: list[str] = []
    name, email, password = _check_credentials(form, errors)
    specialization = (form.get("specialization") or "").strip()
    if not 1 <= len(specialization) <= 100:
        errors.append("Enter a specialization (up to 100 characters).")
    minutes = parse_int(form.get("slot_minutes"), 5, 120)
    if minutes is None:
        errors.append("Slot length must be a whole number of minutes between 5 and 120.")
    return {"name": name, "email": email, "password": password,
            "specialization": specialization, "slot_minutes": minutes}, errors


def parse_time_window(form):
    errors: list[str] = []
    start, end = parse_time(form.get("start_time")), parse_time(form.get("end_time"))
    if start is None or end is None:
        errors.append("Enter a start and end time (HH:MM).")
    elif end <= start:
        errors.append("The end time must be after the start time.")
    return start, end, errors
