"""The clinic's clock. A hosted server runs on UTC, so the app can be told the clinic's own time zone
(CLINIC_TIMEZONE, e.g. Asia/Kolkata). Slot dates and times are stored as clinic-local wall time, so every
"has this slot started?" rule must compare against that same wall time. Without the setting nothing changes."""
from __future__ import annotations

from datetime import datetime

import pytest

from app import clock, create_app


def hours_apart(first: datetime, second: datetime) -> float:
    return (first - second).total_seconds() / 3600


def test_the_clock_follows_the_clinic_time_zone(monkeypatch):
    monkeypatch.setenv("CLINIC_TIMEZONE", "Pacific/Kiritimati")  # UTC+14, no daylight saving
    ahead = clock.now()
    monkeypatch.setenv("CLINIC_TIMEZONE", "Pacific/Pago_Pago")  # UTC-11, no daylight saving
    behind = clock.now()
    assert round(hours_apart(ahead, behind)) == 25  # the same instant, 25 hours apart on the wall clock


def test_the_clock_is_naive_like_the_stored_slot_times(monkeypatch):
    monkeypatch.setenv("CLINIC_TIMEZONE", "Asia/Kolkata")
    assert clock.now().tzinfo is None  # comparable with datetime.combine(slot_date, slot_time)


def test_without_the_setting_the_clock_is_the_machines_local_time(monkeypatch):
    monkeypatch.delenv("CLINIC_TIMEZONE", raising=False)
    assert abs(hours_apart(clock.now(), datetime.now())) < 0.01
    monkeypatch.setenv("CLINIC_TIMEZONE", "   ")  # blank counts as unset
    assert abs(hours_apart(clock.now(), datetime.now())) < 0.01


def test_an_unknown_time_zone_stops_the_app_at_start_up(monkeypatch, tmp_path):
    monkeypatch.setenv("CLINIC_TIMEZONE", "Mars/Olympus_Mons")
    with pytest.raises(RuntimeError, match="CLINIC_TIMEZONE"):
        create_app({"SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'tz.sqlite3'}", "SECRET_KEY": "test"})
