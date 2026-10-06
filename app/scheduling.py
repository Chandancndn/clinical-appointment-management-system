"""Slot generation. Slots are rows the database guards with UNIQUE (doctor_id, slot_date, slot_time)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from .dberrors import is_duplicate_key
from .extensions import db
from .models import Doctor, Slot
from .services import NotFound

MAX_DAYS = 62


class ScheduleError(ValueError):
    """The requested slot range makes no sense; the message is safe to show the user."""


@dataclass
class SlotResult:
    created: int
    existing: int


def _times_for_day(start: time, end: time, minutes: int) -> list[time]:
    """Slot start times from `start` while a whole slot still fits before `end`."""
    first, stop = datetime.combine(date.min, start), datetime.combine(date.min, end)
    times = []
    while first + timedelta(minutes=minutes) <= stop:
        times.append(first.time())
        first += timedelta(minutes=minutes)
    return times


def generate_slots(doctor_id: int, first_day: date, last_day: date, start: time, end: time,
                   skip_weekends: bool = False) -> SlotResult:
    """Create the missing slots for a doctor over a date range, using the doctor's slot length.

    Safe to run twice: slots that already exist are left alone and counted, never duplicated.
    """
    doctor = db.session.get(Doctor, doctor_id)
    if doctor is None:
        raise NotFound(f"doctor {doctor_id} does not exist")
    minutes = doctor.slot_minutes
    if last_day < first_day:
        raise ScheduleError("The last date is before the first date.")
    if (last_day - first_day).days + 1 > MAX_DAYS:
        raise ScheduleError(f"Please generate at most {MAX_DAYS} days at a time.")
    times = _times_for_day(start, end, minutes)
    if not times:
        raise ScheduleError(f"The time window is shorter than one {minutes}-minute slot.")

    days = [first_day + timedelta(days=i) for i in range((last_day - first_day).days + 1)]
    if skip_weekends:
        days = [d for d in days if d.weekday() < 5]
    wanted = [(d, t) for d in days for t in times]

    for attempt in (1, 2):  # a second pass covers a concurrent generator inserting the same slots
        have = {(row.slot_date, row.slot_time) for row in db.session.execute(
            sa.select(Slot.slot_date, Slot.slot_time)
            .where(Slot.doctor_id == doctor_id, Slot.slot_date.between(first_day, last_day)))}
        missing = [(d, t) for d, t in wanted if (d, t) not in have]
        db.session.add_all(Slot(doctor_id=doctor_id, slot_date=d, slot_time=t) for d, t in missing)
        try:
            db.session.commit()
            return SlotResult(created=len(missing), existing=len(wanted) - len(missing))
        except IntegrityError as exc:
            db.session.rollback()
            if not is_duplicate_key(exc) or attempt == 2:
                raise
    raise AssertionError("unreachable")
