"""Slot generation. Slots are rows the database guards with UNIQUE (doctor_id, slot_date, slot_time)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from . import clock, services
from .dberrors import is_duplicate_key
from .extensions import db
from .models import Booking, Doctor, Slot
from .services import NotFound

MAX_DAYS = 62


class ScheduleError(ValueError):
    """The requested slot range makes no sense; the message is safe to show the user."""


@dataclass
class SlotResult:
    created: int
    existing: int


@dataclass
class LeaveResult:
    closed: int  # free slots closed now
    booked: int  # slots in the range that patients hold; closing never touches them


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


def _check_range(first_day: date, last_day: date) -> None:
    if last_day < first_day:
        raise ScheduleError("The last date is before the first date.")
    if (last_day - first_day).days + 1 > MAX_DAYS:
        raise ScheduleError(f"Please choose at most {MAX_DAYS} days at a time.")


def close_free_slots(doctor_id: int, first_day: date, last_day: date, actor_id: int) -> LeaveResult:
    """Close every free slot of the doctor in the range that has not started (the doctor's leave).

    Booked slots are never touched: they are counted so the page can say how many appointments remain. Each slot is closed
    by services.close_slot(), so a patient who books a slot at the same instant either gets it (and it is counted as
    booked) or does not; the database decides. Safe to run twice.
    """
    if db.session.get(Doctor, doctor_id) is None:
        raise NotFound(f"doctor {doctor_id} does not exist")
    _check_range(first_day, last_day)
    now = clock.now()
    rows = db.session.execute(
        sa.select(Slot.id, Slot.slot_date, Slot.slot_time, Booking.id.label("booking_id"), Booking.status)
        .select_from(Slot).outerjoin(Booking, Booking.confirmed_slot_id == Slot.id)
        .where(Slot.doctor_id == doctor_id, Slot.slot_date.between(first_day, last_day))
        .order_by(Slot.slot_date, Slot.slot_time)
    ).all()
    db.session.rollback()  # end the read; every close below is its own short transaction
    closed = booked = 0
    for row in rows:
        if datetime.combine(row.slot_date, row.slot_time) <= now:
            continue
        if row.booking_id is None:
            try:
                services.close_slot(row.id, actor_id)
                closed += 1
            except services.SlotTaken:  # a patient booked it between the read and the close
                booked += 1
            except services.SlotInPast:
                pass
        elif row.status != services.CLOSED:
            booked += 1
    return LeaveResult(closed=closed, booked=booked)


def reopen_slots(doctor_id: int, first_day: date, last_day: date, actor_id: int) -> int:
    """Reopen the doctor's closed slots in the range; returns how many. Appointments are never touched."""
    if db.session.get(Doctor, doctor_id) is None:
        raise NotFound(f"doctor {doctor_id} does not exist")
    _check_range(first_day, last_day)
    slot_ids = [row.id for row in db.session.execute(
        sa.select(Slot.id).where(Slot.doctor_id == doctor_id, Slot.slot_date.between(first_day, last_day)))]
    if not slot_ids:
        db.session.rollback()
        return 0
    statement = (
        sa.update(Booking)
        .where(Booking.slot_id.in_(slot_ids), Booking.status == services.CLOSED)
        .values(status="cancelled", cancelled_at=sa.func.now(), cancelled_by=actor_id)
        .execution_options(synchronize_session=False)
    )
    try:
        reopened = db.session.execute(statement).rowcount
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise
    return reopened
