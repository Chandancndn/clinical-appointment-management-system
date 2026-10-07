"""The standby list: patients waiting for a slot with a doctor on a day that is full (PLAN section 6, stretch item).

It books nothing, holds nothing and moves nobody. A patient on standby is told on their own pages when a slot on that day
is free again (a cancellation, or a slot reopened after leave) and then books it the normal way: the first to book gets
it, because the database's one-active-booking-per-slot rule is the only thing that hands out slots. Staff see who is
waiting for a day. The booking service (app/services.py) does not import this module.
"""
from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from .dberrors import is_duplicate_key
from .extensions import db
from .models import Booking, Doctor, Slot, Standby, User

MAX_WAITING = 10  # days one patient can wait for at the same time


class StandbyError(Exception):
    """The request makes no sense right now; the message is safe to show the patient."""

    http_status = 409


class NothingToWaitFor(StandbyError):
    pass


class SlotsAreFree(StandbyError):
    pass


class TooMany(StandbyError):
    pass


def _day_counts(doctor_id: int, day: date, now: datetime) -> tuple:
    """(slots that have not started, how many of those are free) for a doctor on a day."""
    rows = db.session.execute(
        sa.select(Slot.slot_time, Booking.id)
        .select_from(Slot).outerjoin(Booking, Booking.confirmed_slot_id == Slot.id)
        .where(Slot.doctor_id == doctor_id, Slot.slot_date == day)
    ).all()
    ahead = [booking_id for slot_time, booking_id in rows if datetime.combine(day, slot_time) > now]
    return len(ahead), sum(1 for booking_id in ahead if booking_id is None)


def is_full(doctor_id: int, day: date, now: datetime) -> bool:
    """True when the day still has slots ahead and none of them is free: the only time standby makes sense."""
    ahead, free = _day_counts(doctor_id, day, now)
    return ahead > 0 and free == 0


def _find(patient_id: int, doctor_id: int, day: date):
    return db.session.execute(
        sa.select(Standby).where(Standby.patient_id == patient_id, Standby.doctor_id == doctor_id, Standby.slot_date == day)
    ).scalar_one_or_none()


def is_waiting(patient_id: int, doctor_id: int, day: date) -> bool:
    return _find(patient_id, doctor_id, day) is not None


def join(patient_id: int, doctor_id: int, day: date, now: datetime) -> Standby:
    """Put the patient on standby for the doctor and day. Joining twice keeps one entry (the database's unique key)."""
    ahead, free = _day_counts(doctor_id, day, now)
    if ahead == 0:
        raise NothingToWaitFor("This doctor has no slots still to come on that day, so there is nothing to wait for.")
    if free:
        raise SlotsAreFree("There are free slots on that day. You can book one now.")
    existing = _find(patient_id, doctor_id, day)
    if existing is not None:
        return existing
    waiting = db.session.execute(
        sa.select(sa.func.count()).select_from(Standby).where(Standby.patient_id == patient_id, Standby.slot_date >= now.date())
    ).scalar_one()
    if waiting >= MAX_WAITING:
        raise TooMany(f"You are already on standby for {MAX_WAITING} days. Leave one before joining another.")
    entry = Standby(patient_id=patient_id, doctor_id=doctor_id, slot_date=day)
    db.session.add(entry)
    try:
        db.session.commit()
    except IntegrityError as exc:
        db.session.rollback()
        if is_duplicate_key(exc):  # two clicks at once: the other one created it
            return _find(patient_id, doctor_id, day)
        raise
    return entry


def leave(standby_id: int, patient_id: int) -> bool:
    """Remove the patient's own entry. False if it does not exist or belongs to someone else."""
    removed = db.session.execute(
        sa.delete(Standby).where(Standby.id == standby_id, Standby.patient_id == patient_id)
    ).rowcount
    db.session.commit()
    return removed == 1


def clear_for(patient_id: int, doctor_id: int, day: date) -> None:
    """The patient now has an appointment with this doctor on this day, so the wait is over."""
    db.session.execute(
        sa.delete(Standby).where(Standby.patient_id == patient_id, Standby.doctor_id == doctor_id, Standby.slot_date == day))
    db.session.commit()


def for_patient(patient_id: int, now: datetime) -> list:
    """The patient's entries from today on, each with the number of slots free right now on that day."""
    rows = db.session.execute(
        sa.select(Standby.id, Standby.doctor_id, Standby.slot_date, Doctor.specialization, User.name.label("doctor_name"))
        .join(Doctor, Doctor.id == Standby.doctor_id).join(User, User.id == Doctor.user_id)
        .where(Standby.patient_id == patient_id, Standby.slot_date >= now.date())
        .order_by(Standby.slot_date, Standby.id)
    ).all()
    return [SimpleNamespace(id=r.id, doctor_id=r.doctor_id, slot_date=r.slot_date, specialization=r.specialization,
                            doctor_name=r.doctor_name, free=_day_counts(r.doctor_id, r.slot_date, now)[1]) for r in rows]


def for_doctor_day(doctor_id: int, day: date) -> list:
    """Who is waiting for this doctor on this day, in the order they joined. For staff pages only."""
    return db.session.execute(
        sa.select(User.name.label("patient_name"), Standby.created_at)
        .join(User, User.id == Standby.patient_id)
        .where(Standby.doctor_id == doctor_id, Standby.slot_date == day)
        .order_by(Standby.created_at, Standby.id)
    ).all()


def count_waiting(now: datetime) -> int:
    return db.session.execute(
        sa.select(sa.func.count()).select_from(Standby).where(Standby.slot_date >= now.date())
    ).scalar_one()
