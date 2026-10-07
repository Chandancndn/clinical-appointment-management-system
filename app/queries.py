"""Read-only queries used by the pages. Nothing here changes data.

A slot is "taken" when an active booking points at it through bookings.confirmed_slot_id, the same
generated column the database uses to forbid double-booking.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import aliased

from .extensions import db
from .models import Booking, Doctor, Slot, User

PatientUser = aliased(User)
DoctorUser = aliased(User)


def _start(row) -> datetime:
    return datetime.combine(row.slot_date, row.slot_time)


def _booking_rows():
    """One row per booking with its slot, doctor and patient details."""
    return (
        sa.select(
            Booking.id.label("id"), Booking.status, Booking.reason, Booking.created_at, Booking.cancelled_at,
            Slot.id.label("slot_id"), Slot.slot_date, Slot.slot_time, Slot.doctor_id,
            Doctor.specialization, DoctorUser.name.label("doctor_name"),
            PatientUser.id.label("patient_id"), PatientUser.name.label("patient_name"),
        )
        .join(Slot, Slot.id == Booking.slot_id)
        .join(Doctor, Doctor.id == Slot.doctor_id)
        .join(DoctorUser, DoctorUser.id == Doctor.user_id)
        .join(PatientUser, PatientUser.id == Booking.patient_id)
    )


# doctors ---------------------------------------------------------------------------------------
def _doctor_rows():
    return sa.select(Doctor.id, Doctor.user_id, Doctor.specialization, Doctor.slot_minutes,
                     User.name).join(User, User.id == Doctor.user_id)


def list_doctors():
    return db.session.execute(_doctor_rows().order_by(User.name)).all()


def get_doctor(doctor_id: int):
    return db.session.execute(_doctor_rows().where(Doctor.id == doctor_id)).one_or_none()


def doctor_for_user(user_id: int):
    return db.session.execute(_doctor_rows().where(Doctor.user_id == user_id)).one_or_none()


# slots -----------------------------------------------------------------------------------------
def slot_info(slot_id: int):
    return db.session.execute(
        sa.select(Slot.id, Slot.doctor_id, Slot.slot_date, Slot.slot_time).where(Slot.id == slot_id)
    ).one_or_none()


def _slot_rows(doctor_id: int, first_day: date, last_day: date):
    return db.session.execute(
        sa.select(Slot.id, Slot.slot_date, Slot.slot_time, Booking.id.label("booking_id"))
        .select_from(Slot)
        .outerjoin(Booking, Booking.confirmed_slot_id == Slot.id)
        .where(Slot.doctor_id == doctor_id, Slot.slot_date.between(first_day, last_day))
        .order_by(Slot.slot_date, Slot.slot_time)
    ).all()


def day_slots(doctor_id: int, day: date, now: datetime):
    """A doctor's slots for one day, each with state 'free', 'taken' or 'past' and its booking id."""
    slots = []
    for row in _slot_rows(doctor_id, day, day):
        if datetime.combine(row.slot_date, row.slot_time) <= now:
            state = "past"
        else:
            state = "taken" if row.booking_id is not None else "free"
        slots.append(SimpleNamespace(id=row.id, time=row.slot_time, state=state, booking_id=row.booking_id))
    return slots


def upcoming_days(doctor_id: int, first_day: date, count: int, now: datetime):
    """For each of `count` days from `first_day`: how many future slots are still free."""
    last_day = first_day + timedelta(days=count - 1)
    free = {}
    for row in _slot_rows(doctor_id, first_day, last_day):
        is_free = row.booking_id is None and datetime.combine(row.slot_date, row.slot_time) > now
        free[row.slot_date] = free.get(row.slot_date, 0) + (1 if is_free else 0)
    days = [first_day + timedelta(days=i) for i in range(count)]
    return [SimpleNamespace(day=d, has_slots=d in free, free=free.get(d, 0)) for d in days]


# bookings --------------------------------------------------------------------------------------
def patient_bookings(patient_id: int):
    return db.session.execute(
        _booking_rows().where(Booking.patient_id == patient_id)
        .order_by(Slot.slot_date, Slot.slot_time)
    ).all()


def booking_for_patient(booking_id: int, patient_id: int):
    """The booking if it belongs to this patient, else None (callers answer 404 either way)."""
    return db.session.execute(
        _booking_rows().where(Booking.id == booking_id, Booking.patient_id == patient_id)
    ).one_or_none()


def booking_for_doctor(booking_id: int, doctor_id: int):
    """The booking if it is on this doctor's schedule, else None."""
    return db.session.execute(
        _booking_rows().where(Booking.id == booking_id, Slot.doctor_id == doctor_id)
    ).one_or_none()


def booking_row(booking_id: int):
    return db.session.execute(_booking_rows().where(Booking.id == booking_id)).one_or_none()


def patient_clash(patient_id: int, slot_id: int, ignoring_booking_id: Optional[int] = None):
    """The patient's active booking whose time overlaps the slot `slot_id` (with ANY doctor), or None.

    Two appointments overlap when each starts before the other ends; an appointment lasts as long as its doctor's slot length,
    so a 30-minute appointment also blocks a 15-minute slot that starts inside it, and back-to-back appointments are fine.
    This is a convenience rule for patients (one person cannot be in two consulting rooms at once). The page looks it up
    before it tries to book, so it is best effort for one patient clicking twice at the same instant. The rule that matters,
    one active booking per doctor, date and time, is the database's and is not involved here.
    """
    wanted = db.session.execute(
        sa.select(Slot.slot_date, Slot.slot_time, Doctor.slot_minutes).join(Doctor, Doctor.id == Slot.doctor_id).where(Slot.id == slot_id)
    ).one_or_none()
    if wanted is None:
        return None
    wanted_start = datetime.combine(wanted.slot_date, wanted.slot_time)
    wanted_end = wanted_start + timedelta(minutes=wanted.slot_minutes)
    query = (
        sa.select(Booking.id, Slot.slot_time, Doctor.slot_minutes, DoctorUser.name.label("doctor_name"))
        .join(Slot, Slot.id == Booking.slot_id)
        .join(Doctor, Doctor.id == Slot.doctor_id)
        .join(DoctorUser, DoctorUser.id == Doctor.user_id)
        .where(Booking.patient_id == patient_id, Booking.confirmed_slot_id.is_not(None), Slot.slot_date == wanted.slot_date)
    )
    if ignoring_booking_id is not None:
        query = query.where(Booking.id != ignoring_booking_id)
    for row in db.session.execute(query):  # one patient's bookings on one day: a handful of rows
        start = datetime.combine(wanted.slot_date, row.slot_time)
        if start < wanted_end and wanted_start < start + timedelta(minutes=row.slot_minutes):
            return row
    return None


def doctor_schedule(doctor_id: int, day: date):
    """A doctor's day: every slot, with the active booking and patient name if there is one."""
    return db.session.execute(
        sa.select(
            Slot.id.label("slot_id"), Slot.slot_date, Slot.slot_time,
            Booking.id.label("booking_id"), Booking.status, Booking.reason,
            PatientUser.name.label("patient_name"),
        )
        .select_from(Slot)
        .outerjoin(Booking, Booking.confirmed_slot_id == Slot.id)
        .outerjoin(PatientUser, PatientUser.id == Booking.patient_id)
        .where(Slot.doctor_id == doctor_id, Slot.slot_date == day)
        .order_by(Slot.slot_time)
    ).all()


def week_summary(doctor_id: int, day: date):
    """Mon-Sun of the week containing `day`: (date, total slots, booked slots) for each day."""
    monday = day - timedelta(days=day.weekday())
    sunday = monday + timedelta(days=6)
    rows = db.session.execute(
        sa.select(Slot.slot_date, sa.func.count(Slot.id), sa.func.count(Booking.id))
        .select_from(Slot)
        .outerjoin(Booking, Booking.confirmed_slot_id == Slot.id)
        .where(Slot.doctor_id == doctor_id, Slot.slot_date.between(monday, sunday))
        .group_by(Slot.slot_date)
    ).all()
    by_day = {slot_date: (total, booked) for slot_date, total, booked in rows}
    return [SimpleNamespace(day=monday + timedelta(days=i),
                            total=by_day.get(monday + timedelta(days=i), (0, 0))[0],
                            booked=by_day.get(monday + timedelta(days=i), (0, 0))[1])
            for i in range(7)]


# admin -----------------------------------------------------------------------------------------
def _upcoming_filter(now: datetime):
    return sa.and_(
        Booking.status == "confirmed",
        sa.or_(Slot.slot_date > now.date(), sa.and_(Slot.slot_date == now.date(), Slot.slot_time > now.time())),
    )


def admin_counts(now: datetime):
    by_role = dict(db.session.execute(sa.select(User.role, sa.func.count()).group_by(User.role)).all())
    by_status = dict(db.session.execute(
        sa.select(Booking.status, sa.func.count()).group_by(Booking.status)).all())
    upcoming = db.session.execute(
        sa.select(sa.func.count()).select_from(Booking).join(Slot, Slot.id == Booking.slot_id)
        .where(_upcoming_filter(now))
    ).scalar_one()
    return SimpleNamespace(patients=by_role.get("patient", 0), doctors=by_role.get("doctor", 0),
                           by_status={s: by_status.get(s, 0) for s in
                                      ("confirmed", "completed", "no_show", "cancelled")},
                           upcoming=upcoming)


def upcoming_bookings(limit: int, now: datetime):
    return db.session.execute(
        _booking_rows().where(_upcoming_filter(now)).order_by(Slot.slot_date, Slot.slot_time).limit(limit)
    ).all()


def bookings_page(page: int, per_page: int, status: Optional[str] = None):
    """(rows, total) for the admin's all-bookings list, newest appointment first."""
    query = _booking_rows()
    count = sa.select(sa.func.count()).select_from(Booking)
    if status:
        query = query.where(Booking.status == status)
        count = count.where(Booking.status == status)
    rows = db.session.execute(
        query.order_by(Slot.slot_date.desc(), Slot.slot_time.desc(), Booking.id.desc())
        .limit(per_page).offset((page - 1) * per_page)
    ).all()
    return rows, db.session.execute(count).scalar_one()
