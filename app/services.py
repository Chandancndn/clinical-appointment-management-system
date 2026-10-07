"""Booking service layer: the only code that changes bookings.

Rules (CLAUDE.md hard rule 1):
  * book() just INSERTs. The database's unique key on bookings.confirmed_slot_id decides who gets
    a slot, and a duplicate-key error means "slot taken" (HTTP 409). Nothing here checks whether
    the slot is free and then inserts, because that leaves a gap two requests can slip through.
    (book() does look the slot up first, but only for the time rule below; availability is never
    checked in Python.)
  * cancel(), reschedule() and mark_outcome() change a booking with a conditional UPDATE
    (WHERE status ...) and look at the row count, so two requests cannot both act on one booking.
  * reschedule() inserts the new booking and cancels the old one in ONE transaction. If the new slot
    is taken, everything rolls back and the old booking stays confirmed.
  * Nothing here reads the risk model (hard rule 2). Booking never depends on it.
  * close_slot() closes a free slot for the doctor's leave by INSERTing a row with status 'closed' in the doctor's own
    name. That row holds the slot through the same unique key, so a patient's booking and a doctor's closing of one
    slot are decided by the database exactly like two patients' bookings: one wins, the other gets SlotTaken.
    book() is unchanged and still checks nothing about availability. reopen_slot() cancels the row, freeing the slot.

Time rules (PLAN section 2): a slot whose start time has passed cannot be booked, and a started
appointment cannot be cancelled (it is marked completed or no_show instead).

Authorisation (who may act on which booking) is the route layer's job; these functions record the actor.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from . import clock
from .dberrors import is_duplicate_key, is_missing_reference
from .extensions import db
from .models import Booking, Doctor, Slot

OUTCOMES = ("completed", "no_show")
APPOINTMENT_STATUSES = ("confirmed", "completed", "no_show")
CLOSED = "closed"  # a slot closed by the doctor (leave): a row that holds the slot but is not an appointment


class BookingError(Exception):
    """Base class. `http_status` is what a route should return."""

    http_status = 400


class SlotTaken(BookingError):
    """Another active booking already holds this slot."""

    http_status = 409


class NotFound(BookingError):
    """The slot, booking or user does not exist."""

    http_status = 404


class InvalidState(BookingError):
    """The booking is not in a state that allows this action."""

    http_status = 409


class SlotInPast(BookingError):
    """The slot's start time has already passed."""

    http_status = 409


class AppointmentStarted(InvalidState):
    """The appointment has started, so it can no longer be cancelled or moved."""


class NotStarted(InvalidState):
    """An outcome can only be recorded once the appointment has started."""


class InvalidOutcome(BookingError):
    """The outcome is not one of completed / no_show."""

    http_status = 400


def _refuse(error: Exception) -> Exception:
    """End the read-only transaction opened by a pre-check, then hand back the error to raise."""
    db.session.rollback()
    return error


def _translate(exc: IntegrityError, slot_id: Optional[int]) -> Exception:
    """Turn a database constraint error into a service error; anything unrecognised is re-raised as is."""
    if is_duplicate_key(exc):
        return SlotTaken(f"slot {slot_id} is already booked")
    if is_missing_reference(exc):
        return NotFound("slot or user does not exist")
    return exc


def _slot_start(slot_id: int) -> Optional[datetime]:
    row = db.session.execute(sa.select(Slot.slot_date, Slot.slot_time).where(Slot.id == slot_id)).one_or_none()
    return None if row is None else datetime.combine(row.slot_date, row.slot_time)


def _booking_state(booking_id: int):
    """(status, slot start) of a booking, or None if it does not exist."""
    row = db.session.execute(
        sa.select(Booking.status, Slot.slot_date, Slot.slot_time)
        .join(Slot, Slot.id == Booking.slot_id)
        .where(Booking.id == booking_id)
    ).one_or_none()
    return None if row is None else (row.status, datetime.combine(row.slot_date, row.slot_time))


def book(slot_id: int, patient_id: int, reason: Optional[str] = None) -> Booking:
    """Confirm `slot_id` for `patient_id`. Raises SlotTaken if it is already held (409)."""
    start = _slot_start(slot_id)
    if start is None:
        raise _refuse(NotFound(f"slot {slot_id} does not exist"))
    if start <= clock.now():
        raise _refuse(SlotInPast(f"slot {slot_id} has already started"))

    booking = Booking(slot_id=slot_id, patient_id=patient_id, status="confirmed", reason=reason)
    db.session.add(booking)
    try:
        db.session.commit()
    except IntegrityError as exc:
        db.session.rollback()
        raise _translate(exc, slot_id) from exc
    except Exception:
        db.session.rollback()
        raise
    return booking


def _update_statement(booking_id: int, allowed_from: tuple, **values):
    """UPDATE a booking only if it is still in one of `allowed_from`; the row count tells who won."""
    return (
        sa.update(Booking)
        .where(Booking.id == booking_id, Booking.status.in_(allowed_from))
        .values(**values)
        .execution_options(synchronize_session=False)
    )


def _cancel_statement(booking_id: int, actor_id: int):
    """Cancels a booking only if it is still confirmed; the generated column then frees its slot."""
    return _update_statement(booking_id, ("confirmed",), status="cancelled",
                             cancelled_at=sa.func.now(), cancelled_by=actor_id)


def cancel(booking_id: int, actor_id: int) -> Booking:
    """Cancel a confirmed booking that has not started, freeing its slot.

    Completed and no_show bookings keep their slot; a started appointment is marked, not cancelled.
    """
    state = _booking_state(booking_id)
    if state is None:
        raise _refuse(NotFound(f"booking {booking_id} does not exist"))
    status, start = state
    if status != "confirmed":
        raise _refuse(InvalidState(f"booking {booking_id} is {status}; only a confirmed booking can be cancelled"))
    if start <= clock.now():
        raise _refuse(AppointmentStarted(f"booking {booking_id} has already started and cannot be cancelled"))

    try:
        if db.session.execute(_cancel_statement(booking_id, actor_id)).rowcount != 1:
            raise _why_not_changeable(booking_id)  # another request changed it first
        db.session.commit()
    except IntegrityError as exc:  # actor_id is not a user
        db.session.rollback()
        raise _translate(exc, None) from exc
    except Exception:
        db.session.rollback()
        raise
    return db.session.get(Booking, booking_id)


def _why_not_changeable(booking_id: int) -> BookingError:
    status = db.session.execute(sa.select(Booking.status).where(Booking.id == booking_id)).scalar_one_or_none()
    if status is None:
        return NotFound(f"booking {booking_id} does not exist")
    return InvalidState(f"booking {booking_id} is {status}; only a confirmed booking can be changed")


def reschedule(booking_id: int, new_slot_id: int, actor_id: int) -> Booking:
    """Move a confirmed booking to `new_slot_id`, atomically.

    Inserts the new booking and cancels the old one in one transaction. If the new slot is taken
    (SlotTaken) or anything else fails, nothing changes and the old booking stays confirmed.
    """
    old = db.session.get(Booking, booking_id)
    if old is None:
        raise _refuse(NotFound(f"booking {booking_id} does not exist"))
    old_slot_id, old_status, patient_id, reason = old.slot_id, old.status, old.patient_id, old.reason
    if old_status != "confirmed":
        raise _refuse(InvalidState(f"booking {booking_id} is {old_status}; only a confirmed booking can be moved"))
    if _slot_start(old_slot_id) <= clock.now():
        raise _refuse(AppointmentStarted(f"booking {booking_id} has already started and cannot be moved"))
    if old_slot_id == new_slot_id:
        raise _refuse(InvalidState(f"booking {booking_id} is already in slot {new_slot_id}"))
    new_start = _slot_start(new_slot_id)
    if new_start is None:
        raise _refuse(NotFound(f"slot {new_slot_id} does not exist"))
    if new_start <= clock.now():
        raise _refuse(SlotInPast(f"slot {new_slot_id} has already started"))

    new = Booking(slot_id=new_slot_id, patient_id=patient_id, status="confirmed", reason=reason)
    db.session.add(new)
    try:
        db.session.flush()  # INSERT the new booking: a duplicate key here means the slot is taken
        if db.session.execute(_cancel_statement(booking_id, actor_id)).rowcount != 1:
            raise _why_not_changeable(booking_id)  # the old booking changed under us
        db.session.commit()
    except IntegrityError as exc:
        db.session.rollback()
        raise _translate(exc, new_slot_id) from exc
    except Exception:
        db.session.rollback()
        raise
    return new


def mark_outcome(booking_id: int, outcome: str) -> Booking:
    """Record that a started appointment was completed or a no_show (a doctor or admin action).

    The booking keeps its slot either way. Marking can be corrected (completed <-> no_show);
    a cancelled booking cannot be marked.
    """
    if outcome not in OUTCOMES:
        raise InvalidOutcome(f"outcome must be one of {OUTCOMES}")
    state = _booking_state(booking_id)
    if state is None:
        raise _refuse(NotFound(f"booking {booking_id} does not exist"))
    status, start = state
    if status not in APPOINTMENT_STATUSES:  # cancelled, or a closed slot: neither is an appointment that can have an outcome
        raise _refuse(InvalidState(f"booking {booking_id} is {status}"))
    if start > clock.now():
        raise _refuse(NotStarted(f"booking {booking_id} has not started yet"))

    try:
        statement = _update_statement(booking_id, ("confirmed", "completed", "no_show"), status=outcome)
        if db.session.execute(statement).rowcount != 1:
            raise InvalidState(f"booking {booking_id} was cancelled while you were marking it")
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise
    return db.session.get(Booking, booking_id)


def close_slot(slot_id: int, actor_id: int) -> Booking:
    """Close a free slot that has not started (the doctor is unavailable). Raises SlotTaken if a patient holds it or it is
    already closed. The row is held in the doctor's own name whoever closes it (the doctor or an admin)."""
    row = db.session.execute(
        sa.select(Slot.slot_date, Slot.slot_time, Doctor.user_id).join(Doctor, Doctor.id == Slot.doctor_id).where(Slot.id == slot_id)
    ).one_or_none()
    if row is None:
        raise _refuse(NotFound(f"slot {slot_id} does not exist"))
    if datetime.combine(row.slot_date, row.slot_time) <= clock.now():
        raise _refuse(SlotInPast(f"slot {slot_id} has already started"))

    closure = Booking(slot_id=slot_id, patient_id=row.user_id, status=CLOSED, reason=f"Closed for leave (by user {actor_id})")
    db.session.add(closure)
    try:
        db.session.commit()  # the unique key on confirmed_slot_id decides: a booked or closed slot is a duplicate
    except IntegrityError as exc:
        db.session.rollback()
        raise _translate(exc, slot_id) from exc
    except Exception:
        db.session.rollback()
        raise
    return closure


def reopen_slot(slot_id: int, actor_id: int) -> None:
    """Reopen a closed slot. Only a row with status 'closed' is touched, so this can never cancel a patient's booking."""
    if _slot_start(slot_id) is None:
        raise _refuse(NotFound(f"slot {slot_id} does not exist"))
    statement = (
        sa.update(Booking)
        .where(Booking.slot_id == slot_id, Booking.status == CLOSED)
        .values(status="cancelled", cancelled_at=sa.func.now(), cancelled_by=actor_id)
        .execution_options(synchronize_session=False)
    )
    try:
        if db.session.execute(statement).rowcount != 1:
            raise InvalidState(f"slot {slot_id} is not closed")
        db.session.commit()
    except IntegrityError as exc:  # actor_id is not a user
        db.session.rollback()
        raise _translate(exc, None) from exc
    except Exception:
        db.session.rollback()
        raise
