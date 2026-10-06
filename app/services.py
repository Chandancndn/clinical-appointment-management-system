"""Booking service layer: the only code that changes bookings.

Rules (CLAUDE.md hard rule 1):
  * book() just INSERTs. The database's unique key on bookings.confirmed_slot_id decides who gets
    a slot, and a duplicate-key error means "slot taken" (HTTP 409). Nothing here checks first
    and then inserts, because that leaves a gap two requests can slip through.
  * cancel() and reschedule() change a booking with a conditional UPDATE (WHERE status = 'confirmed')
    and look at the row count, so two requests cannot both act on the same booking.
  * reschedule() inserts the new booking and cancels the old one in ONE transaction. If the new slot
    is taken, everything rolls back and the old booking stays confirmed.
  * Nothing here reads the risk model (hard rule 2). Booking never depends on it.

Authorisation (who may cancel what) is the route layer's job; these functions record the actor.
"""
from __future__ import annotations

from typing import Optional

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from .extensions import db
from .models import Booking

MYSQL_DUPLICATE_ENTRY = 1062
MYSQL_NO_REFERENCED_ROW = 1452


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


def _is_duplicate_key(exc: IntegrityError) -> bool:
    code = exc.orig.args[0] if getattr(exc.orig, "args", None) else None
    return code == MYSQL_DUPLICATE_ENTRY or "UNIQUE constraint failed" in str(exc.orig)  # SQLite


def _is_missing_reference(exc: IntegrityError) -> bool:
    code = exc.orig.args[0] if getattr(exc.orig, "args", None) else None
    return code == MYSQL_NO_REFERENCED_ROW or "FOREIGN KEY constraint failed" in str(exc.orig)  # SQLite


def _translate(exc: IntegrityError, slot_id: Optional[int]) -> Exception:
    """Turn a database constraint error into a service error; anything unrecognised is re-raised as is."""
    if _is_duplicate_key(exc):
        return SlotTaken(f"slot {slot_id} is already booked")
    if _is_missing_reference(exc):
        return NotFound("slot or user does not exist")
    return exc


def book(slot_id: int, patient_id: int, reason: Optional[str] = None) -> Booking:
    """Confirm `slot_id` for `patient_id`. Raises SlotTaken if it is already held (409)."""
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


def _cancel_statement(booking_id: int, actor_id: int):
    """UPDATE that cancels a booking only if it is still confirmed; frees its slot via the generated column."""
    return (
        sa.update(Booking)
        .where(Booking.id == booking_id, Booking.status == "confirmed")
        .values(status="cancelled", cancelled_at=sa.func.now(), cancelled_by=actor_id)
        .execution_options(synchronize_session=False)
    )


def cancel(booking_id: int, actor_id: int) -> Booking:
    """Cancel a confirmed booking, freeing its slot. Completed and no_show bookings keep their slot."""
    try:
        cancelled = db.session.execute(_cancel_statement(booking_id, actor_id)).rowcount
        if cancelled != 1:
            raise _why_not_cancellable(booking_id)
        db.session.commit()
    except IntegrityError as exc:  # actor_id is not a user
        db.session.rollback()
        raise _translate(exc, None) from exc
    except Exception:
        db.session.rollback()
        raise
    return db.session.get(Booking, booking_id)


def _why_not_cancellable(booking_id: int) -> BookingError:
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
        raise NotFound(f"booking {booking_id} does not exist")
    if old.status != "confirmed":
        raise InvalidState(f"booking {booking_id} is {old.status}; only a confirmed booking can be changed")
    if old.slot_id == new_slot_id:
        raise InvalidState(f"booking {booking_id} is already in slot {new_slot_id}")

    new = Booking(slot_id=new_slot_id, patient_id=old.patient_id, status="confirmed", reason=old.reason)
    db.session.add(new)
    try:
        db.session.flush()  # INSERT the new booking: a duplicate key here means the slot is taken
        if db.session.execute(_cancel_statement(booking_id, actor_id)).rowcount != 1:
            raise _why_not_cancellable(booking_id)  # the old booking changed under us
        db.session.commit()
    except IntegrityError as exc:
        db.session.rollback()
        raise _translate(exc, new_slot_id) from exc
    except Exception:
        db.session.rollback()
        raise
    return new
