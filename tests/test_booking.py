"""Booking core. The database, not Python, must stop double-booking (CLAUDE.md hard rule 1).

Runs on SQLite by default and on real MySQL with `pytest --engine mysql`.
"""
from __future__ import annotations

import threading

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app import services
from app.extensions import db


def scalar(sql: str, **params):
    """Read one value on a fresh connection, so MySQL snapshot isolation can never show stale rows."""
    with db.engine.connect() as conn:
        return conn.execute(text(sql), params).scalar_one()


def rows(sql: str, **params):
    with db.engine.connect() as conn:
        return conn.execute(text(sql), params).all()


def run_concurrently(app, jobs):
    """Run each job in its own thread and app context, all released at the same instant."""
    barrier = threading.Barrier(len(jobs))
    results = [None] * len(jobs)

    def target(i, job):
        with app.app_context():
            barrier.wait(timeout=30)
            results[i] = job()

    threads = [threading.Thread(target=target, args=(i, job)) for i, job in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads), "a booking thread hung"
    return results


def attempt(slot_id: int, patient_id: int) -> str:
    """One booking request, reduced to an outcome label. Unexpected errors are kept, not hidden."""
    try:
        services.book(slot_id, patient_id)
        return "booked"
    except services.SlotTaken:
        return "slot_taken"
    except Exception as exc:  # noqa: BLE001 - the test must see anything unexpected
        return f"error: {exc!r}"


# (a) two concurrent requests for the same slot ------------------------------------------------
def test_two_concurrent_requests_for_one_slot_give_one_winner(app, world):
    rounds = 20  # repeat on fresh slots: a race that is missing a constraint shows up in some rounds
    slots = world.slots(world.doctor(), rounds)
    first, second = world.patient(), world.patient()
    for slot in slots:
        outcomes = run_concurrently(app, [lambda s=slot: attempt(s, first), lambda s=slot: attempt(s, second)])
        assert sorted(outcomes) == ["booked", "slot_taken"], outcomes
        assert scalar("SELECT COUNT(*) FROM bookings WHERE slot_id = :s AND status = 'confirmed'", s=slot) == 1


# (b) six concurrent requests across 25 slots --------------------------------------------------
def test_six_concurrent_requests_across_25_slots_give_one_winner_per_slot(app, world):
    slots = world.slots(world.doctor(), 25)
    patients = [world.patient() for _ in range(6)]
    results = run_concurrently(app, [lambda p=p: [attempt(s, p) for s in slots] for p in patients])
    outcomes = [o for per_patient in results for o in per_patient]
    assert outcomes.count("booked") == 25, outcomes
    assert outcomes.count("slot_taken") == 6 * 25 - 25, outcomes
    per_slot = rows("SELECT slot_id, COUNT(*) FROM bookings WHERE status = 'confirmed' GROUP BY slot_id")
    assert len(per_slot) == 25
    assert all(count == 1 for _, count in per_slot)


# (c) the database itself rejects a second active booking --------------------------------------
INSERT_BOOKING = "INSERT INTO bookings (slot_id, patient_id, status) VALUES (:s, :p, :status)"


@pytest.mark.parametrize("status", ["confirmed", "completed", "no_show"])
def test_direct_sql_insert_of_a_second_active_booking_is_rejected(app, world, status):
    [slot] = world.slots(world.doctor(), 1)
    first, second = world.patient(), world.patient()
    services.book(slot, first)
    with pytest.raises(IntegrityError):
        db.session.execute(text(INSERT_BOOKING), {"s": slot, "p": second, "status": status})
        db.session.commit()
    db.session.rollback()
    assert scalar("SELECT COUNT(*) FROM bookings WHERE slot_id = :s", s=slot) == 1


def test_direct_sql_insert_of_cancelled_bookings_is_allowed_beside_an_active_one(app, world):
    [slot] = world.slots(world.doctor(), 1)
    first, second = world.patient(), world.patient()
    services.book(slot, first)
    for _ in range(2):  # any number of cancelled rows may sit beside the active one
        db.session.execute(text(INSERT_BOOKING), {"s": slot, "p": second, "status": "cancelled"})
    db.session.commit()
    assert scalar("SELECT COUNT(*) FROM bookings WHERE slot_id = :s", s=slot) == 3


def test_direct_sql_reactivating_a_cancelled_booking_is_rejected_when_the_slot_is_taken(app, world):
    [slot] = world.slots(world.doctor(), 1)
    first, second = world.patient(), world.patient()
    old = services.book(slot, first)
    services.cancel(old.id, actor_id=first)
    services.book(slot, second)
    with pytest.raises(IntegrityError):
        db.session.execute(text("UPDATE bookings SET status = 'confirmed' WHERE id = :i"), {"i": old.id})
        db.session.commit()
    db.session.rollback()


# (d) cancelling frees the slot ----------------------------------------------------------------
def test_cancelling_frees_the_slot_and_it_can_be_booked_again(app, world):
    [slot] = world.slots(world.doctor(), 1)
    first, second = world.patient(), world.patient()

    booking = services.book(slot, first)
    with pytest.raises(services.SlotTaken):
        services.book(slot, second)

    cancelled = services.cancel(booking.id, actor_id=first)
    assert cancelled.status == "cancelled"
    assert cancelled.cancelled_by == first
    assert cancelled.cancelled_at is not None
    assert cancelled.confirmed_slot_id is None

    rebooked = services.book(slot, second)
    assert rebooked.status == "confirmed"
    assert rebooked.confirmed_slot_id == slot
    with pytest.raises(services.SlotTaken):
        services.book(slot, first)

    services.cancel(rebooked.id, actor_id=second)
    services.book(slot, first)  # and again: cancelled rows pile up beside the active one
    assert scalar("SELECT COUNT(*) FROM bookings WHERE slot_id = :s AND status = 'cancelled'", s=slot) == 2


# (e) completed and no_show keep the slot ------------------------------------------------------
@pytest.mark.parametrize("outcome", ["completed", "no_show"])
def test_completed_and_no_show_bookings_keep_the_slot(app, world, outcome):
    [slot] = world.slots(world.doctor(), 1)
    first, second = world.patient(), world.patient()
    booking = services.book(slot, first)

    # Marking an outcome is a doctor/admin screen in M2; here it is a plain status update.
    db.session.execute(text("UPDATE bookings SET status = :o WHERE id = :i"), {"o": outcome, "i": booking.id})
    db.session.commit()

    assert scalar("SELECT confirmed_slot_id FROM bookings WHERE id = :i", i=booking.id) == slot
    with pytest.raises(services.SlotTaken):
        services.book(slot, second)
    with pytest.raises(services.InvalidState):  # only a confirmed booking can be cancelled
        services.cancel(booking.id, actor_id=first)
    with pytest.raises(services.SlotTaken):  # the failed cancel did not free it
        services.book(slot, second)


# (f) reschedule is atomic ---------------------------------------------------------------------
def test_reschedule_moves_the_booking_and_frees_the_old_slot(app, world):
    old_slot, new_slot = world.slots(world.doctor(), 2)
    patient, other = world.patient(), world.patient()
    booking = services.book(old_slot, patient, reason="check-up")

    moved = services.reschedule(booking.id, new_slot, actor_id=patient)

    assert moved.slot_id == new_slot
    assert moved.patient_id == patient
    assert moved.status == "confirmed"
    assert moved.reason == "check-up"
    old = db.session.get(type(moved), booking.id)
    assert old.status == "cancelled"
    assert old.cancelled_by == patient
    services.book(old_slot, other)  # the old slot is free again
    with pytest.raises(services.SlotTaken):
        services.book(new_slot, other)


def test_reschedule_into_a_taken_slot_changes_nothing(app, world):
    slot_a, slot_b = world.slots(world.doctor(), 2)
    patient, other = world.patient(), world.patient()
    mine = services.book(slot_a, patient)
    services.book(slot_b, other)

    with pytest.raises(services.SlotTaken):
        services.reschedule(mine.id, slot_b, actor_id=patient)

    db.session.expire_all()
    still_mine = db.session.get(type(mine), mine.id)
    assert still_mine.status == "confirmed"
    assert still_mine.slot_id == slot_a
    assert still_mine.cancelled_at is None
    assert scalar("SELECT COUNT(*) FROM bookings WHERE patient_id = :p", p=patient) == 1
    with pytest.raises(services.SlotTaken):  # slot_a is still held by the old booking
        services.book(slot_a, other)


def test_reschedule_to_the_same_slot_is_rejected(app, world):
    [slot] = world.slots(world.doctor(), 1)
    patient = world.patient()
    booking = services.book(slot, patient)
    with pytest.raises(services.InvalidState):
        services.reschedule(booking.id, slot, actor_id=patient)
    assert services.cancel(booking.id, actor_id=patient).status == "cancelled"


# service error mapping ------------------------------------------------------------------------
def test_unknown_slot_booking_or_user_is_not_found_not_slot_taken(app, world):
    [slot] = world.slots(world.doctor(), 1)
    patient = world.patient()
    with pytest.raises(services.NotFound):
        services.book(999_999, patient)
    with pytest.raises(services.NotFound):
        services.book(slot, 999_999)
    with pytest.raises(services.NotFound):
        services.cancel(999_999, actor_id=patient)
    with pytest.raises(services.NotFound):
        services.reschedule(999_999, slot, actor_id=patient)


def test_http_status_of_each_service_error(app):
    assert services.SlotTaken.http_status == 409
    assert services.InvalidState.http_status == 409
    assert services.NotFound.http_status == 404
