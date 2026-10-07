"""A patient cannot hold two appointments at the same time with different doctors.

This is a convenience rule on top of the real one. The no-double-booking rule (one doctor, date and time, one active
booking) stays in the database, untouched. This one is checked in the page layer before the booking is attempted, so
it is best effort for one patient clicking twice at once, and nothing in app/services.py knows about it.
"""
from __future__ import annotations

from app import services
from app.extensions import db
from app.models import Booking
from helpers import post, text


def other_doctor_slots(world, scene, count=3):
    return world.slots(scene.other_doctor, count)  # the same date and times as the first doctor's slots


def booking_count(patient_id, status="confirmed"):
    db.session.expire_all()
    return db.session.query(Booking).filter_by(patient_id=patient_id, status=status).count()


def test_booking_a_second_doctor_at_the_same_time_is_refused(scene, world, client):
    clash = other_doctor_slots(world, scene)[0]  # 09:00, the time of Alice's existing booking
    world.login(client, scene.alice)
    response = post(client, "/patient/book", slot_id=clash)
    assert response.status_code == 409 and "already have an appointment at that time" in text(response)
    assert booking_count(scene.alice) == 1


def test_a_different_time_with_the_other_doctor_is_fine(scene, world, client):
    later = other_doctor_slots(world, scene)[1]  # 09:15
    world.login(client, scene.alice)
    assert post(client, "/patient/book", slot_id=later).status_code == 302
    assert booking_count(scene.alice) == 2


def test_a_cancelled_booking_no_longer_blocks_the_time(scene, world, client):
    clash = other_doctor_slots(world, scene)[0]
    world.login(client, scene.alice)
    assert post(client, f"/patient/bookings/{scene.booking}/cancel").status_code == 302
    assert post(client, "/patient/book", slot_id=clash).status_code == 302


def test_other_patients_are_not_affected(scene, world, client):
    clash = other_doctor_slots(world, scene)[0]
    world.login(client, scene.bob)
    assert post(client, "/patient/book", slot_id=clash).status_code == 302


def test_the_real_double_booking_rule_is_unchanged(scene, world, client):
    world.login(client, scene.bob)
    response = post(client, "/patient/book", slot_id=scene.slots[0])  # Alice already holds this exact slot
    assert response.status_code == 409 and "just taken" in text(response).lower()


def test_moving_to_a_time_that_clashes_with_another_appointment_is_refused(scene, world, client):
    other = other_doctor_slots(world, scene)
    second = services.book(scene.slots[1], scene.alice).id  # Alice: doctor 1 at 09:00 and 09:15
    world.login(client, scene.alice)
    response = post(client, f"/patient/bookings/{second}/reschedule", new_slot_id=other[0])  # 09:15 -> 09:00 elsewhere
    assert response.status_code == 409 and "already have an appointment at that time" in text(response)
    db.session.expire_all()
    assert db.session.get(Booking, second).status == "confirmed"


def test_moving_to_the_same_time_with_another_doctor_is_allowed(scene, world, client):
    other = other_doctor_slots(world, scene)
    world.login(client, scene.alice)  # her only booking is doctor 1 at 09:00; moving it to doctor 2 at 09:00 is not a clash
    assert post(client, f"/patient/bookings/{scene.booking}/reschedule", new_slot_id=other[0]).status_code == 302
    assert booking_count(scene.alice) == 1


def test_the_booking_service_itself_has_no_such_rule(scene, world):
    other = other_doctor_slots(world, scene)
    assert services.book(other[0], scene.alice).status == "confirmed"  # hard rule 1 is about the slot, nothing else


# ---- overlapping times, not only identical start times -------------------------------------------------------
def long_doctor_slots(world, count=4):
    """A doctor with 30-minute appointments whose slots start at 09:00, 09:15, ... on the same day as the others."""
    return world.slots(world.doctor("Cardiology", slot_minutes=30), count)


def test_a_longer_appointment_blocks_a_slot_that_starts_inside_it(scene, world, client):
    long_slots = long_doctor_slots(world)
    services.book(long_slots[0], scene.bob)  # Bob, 09:00 to 09:30 with the 30-minute doctor
    world.login(client, scene.bob)
    response = post(client, "/patient/book", slot_id=scene.slots[1])  # 09:15 to 09:30 with the first doctor
    assert response.status_code == 409 and "already have an appointment at that time" in text(response)
    assert booking_count(scene.bob) == 1


def test_a_slot_that_starts_when_the_other_ends_is_fine(scene, world, client):
    long_slots = long_doctor_slots(world)
    services.book(long_slots[0], scene.bob)  # 09:00 to 09:30
    world.login(client, scene.bob)
    assert post(client, "/patient/book", slot_id=scene.slots[2]).status_code == 302  # 09:30 to 09:45
    assert booking_count(scene.bob) == 2


def test_a_slot_that_ends_when_the_other_starts_is_fine(scene, world, client):
    long_slots = long_doctor_slots(world)
    services.book(long_slots[2], scene.bob)  # 09:30 to 10:00
    world.login(client, scene.bob)
    assert post(client, "/patient/book", slot_id=scene.slots[1]).status_code == 302  # 09:15 to 09:30
    assert post(client, "/patient/book", slot_id=scene.slots[3]).status_code == 409  # 09:45 to 10:00, inside the 30-minute one


def test_a_short_appointment_blocks_a_longer_one_that_starts_before_it_ends(scene, world, client):
    long_slots = long_doctor_slots(world)
    services.book(scene.slots[2], scene.bob)  # 09:30 to 09:45 with the 15-minute doctor
    world.login(client, scene.bob)
    assert post(client, "/patient/book", slot_id=long_slots[1]).status_code == 409  # 09:15 to 09:45 covers it
    assert post(client, "/patient/book", slot_id=long_slots[0]).status_code == 302  # 09:00 to 09:30 stops just before


def test_moving_a_booking_ignores_its_own_old_time_but_not_others(scene, world, client):
    long_slots = long_doctor_slots(world)
    mine = services.book(scene.slots[1], scene.bob).id  # 09:15 to 09:30
    services.book(long_slots[2], scene.bob)  # 09:30 to 10:00
    world.login(client, scene.bob)
    assert post(client, f"/patient/bookings/{mine}/reschedule", new_slot_id=scene.slots[3]).status_code == 409  # lands inside the 09:30 one
    assert post(client, f"/patient/bookings/{mine}/reschedule", new_slot_id=long_slots[0]).status_code == 302  # 09:00 to 09:30: only its own old time overlapped
