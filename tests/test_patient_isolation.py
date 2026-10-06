"""A patient asking for another patient's booking gets a 404, the same as for a booking that doesn't exist."""
from __future__ import annotations

from app.extensions import db
from app.models import Booking
from helpers import post, text

MISSING = 999_999


def status_of(booking_id):
    db.session.expire_all()
    return db.session.get(Booking, booking_id).status


def test_cancelling_someone_elses_booking_is_404_and_changes_nothing(scene, world, client):
    world.login(client, scene.bob)
    assert post(client, f"/patient/bookings/{scene.booking}/cancel").status_code == 404
    assert status_of(scene.booking) == "confirmed"


def test_rescheduling_someone_elses_booking_is_404_and_changes_nothing(scene, world, client):
    world.login(client, scene.bob)
    assert client.get(f"/patient/bookings/{scene.booking}/reschedule").status_code == 404
    response = post(client, f"/patient/bookings/{scene.booking}/reschedule", new_slot_id=scene.slots[1])
    assert response.status_code == 404
    booking = db.session.get(Booking, scene.booking)
    db.session.refresh(booking)
    assert (booking.status, booking.slot_id) == ("confirmed", scene.slots[0])


def test_a_missing_booking_looks_exactly_like_someone_elses(scene, world, client):
    world.login(client, scene.bob)
    theirs = post(client, f"/patient/bookings/{scene.booking}/cancel")
    missing = post(client, f"/patient/bookings/{MISSING}/cancel")
    assert theirs.status_code == missing.status_code == 404
    assert text(theirs) == text(missing)  # no hint that the booking exists


def test_my_appointments_lists_only_my_own(scene, world, client):
    world.login(client, scene.bob)
    page = text(client.get("/patient/bookings"))
    assert "Alice Patient" not in page
    assert f"/patient/bookings/{scene.booking}/" not in page


def test_the_owner_can_still_cancel(scene, world, client):
    world.login(client, scene.alice)
    assert post(client, f"/patient/bookings/{scene.booking}/cancel").status_code == 302
    assert status_of(scene.booking) == "cancelled"


def test_a_doctor_cannot_touch_another_doctors_booking(scene, world, client):
    other_user = world.doctor_user_ids[scene.other_doctor]
    world.login(client, other_user)
    assert post(client, f"/doctor/bookings/{scene.booking}/cancel").status_code == 404
    assert post(client, f"/doctor/bookings/{scene.booking}/outcome", outcome="completed").status_code == 404
    assert status_of(scene.booking) == "confirmed"
