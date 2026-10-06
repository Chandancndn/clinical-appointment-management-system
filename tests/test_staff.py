"""Doctor and admin actions: outcomes, cancelling, adding doctors, the all-bookings list."""
from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import text as sql

from app import services
from app.extensions import db
from app.models import Booking, Doctor, User
from helpers import PASSWORD, post, text

PAST = date(2020, 1, 6)  # long gone, so no clock tricks are needed


@pytest.fixture
def past(world, scene):
    """Three past slots for the scene's doctor; alice has a confirmed (unmarked) booking in the first."""
    slots = world.slots(scene.doctor, 3, day=PAST)
    return slots, world.booking(slots[0], scene.alice)


def status_of(booking_id):
    db.session.expire_all()
    return db.session.get(Booking, booking_id).status


# marking outcomes -------------------------------------------------------------------------------
@pytest.mark.parametrize("outcome", ["completed", "no_show"])
def test_a_doctor_marks_a_past_appointment(scene, world, client, past, outcome):
    slots, booking = past
    world.login(client, scene.doctor_user)
    response = post(client, f"/doctor/bookings/{booking}/outcome", outcome=outcome)
    assert response.status_code == 302 and f"date={PAST.isoformat()}" in response.headers["Location"]
    assert status_of(booking) == outcome
    # the slot stays held: an outcome never frees it
    assert db.session.execute(
        sql("SELECT confirmed_slot_id FROM bookings WHERE id = :i"), {"i": booking}).scalar_one() == slots[0]


def test_the_schedule_page_offers_outcome_buttons_for_started_appointments(scene, world, client, past):
    world.login(client, scene.doctor_user)
    page = text(client.get(f"/doctor/?date={PAST.isoformat()}"))
    assert "Alice Patient" in page and "Mark completed" in page and "Mark no-show" in page
    assert "Cancel" not in page.split("<tbody>")[1]  # a started appointment is marked, not cancelled


def test_an_outcome_for_a_future_appointment_is_refused(scene, world, client):
    world.login(client, scene.doctor_user)
    post(client, f"/doctor/bookings/{scene.booking}/outcome", outcome="completed")  # 2030: not started
    assert status_of(scene.booking) == "confirmed"


def test_an_unknown_outcome_value_is_a_400(scene, world, client, past):
    world.login(client, scene.doctor_user)
    assert post(client, f"/doctor/bookings/{past[1]}/outcome", outcome="cancelled").status_code == 400
    assert status_of(past[1]) == "confirmed"


def test_an_admin_can_mark_any_booking(scene, world, client, past):
    world.login(client, scene.admin)
    response = post(client, f"/doctor/bookings/{past[1]}/outcome", outcome="no_show")
    assert response.status_code == 302 and response.headers["Location"].endswith("/admin/bookings")
    assert status_of(past[1]) == "no_show"


# cancelling -------------------------------------------------------------------------------------
def test_a_doctor_cancels_a_future_appointment_and_the_slot_is_free_again(scene, world, client):
    world.login(client, scene.doctor_user)
    assert post(client, f"/doctor/bookings/{scene.booking}/cancel").status_code == 302
    assert status_of(scene.booking) == "cancelled"
    assert services.book(scene.slots[0], scene.bob).status == "confirmed"


def test_a_doctor_cannot_cancel_a_started_appointment(scene, world, client, past):
    world.login(client, scene.doctor_user)
    post(client, f"/doctor/bookings/{past[1]}/cancel")
    assert status_of(past[1]) == "confirmed"


def test_an_admin_cancels_any_booking(scene, world, client):
    world.login(client, scene.admin)
    response = post(client, f"/admin/bookings/{scene.booking}/cancel")
    assert response.status_code == 302 and response.headers["Location"].endswith("/admin/bookings")
    assert status_of(scene.booking) == "cancelled"
    assert post(client, "/admin/bookings/999999/cancel").status_code == 404


# schedule view ----------------------------------------------------------------------------------
def test_the_schedule_shows_only_my_own_patients(scene, world, client, app):
    world.login(client, scene.doctor_user)
    assert "Alice Patient" in text(client.get(f"/doctor/?date={scene.day}"))
    other_client = app.test_client()
    world.login(other_client, world.doctor_user_ids[scene.other_doctor])
    assert "Alice Patient" not in text(other_client.get(f"/doctor/?date={scene.day}"))


# admin: add doctor ------------------------------------------------------------------------------
NEW = {"name": "Dr. Newly Added", "email": "dr.new@example.test", "password": PASSWORD,
       "specialization": "Cardiology", "slot_minutes": "20"}


def test_an_admin_adds_a_doctor_who_can_then_log_in(scene, world, client, app):
    world.login(client, scene.admin)
    response = post(client, "/admin/doctors/new", **NEW)
    assert response.status_code == 302 and "/slots" in response.headers["Location"]
    user = db.session.query(User).filter_by(email=NEW["email"]).one()
    doctor = db.session.query(Doctor).filter_by(user_id=user.id).one()
    assert (user.role, doctor.specialization, doctor.slot_minutes) == ("doctor", "Cardiology", 20)
    assert user.password_hash != PASSWORD

    fresh = app.test_client()
    login = post(fresh, "/login", email=NEW["email"], password=PASSWORD)
    assert login.status_code == 302 and login.headers["Location"].endswith("/doctor/")
    assert fresh.get("/doctor/").status_code == 200


def test_adding_a_doctor_with_a_taken_email_is_a_409_and_creates_nothing(scene, world, client):
    world.login(client, scene.admin)
    before = db.session.query(User).count()
    response = post(client, "/admin/doctors/new", **{**NEW, "email": world.emails[scene.alice]})
    assert response.status_code == 409 and "already registered" in text(response)
    assert db.session.query(User).count() == before
    assert db.session.query(Doctor).count() == 2


@pytest.mark.parametrize("field, value", [("password", "short"), ("slot_minutes", "3"),
                                          ("email", "not-an-email"), ("specialization", "")])
def test_adding_a_doctor_validates_input(scene, world, client, field, value):
    world.login(client, scene.admin)
    response = post(client, "/admin/doctors/new", **{**NEW, field: value})
    assert response.status_code == 400
    assert db.session.query(User).filter_by(email=NEW["email"]).count() == 0


def test_an_admin_generates_slots_for_a_date_range_skipping_weekends(scene, world, client):
    world.login(client, scene.admin)
    from app.models import Slot
    before = db.session.query(Slot).filter_by(doctor_id=scene.doctor).count()
    response = post(client, f"/admin/doctors/{scene.doctor}/slots", date_from="2031-03-07", date_to="2031-03-10",
                    start_time="09:00", end_time="10:00", skip_weekends="on")  # Fri..Mon
    assert response.status_code == 302
    db.session.expire_all()
    assert db.session.query(Slot).filter_by(doctor_id=scene.doctor).count() == before + 2 * 4
    assert post(client, "/admin/doctors/999999/slots", date_from="2031-03-07", date_to="2031-03-07",
                start_time="09:00", end_time="10:00").status_code == 404


# admin: lists -----------------------------------------------------------------------------------
def test_the_overview_and_booking_list_show_counts_and_filter_by_status(scene, world, client, past):
    world.login(client, scene.admin)
    overview = text(client.get("/admin/"))
    assert "Alice Patient" in overview  # next 25 appointments include alice's future booking
    everything = text(client.get("/admin/bookings"))
    assert everything.count("Alice Patient") == 2
    confirmed = text(client.get("/admin/bookings?status=cancelled"))
    assert "Alice Patient" not in confirmed and "No bookings match" in confirmed
    assert client.get("/admin/bookings?page=abc&status=bogus").status_code == 200  # bad params fall back
