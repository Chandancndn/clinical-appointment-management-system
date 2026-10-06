"""Patient actions over HTTP: reschedule, validation, registration errors, logout."""
from __future__ import annotations

from app import services
from app.extensions import db
from app.models import Booking, User
from helpers import PASSWORD, form_token, post, slot_ids, text


def row(booking_id):
    db.session.expire_all()
    return db.session.get(Booking, booking_id)


def test_rescheduling_moves_the_booking_and_frees_the_old_slot(scene, world, client):
    world.login(client, scene.alice)
    page = text(client.get(f"/patient/bookings/{scene.booking}/reschedule?date={scene.day}"))
    assert slot_ids(page) == scene.slots[1:]  # her own slot is shown as "Yours", not offered
    response = post(client, f"/patient/bookings/{scene.booking}/reschedule", new_slot_id=scene.slots[2])
    assert response.status_code == 302 and response.headers["Location"].endswith("/patient/bookings")
    assert row(scene.booking).status == "cancelled"
    assert services.book(scene.slots[0], scene.bob).status == "confirmed"  # old slot is free again
    mine = text(client.get("/patient/bookings"))
    assert "Cancelled" in mine and "Confirmed" in mine


def test_rescheduling_into_a_taken_slot_is_409_and_keeps_the_old_booking(scene, world, client):
    services.book(scene.slots[1], scene.bob)
    world.login(client, scene.alice)
    stale = text(client.get(f"/patient/bookings/{scene.booking}/reschedule?date={scene.day}"))
    assert scene.slots[1] not in slot_ids(stale)
    response = client.post(f"/patient/bookings/{scene.booking}/reschedule",
                           data={"new_slot_id": scene.slots[1], "_csrf": form_token(stale)})
    assert response.status_code == 409 and "just taken" in text(response).lower()
    assert row(scene.booking).status == "confirmed" and row(scene.booking).slot_id == scene.slots[0]
    assert slot_ids(text(response)) == scene.slots[2:]


def test_booking_an_unknown_slot_is_404_and_a_garbage_slot_id_is_400(scene, world, client):
    world.login(client, scene.bob)
    assert post(client, "/patient/book", slot_id=999999).status_code == 404
    assert post(client, "/patient/book", slot_id="abc").status_code == 400
    assert post(client, "/patient/book").status_code == 400
    assert post(client, f"/patient/bookings/{scene.booking}/reschedule", new_slot_id="x").status_code == 404


def test_a_patient_cannot_cancel_twice(scene, world, client):
    world.login(client, scene.alice)
    assert post(client, f"/patient/bookings/{scene.booking}/cancel").status_code == 302
    again = post(client, f"/patient/bookings/{scene.booking}/cancel")
    assert again.status_code == 409 and "cancelled" in text(again).lower()


def test_my_appointments_separates_upcoming_from_history(scene, world, client):
    world.login(client, scene.alice)
    services.cancel(scene.booking, actor_id=scene.alice)
    services.book(scene.slots[1], scene.alice, reason="second try")
    page = text(client.get("/patient/bookings"))
    upcoming, history = page.split("Past and cancelled")
    assert "second try" in upcoming and "Cancelled" not in upcoming
    assert "Cancelled" in history


# registration and sessions ------------------------------------------------------------------------
GOOD = {"name": "Reg Tester", "email": "reg@example.test", "password": PASSWORD,
        "date_of_birth": "1990-01-01", "sex": "F"}


def test_registration_validates_every_field(client):
    for field, value, message in [
        ("email", "nope", "valid email"), ("password", "short", "password of 8"),
        ("date_of_birth", "2999-01-01", "date of birth"), ("sex", "X", "Select a sex"),
        ("name", "   ", "Enter a name"),
    ]:
        response = post(client, "/register", **{**GOOD, field: value})
        assert response.status_code == 400 and message in text(response), (field, response.status_code)
    assert db.session.query(User).count() == 0


def test_registering_twice_with_one_email_is_a_409(client, world):
    assert post(client, "/register", **GOOD).status_code == 302
    again = post(client, "/register", **{**GOOD, "email": GOOD["email"].upper()})  # case-insensitive
    assert again.status_code == 409
    assert db.session.query(User).filter_by(email=GOOD["email"]).count() == 1


def test_logout_ends_the_session(scene, world, client):
    world.login(client, scene.alice)
    assert client.get("/patient/").status_code == 200
    assert post(client, "/logout").status_code == 302
    assert client.get("/patient/").status_code == 302  # back to the login page


def test_a_logged_in_user_is_sent_home_from_login_and_register(scene, world, client):
    world.login(client, scene.alice)
    for url in ("/login", "/register", "/"):
        response = client.get(url)
        assert response.status_code == 302 and response.headers["Location"].endswith("/patient/")


def test_unknown_email_and_wrong_password_look_the_same(scene, world, client):
    wrong_pw = post(client, "/login", email=world.emails[scene.alice], password="wrong")
    no_user = post(client, "/login", email="nobody@example.test", password="wrong")
    assert wrong_pw.status_code == no_user.status_code == 401
    assert "Invalid email or password" in text(wrong_pw) and "Invalid email or password" in text(no_user)
