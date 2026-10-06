"""End to end through the real HTML forms: register, log in, book, cancel, book again.
Also the "slot just taken" path and password hashing."""
from __future__ import annotations

import re

from app.extensions import db
from app.models import Booking, PatientProfile, User
from helpers import PASSWORD, form_token, slot_ids, text

EMAIL = "new.patient@example.test"


def test_register_login_book_cancel_book_again(app, world, client):
    doctor = world.doctor("Pediatrics")
    world.slots(doctor, 3)
    day_url = f"/patient/doctors/{doctor}?date=2030-01-07"

    # register
    response = client.post("/register", data={
        "name": "Nina Newpatient", "email": EMAIL, "password": PASSWORD,
        "date_of_birth": "1990-05-17", "sex": "F", "_csrf": form_token(text(client.get("/register")))})
    assert response.status_code == 302 and response.headers["Location"].endswith("/login")
    user = db.session.query(User).filter_by(email=EMAIL).one()
    profile = db.session.query(PatientProfile).filter_by(user_id=user.id).one()
    assert (user.role, str(profile.date_of_birth), profile.sex) == ("patient", "1990-05-17", "F")

    # log in
    response = client.post("/login", data={
        "email": EMAIL, "password": PASSWORD, "_csrf": form_token(text(client.get("/login")))})
    assert response.status_code == 302 and response.headers["Location"].endswith("/patient/")

    # the doctor is listed, with the specialization
    home = text(client.get("/patient/"))
    assert "Pediatrics" in home and f"/patient/doctors/{doctor}" in home

    # book the first free slot
    page = text(client.get(day_url))
    offered = slot_ids(page)
    assert len(offered) == 3
    first = offered[0]
    response = client.post("/patient/book", data={
        "slot_id": first, "reason": "annual check-up", "_csrf": form_token(page)})
    assert response.status_code == 302 and response.headers["Location"].endswith("/patient/bookings")

    # it shows under my appointments and is no longer offered
    mine = text(client.get("/patient/bookings"))
    assert "annual check-up" in mine and "Confirmed" in mine
    assert first not in slot_ids(text(client.get(day_url)))

    # cancel it
    booking_id = int(re.search(r"/patient/bookings/(\d+)/cancel", mine).group(1))
    response = client.post(f"/patient/bookings/{booking_id}/cancel", data={"_csrf": form_token(mine)})
    assert response.status_code == 302
    assert "Cancelled" in text(client.get("/patient/bookings"))

    # the slot is free again; book it again
    page = text(client.get(day_url))
    assert first in slot_ids(page)
    response = client.post("/patient/book", data={"slot_id": first, "_csrf": form_token(page)})
    assert response.status_code == 302

    db.session.expire_all()
    statuses = sorted(b.status for b in db.session.query(Booking).filter_by(slot_id=first))
    assert statuses == ["cancelled", "confirmed"]


def test_a_slot_taken_a_moment_ago_shows_a_message_with_409_and_a_fresh_list(scene, world, client):
    world.login(client, scene.bob)
    stale_page = text(client.get(f"/patient/doctors/{scene.doctor}?date={scene.day}"))
    assert scene.slots[1] in slot_ids(stale_page)

    # someone else books slot 1 after bob loaded the page
    from app import services
    services.book(scene.slots[1], scene.alice)

    response = client.post("/patient/book", data={"slot_id": scene.slots[1], "_csrf": form_token(stale_page)})
    assert response.status_code == 409
    body = text(response)
    assert "just taken" in body.lower()
    assert scene.slots[1] not in slot_ids(body)  # the refreshed list no longer offers it
    assert scene.slots[2] in slot_ids(body)


def test_passwords_are_stored_hashed_and_login_checks_them(app, client):
    client.post("/register", data={
        "name": "Hash Check", "email": EMAIL, "password": PASSWORD, "date_of_birth": "1985-02-03",
        "sex": "M", "_csrf": form_token(text(client.get("/register")))})
    stored = db.session.query(User).filter_by(email=EMAIL).one().password_hash
    assert stored != PASSWORD and PASSWORD not in stored
    assert stored.split(":")[0] in {"pbkdf2", "scrypt"}

    bad = client.post("/login", data={"email": EMAIL, "password": "wrong password", "_csrf": form_token(text(client.get("/login")))})
    assert bad.status_code == 401
    good = client.post("/login", data={"email": EMAIL, "password": PASSWORD, "_csrf": form_token(text(client.get("/login")))})
    assert good.status_code == 302
