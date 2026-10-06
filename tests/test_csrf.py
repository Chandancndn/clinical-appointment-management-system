"""Every POST needs a valid CSRF token; without one nothing happens."""
from __future__ import annotations

import pytest

from app.extensions import db
from app.models import Booking, User
from helpers import PASSWORD, csrf_token, text


def count(model):
    db.session.expire_all()
    return db.session.query(model).count()


def test_booking_without_a_token_is_rejected_and_creates_nothing(scene, world, client):
    world.login(client, scene.bob)
    before = count(Booking)
    response = client.post("/patient/book", data={"slot_id": scene.slots[1]})
    assert response.status_code == 400
    assert count(Booking) == before


def test_booking_with_a_wrong_token_is_rejected(scene, world, client):
    world.login(client, scene.bob)
    before = count(Booking)
    response = client.post("/patient/book", data={"slot_id": scene.slots[1], "_csrf": "not-the-token"})
    assert response.status_code == 400
    assert count(Booking) == before


def test_cancel_without_a_token_leaves_the_booking_confirmed(scene, world, client):
    world.login(client, scene.alice)
    assert client.post(f"/patient/bookings/{scene.booking}/cancel").status_code == 400
    db.session.expire_all()
    assert db.session.get(Booking, scene.booking).status == "confirmed"


def test_a_token_from_another_session_is_rejected(scene, world, app, client):
    """A token valid for someone else's session must not work for mine."""
    world.login(client, scene.bob)
    attacker = app.test_client()
    stolen = csrf_token(attacker)
    before = count(Booking)
    assert client.post("/patient/book", data={"slot_id": scene.slots[1], "_csrf": stolen}).status_code == 400
    assert count(Booking) == before


@pytest.mark.parametrize("url", ["/login", "/register", "/logout"])
def test_public_and_auth_posts_need_a_token_too(url, client):
    assert client.post(url, data={}).status_code == 400


def test_login_without_a_token_does_not_log_in(scene, world, client):
    response = client.post("/login", data={"email": world.emails[scene.alice], "password": PASSWORD})
    assert response.status_code == 400
    assert client.get("/patient/").status_code == 302  # still anonymous


def test_registration_without_a_token_creates_no_user(client):
    response = client.post("/register", data={
        "name": "No Token", "email": "no.token@example.test", "password": "longenough1",
        "date_of_birth": "1990-01-01", "sex": "F"})
    assert response.status_code == 400
    assert count(User) == 0


def test_a_valid_token_is_accepted(scene, world, client):
    world.login(client, scene.bob)
    token = csrf_token(client)
    response = client.post("/patient/book", data={"slot_id": scene.slots[1], "_csrf": token})
    assert response.status_code == 302


def test_forms_carry_the_token(scene, world, client):
    for url in ("/login", "/register"):
        assert 'name="_csrf"' in text(client.get(url))
    world.login(client, scene.bob)
    assert 'name="_csrf"' in text(client.get(f"/patient/doctors/{scene.doctor}?date={scene.day}"))
    assert 'name="_csrf"' in text(client.get("/patient/bookings"))


def test_get_requests_do_not_need_a_token(client):
    assert client.get("/login").status_code == 200
