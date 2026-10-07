"""Brute-force protection: repeated wrong passwords lock the address-and-email pair for a while.

Five failures from one address for one email within fifteen minutes block further tries from that address for that email
(the correct password too), twenty failures from one address block that address, a success clears the count, other emails
and other addresses are unaffected, and an email that does not exist is throttled exactly like one that does, so the
lock reveals nothing about which accounts exist.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app import clock
from helpers import PASSWORD, csrf_token

START = datetime(2030, 1, 7, 9, 0)


@pytest.fixture
def now(monkeypatch):
    """A clock the test can move."""
    state = {"t": START}
    monkeypatch.setattr(clock, "now", lambda: state["t"])
    return state


def attempt(client, email, password, address="10.0.0.1"):
    return client.post("/login", data={"email": email, "password": password, "_csrf": csrf_token(client)},
                       environ_base={"REMOTE_ADDR": address})


def logged_in(client) -> bool:
    return client.get("/patient/").status_code == 200


def test_five_wrong_passwords_lock_out_even_the_correct_one(scene, world, client, now):
    email = world.emails[scene.alice]
    for _ in range(5):
        assert attempt(client, email, "wrong password").status_code == 401
    response = attempt(client, email, PASSWORD)
    assert response.status_code == 429 and "too many" in response.get_data(as_text=True).lower()
    assert not logged_in(client)


def test_the_lock_message_says_how_long_to_wait(scene, world, client, now):
    email = world.emails[scene.alice]
    for _ in range(5):
        attempt(client, email, "wrong password")
    now["t"] += timedelta(minutes=4)
    page = attempt(client, email, PASSWORD).get_data(as_text=True)
    assert "11 minutes" in page or "12 minutes" in page


def test_the_lock_ends_after_fifteen_minutes(scene, world, client, now):
    email = world.emails[scene.alice]
    for _ in range(5):
        attempt(client, email, "wrong password")
    now["t"] += timedelta(minutes=14, seconds=50)
    assert attempt(client, email, PASSWORD).status_code == 429
    now["t"] += timedelta(seconds=20)
    assert attempt(client, email, PASSWORD).status_code == 302 and logged_in(client)


def test_a_blocked_try_does_not_extend_the_lock(scene, world, client, now):
    email = world.emails[scene.alice]
    for _ in range(5):
        attempt(client, email, "wrong password")
    for _ in range(10):  # hammering during the lock changes nothing
        assert attempt(client, email, "wrong password").status_code == 429
    now["t"] += timedelta(minutes=15, seconds=1)
    assert attempt(client, email, PASSWORD).status_code == 302


def test_a_successful_login_clears_the_count(scene, world, client, now):
    email = world.emails[scene.alice]
    for _ in range(4):
        attempt(client, email, "wrong password")
    assert attempt(client, email, PASSWORD).status_code == 302
    client.post("/logout", data={"_csrf": csrf_token(client)})
    for _ in range(4):
        assert attempt(client, email, "wrong password").status_code == 401  # not 429: the count started again


def test_old_failures_age_out(scene, world, client, now):
    email = world.emails[scene.alice]
    for _ in range(4):
        attempt(client, email, "wrong password")
    now["t"] += timedelta(minutes=16)
    for _ in range(4):
        assert attempt(client, email, "wrong password").status_code == 401


def test_other_emails_and_other_addresses_are_not_locked(scene, world, client, now):
    email = world.emails[scene.alice]
    for _ in range(5):
        attempt(client, email, "wrong password")
    assert attempt(client, world.emails[scene.bob], PASSWORD).status_code == 302
    other = client.application.test_client()
    assert attempt(other, email, PASSWORD, address="10.0.0.9").status_code == 302  # locked out of nothing from elsewhere


def test_an_unknown_email_is_throttled_the_same_way(scene, world, client, now):
    for _ in range(5):
        assert attempt(client, "nobody@example.test", "whatever").status_code == 401
    assert attempt(client, "nobody@example.test", "whatever").status_code == 429


def test_one_address_trying_many_emails_is_stopped_too(scene, world, client, now):
    for i in range(20):
        assert attempt(client, f"guess{i}@example.test", "whatever").status_code == 401
    assert attempt(client, world.emails[scene.alice], PASSWORD).status_code == 429  # the whole address is blocked
    assert attempt(client.application.test_client(), world.emails[scene.alice], PASSWORD, address="10.0.0.9").status_code == 302


def test_the_limits_can_be_configured(scene, world, client, now, app):
    app.config["LOGIN_MAX_FAILURES"] = 2
    email = world.emails[scene.alice]
    attempt(client, email, "x")
    attempt(client, email, "x")
    assert attempt(client, email, PASSWORD).status_code == 429


def test_email_case_and_spaces_do_not_dodge_the_lock(scene, world, client, now):
    email = world.emails[scene.alice]
    for variant in (email.upper(), f" {email} ", email, email.title(), email):
        attempt(client, variant, "wrong password")
    assert attempt(client, email, PASSWORD).status_code == 429
