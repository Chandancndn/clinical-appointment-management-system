"""The account page: every role can change their own password, and nothing else about the account."""
from __future__ import annotations

from datetime import datetime

import pytest

from app import clock
from app.extensions import db
from app.models import User
from helpers import PASSWORD, csrf_token, post, text

NEW = "a brand new passphrase"
ROLES = ("patient", "doctor", "admin")


def user_for(role, scene):
    return {"patient": scene.alice, "doctor": scene.doctor_user, "admin": scene.admin}[role]


def change(client, current=PASSWORD, new=NEW, confirm=None):
    return post(client, "/account/", current_password=current, new_password=new, confirm_password=new if confirm is None else confirm)


def stored_hash(user_id):
    db.session.expire_all()
    return db.session.get(User, user_id).password_hash


def can_log_in(app, email, password):
    other = app.test_client()
    return other.post("/login", data={"email": email, "password": password, "_csrf": csrf_token(other)}).status_code == 302


def test_the_account_page_needs_a_login(client):
    response = client.get("/account/")
    assert response.status_code == 302 and "/login" in response.headers["Location"]


@pytest.mark.parametrize("role", ROLES)
def test_every_role_sees_its_own_details_and_the_form(role, scene, world, client):
    world.login(client, user_for(role, scene))
    page = text(client.get("/account/"))
    assert world.emails[user_for(role, scene)] in page and "Change password" in page
    assert 'name="current_password"' in page and 'name="new_password"' in page and 'name="confirm_password"' in page
    assert ">Account<" in page  # the navigation links to it


@pytest.mark.parametrize("role", ROLES)
def test_changing_the_password_works_and_the_old_one_stops_working(role, scene, world, client, app):
    uid = user_for(role, scene)
    world.login(client, uid)
    before = stored_hash(uid)
    response = change(client)
    assert response.status_code == 302
    assert stored_hash(uid) != before and NEW not in stored_hash(uid)
    assert can_log_in(app, world.emails[uid], NEW)
    assert not can_log_in(app, world.emails[uid], PASSWORD)


def test_the_user_stays_logged_in_with_a_fresh_session_token(scene, world, client):
    world.login(client, scene.alice)
    with client.session_transaction() as sess:
        token_before = sess["_csrf"]
    change(client)
    with client.session_transaction() as sess:
        assert sess["_csrf"] != token_before
    assert client.get("/patient/").status_code == 200


@pytest.mark.parametrize("current,new,confirm,fragment", [
    ("wrong password", NEW, NEW, "current password"),
    (PASSWORD, NEW, NEW + "x", "do not match"),
    (PASSWORD, "short", "short", "8"),
    (PASSWORD, PASSWORD, PASSWORD, "different"),
    (PASSWORD, "x" * 129, "x" * 129, "128"),
])
def test_bad_changes_are_refused_with_a_reason_and_change_nothing(current, new, confirm, fragment, scene, world, client):
    world.login(client, scene.alice)
    before = stored_hash(scene.alice)
    response = change(client, current, new, confirm)
    assert response.status_code == 400 and fragment in text(response)
    assert stored_hash(scene.alice) == before


def test_a_post_without_the_csrf_token_is_refused(scene, world, client):
    world.login(client, scene.alice)
    before = stored_hash(scene.alice)
    response = client.post("/account/", data={"current_password": PASSWORD, "new_password": NEW, "confirm_password": NEW})
    assert response.status_code == 400 and stored_hash(scene.alice) == before


def test_guessing_the_current_password_is_throttled(scene, world, client, monkeypatch):
    monkeypatch.setattr(clock, "now", lambda: datetime(2030, 1, 7, 9, 0))
    world.login(client, scene.alice)
    before = stored_hash(scene.alice)
    for _ in range(5):
        assert change(client, "wrong password").status_code == 400
    response = change(client, PASSWORD)  # even the right one is refused while locked
    assert response.status_code == 429 and stored_hash(scene.alice) == before
    assert can_log_in(client.application, world.emails[scene.alice], PASSWORD)  # the old password still logs in


def test_only_the_password_can_be_changed_here(scene, world, client):
    world.login(client, scene.alice)
    page = text(client.get("/account/"))
    for field in ("role", "email", "name"):
        assert f'name="{field}"' not in page
    post(client, "/account/", current_password=PASSWORD, new_password=NEW, confirm_password=NEW, role="admin", email="x@y.test")
    db.session.expire_all()
    user = db.session.get(User, scene.alice)
    assert user.role == "patient" and user.email == world.emails[scene.alice]
