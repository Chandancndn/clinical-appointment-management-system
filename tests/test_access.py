"""Role access: every role is blocked from the other roles' pages, and every route checks the role."""
from __future__ import annotations

import pytest

ROLES = ("patient", "doctor", "admin")
# Endpoints that need no login. Every other endpoint must declare which roles may use it.
PUBLIC_ENDPOINTS = {"static", "main.index", "auth.login", "auth.register"}


def pages(s):
    """role -> [(method, url)] of the pages only that role may use."""
    return {
        "patient": [
            ("GET", "/patient/"),
            ("GET", f"/patient/doctors/{s.doctor}?date={s.day}"),
            ("GET", "/patient/bookings"),
            ("GET", f"/patient/bookings/{s.booking}/reschedule"),
            ("POST", "/patient/book"),
            ("POST", f"/patient/bookings/{s.booking}/cancel"),
            ("POST", f"/patient/bookings/{s.booking}/reschedule"),
        ],
        "doctor": [
            ("GET", f"/doctor/?date={s.day}"),
            ("POST", "/doctor/slots"),
            ("POST", f"/doctor/bookings/{s.booking}/cancel"),
        ],
        "admin": [
            ("GET", "/admin/"),
            ("GET", "/admin/bookings"),
            ("GET", "/admin/doctors/new"),
            ("POST", "/admin/doctors/new"),
            ("GET", f"/admin/doctors/{s.doctor}/slots"),
            ("POST", f"/admin/doctors/{s.doctor}/slots"),
            ("POST", f"/admin/bookings/{s.booking}/cancel"),
        ],
    }


def request(client, method, url, token):
    if method == "GET":
        return client.get(url)
    return client.post(url, data={"_csrf": token})


def login_as(role, scene, world, client):
    user = {"patient": scene.alice, "doctor": scene.doctor_user, "admin": scene.admin}[role]
    world.login(client, user)
    with client.session_transaction() as sess:
        return sess["_csrf"]


@pytest.mark.parametrize("role", ROLES)
def test_each_role_is_blocked_from_the_other_roles_pages(role, scene, world, client):
    token = login_as(role, scene, world, client)
    wrongly_allowed = []
    for other, other_pages in pages(scene).items():
        if other == role:
            continue
        for method, url in other_pages:
            status = request(client, method, url, token).status_code
            if status != 403:
                wrongly_allowed.append((role, method, url, status))
    assert not wrongly_allowed, f"expected 403, got: {wrongly_allowed}"


@pytest.mark.parametrize("role", ROLES)
def test_each_role_can_open_its_own_pages(role, scene, world, client):
    login_as(role, scene, world, client)
    failed = []
    for method, url in pages(scene)[role]:
        if method == "GET":
            status = client.get(url).status_code
            if status != 200:
                failed.append((method, url, status))
    assert not failed, f"expected 200, got: {failed}"


def test_the_outcome_route_is_for_doctors_and_admins_only(scene, world, client):
    url = f"/doctor/bookings/{scene.booking}/outcome"
    token = login_as("patient", scene, world, client)
    assert client.post(url, data={"_csrf": token, "outcome": "completed"}).status_code == 403
    for role in ("doctor", "admin"):
        client.post("/logout", data={"_csrf": token})
        token = login_as(role, scene, world, client)
        # Allowed through the role check; the 2030 appointment hasn't started, so it is refused later.
        assert client.post(url, data={"_csrf": token, "outcome": "completed"}).status_code != 403


def test_anonymous_visitors_are_sent_to_login(scene, client):
    client.get("/login")
    with client.session_transaction() as sess:
        token = sess["_csrf"]
    for role_pages in pages(scene).values():
        for method, url in role_pages:
            response = request(client, method, url, token)
            assert response.status_code == 302, (method, url, response.status_code)
            assert response.headers["Location"].endswith("/login"), (method, url)


def test_every_route_declares_which_roles_may_use_it(app):
    """Structural check: no view can be added without a role decorator unless it is listed public."""
    checked = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint in PUBLIC_ENDPOINTS:
            continue
        view = app.view_functions[rule.endpoint]
        roles = getattr(view, "required_roles", None)
        assert roles, f"{rule.rule} ({rule.endpoint}) has no role check"
        assert set(roles) <= set(ROLES)
        checked.append(rule.endpoint)
    assert len(checked) >= 15, f"only {len(checked)} protected routes found: {checked}"


def test_logout_only_accepts_post(client):
    assert client.get("/logout").status_code == 405
