"""Junk in, no server error out: every route, every role, awkward input.

A route can answer 200, 302, 400, 403, 404, 405 or 409 to a bad request, but never 500. The cases are the usual ways
real input goes wrong: ids that are zero or enormous, dates at the ends of the calendar, impossible dates, page numbers
that are negative or absurd, very long text, control characters, markup and SQL.
"""
from __future__ import annotations

import pytest

ROLES = ("anonymous", "patient", "doctor", "admin")
HUGE = "9" * 30
BAD_DATES = ["", "abc", "0000-00-00", "0001-01-01", "9999-12-31", "2030-02-30", "1000-01-01", "2030-1-7", "٢٠٣٠-٠١-٠٧"]
BAD_PAGES = ["0", "-1", "abc", HUGE, "1.5", ""]
JUNK = ["", "abc", "0", "-1", HUGE, "x" * 5000, "9999-12-31", "0001-01-01", "25:61", "\x00", "💥", "<script>alert(1)</script>",
        "'; DROP TABLE users;--", "2030-02-30", " "]
FORM_FIELDS = ["slot_id", "new_slot_id", "reason", "date", "start_time", "end_time", "date_from", "date_to", "skip_weekends",
               "name", "email", "password", "specialization", "slot_minutes", "outcome", "sex", "date_of_birth"]


def login_as(role, scene, world, client):
    if role == "anonymous":
        client.get("/login")
    else:
        user = {"patient": scene.alice, "doctor": scene.doctor_user, "admin": scene.admin}[role]
        world.login(client, user)
    with client.session_transaction() as sess:
        return sess["_csrf"]


ODD_TEXT = ["nope", "..", "x" * 3000, "💥", "%00", "thresholds.png", "a b"]


def urls_for(app, scene):
    """(method, url) for every non-static route, with each path value replaced by a valid, an odd and an enormous one."""
    valid = {"booking_id": scene.booking, "doctor_id": scene.doctor, "slug": "thresholds"}
    adapter = app.url_map.bind("localhost")
    found = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint == "static":
            continue
        methods = sorted(rule.methods & {"GET", "POST"})
        argument_sets = [dict(valid)]
        for name in rule.arguments:
            odd = ODD_TEXT if name == "slug" else [0, int(HUGE)]
            argument_sets += [{**valid, name: value} for value in odd]
        for arguments in argument_sets:
            url = adapter.build(rule.endpoint, {k: v for k, v in arguments.items() if k in rule.arguments})
            for method in methods:
                found.append((method, url))
    return sorted(set(found))


@pytest.mark.parametrize("role", ROLES)
def test_no_route_answers_500_to_awkward_query_strings(role, app, scene, world, client):
    app.config["PROPAGATE_EXCEPTIONS"] = False
    login_as(role, scene, world, client)
    failures = []
    for method, url in urls_for(app, scene):
        if method != "GET":
            continue
        variants = [{}]
        variants += [{"date": value} for value in BAD_DATES] + [{"page": value} for value in BAD_PAGES]
        variants += [{"status": value} for value in ("x", "confirmed", "'; DROP TABLE users;--", "💥")]
        variants += [{"date": "2030-01-07", "page": "2", "status": "no_show"}]
        for query in variants:
            response = client.get(url, query_string=query)
            if response.status_code >= 500:
                failures.append(f"GET {url} {query} -> {response.status_code}")
    assert not failures, "\n".join(failures[:15])


@pytest.mark.parametrize("role", ROLES)
def test_no_route_answers_500_to_awkward_form_posts(role, app, scene, world, client):
    app.config["PROPAGATE_EXCEPTIONS"] = False
    token = login_as(role, scene, world, client)
    failures = []
    for method, url in urls_for(app, scene):
        if method != "POST" or url == "/logout":
            continue
        for junk in JUNK:
            form = {name: junk for name in FORM_FIELDS}
            form["_csrf"] = token
            response = client.post(url, data=form)
            if response.status_code >= 500:
                failures.append(f"POST {url} all fields {junk[:20]!r} -> {response.status_code}")
        # real-looking values with one field awkward at a time
        base = {"slot_id": str(scene.slots[1]), "new_slot_id": str(scene.slots[2]), "date": "2030-01-08", "start_time": "09:00",
                "end_time": "10:00", "date_from": "2030-01-08", "date_to": "2030-01-09", "name": "Test Person",
                "email": "someone.else@example.test", "password": "longenough1", "specialization": "General",
                "slot_minutes": "15", "outcome": "completed", "sex": "F", "date_of_birth": "1990-01-01", "reason": "check-up"}
        for field in base:
            for junk in JUNK:
                response = client.post(url, data={**base, field: junk, "_csrf": token})
                if response.status_code >= 500:
                    failures.append(f"POST {url} {field}={junk[:20]!r} -> {response.status_code}")
    assert not failures, "\n".join(failures[:15])


def test_a_missing_or_wrong_csrf_token_is_refused_everywhere(app, scene, world, client):
    app.config["PROPAGATE_EXCEPTIONS"] = False
    world.login(client, scene.admin)
    for method, url in urls_for(app, scene):
        if method == "POST":
            for data in ({}, {"_csrf": "wrong"}, {"_csrf": ""}):
                assert client.post(url, data=data).status_code == 400, (url, data)
