"""Small helpers for tests that drive the app through its HTTP interface."""
from __future__ import annotations

import re

from werkzeug.security import generate_password_hash

PASSWORD = "correct horse battery"
# Cheap hash so tests stay fast; the app verifies any werkzeug hash format.
FAST_HASH = generate_password_hash(PASSWORD, method="pbkdf2:sha256:1000")


def csrf_token(client) -> str:
    """The client's current CSRF token (the app puts one in every session)."""
    client.get("/login")
    with client.session_transaction() as sess:
        return sess["_csrf"]


def post(client, url: str, **data):
    """POST with a valid CSRF token."""
    return client.post(url, data={**data, "_csrf": csrf_token(client)})


def form_token(html: str) -> str:
    """The CSRF token embedded in a rendered form."""
    match = re.search(r'name="_csrf" value="([^"]+)"', html)
    assert match, "page has no CSRF hidden field"
    return match.group(1)


def slot_ids(html: str) -> list[int]:
    """Slot ids offered as bookable buttons on a page."""
    return [int(x) for x in re.findall(r'name="(?:slot_id|new_slot_id)"\s+value="(\d+)"', html)]


def text(response) -> str:
    return response.get_data(as_text=True)
