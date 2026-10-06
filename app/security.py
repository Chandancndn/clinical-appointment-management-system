"""Session login, CSRF protection, role checks and safe response headers.

* Login state is the signed session cookie holding `user_id`; `g.user` is loaded from it per request.
* Every session carries a random CSRF token. Every POST must send it back (form field `_csrf` or the
  X-CSRFToken header), otherwise the request is refused with 400 before any view runs.
* Every route that is not public is wrapped in roles_required(...). The decorator records the roles on
  the view so a test can prove no route was left unprotected.
"""
from __future__ import annotations

import hmac
import secrets
from functools import wraps

from flask import abort, flash, g, redirect, request, session, url_for

from . import accounts

ROLES = ("patient", "doctor", "admin")
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
HOME_ENDPOINT = {"patient": "patient.index", "doctor": "doctor.schedule", "admin": "admin.dashboard"}


def home_for(user) -> str:
    return url_for(HOME_ENDPOINT[user.role])


def roles_required(*roles: str):
    """Allow only logged-in users whose role is in `roles`. Anonymous visitors go to the login page."""
    assert roles and set(roles) <= set(ROLES), roles

    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if g.user is None:
                flash("Please log in to continue.", "info")
                return redirect(url_for("auth.login"))
            if g.user.role not in roles:
                abort(403)
            return view(*args, **kwargs)

        wrapped.required_roles = roles
        return wrapped

    return decorator


def start_session(user_id=None) -> None:
    """Replace the whole session (no fixation), give it a fresh CSRF token, and log `user_id` in if given."""
    session.clear()
    session["_csrf"] = secrets.token_urlsafe(32)
    if user_id is not None:
        session["user_id"] = user_id
        session.permanent = True


def _load_user():
    user_id = session.get("user_id")
    if user_id is None:
        return None
    user = accounts.get_user(user_id)
    if user is None:  # account was removed; drop the stale login
        session.pop("user_id", None)
    return user


def _check_csrf() -> None:
    expected = session.get("_csrf", "")
    sent = request.form.get("_csrf") or request.headers.get("X-CSRFToken") or ""
    if not expected or not hmac.compare_digest(expected, sent):
        abort(400, description="The form could not be verified. It may have expired.")


def init_security(app) -> None:
    @app.before_request
    def guard():
        if request.endpoint == "static":
            return None
        if "_csrf" not in session:
            session["_csrf"] = secrets.token_urlsafe(32)
        g.user = _load_user()
        if request.method in UNSAFE_METHODS:
            _check_csrf()
        return None

    @app.context_processor
    def template_globals():
        return {"csrf_token": lambda: session.get("_csrf", ""), "current_user": g.get("user")}

    @app.after_request
    def secure_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; frame-ancestors 'none'; form-action 'self'; base-uri 'self'")
        if g.get("user") is not None:
            response.headers.setdefault("Cache-Control", "no-store")
        return response
