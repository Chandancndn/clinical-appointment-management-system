"""The account page: any logged-in user can change their own password."""
from __future__ import annotations

from datetime import timedelta

from flask import Blueprint, current_app, flash, g, redirect, render_template, request, url_for

from .. import accounts, clock, forms
from ..security import ROLES, roles_required, start_session
from ..throttle import minutes_text

bp = Blueprint("account", __name__, url_prefix="/account")


def _page(errors=(), status=200):
    return render_template("account/password.html", errors=list(errors)), status


@bp.get("/")
@roles_required(*ROLES)
def password():
    return _page()


@bp.post("/")
@roles_required(*ROLES)
def change_password():
    limit = current_app.config["LOGIN_MAX_FAILURES"]
    window = timedelta(minutes=current_app.config["LOGIN_WINDOW_MINUTES"])
    throttle, now, key = current_app.extensions["throttle"], clock.now(), ("password", g.user.id)
    wait = throttle.blocked_for(key, limit, window, now)
    if wait is not None:  # guessing the current password is throttled like guessing at login
        return _page([f"Too many wrong passwords. Try again in {minutes_text(wait)}."], 429)

    current = request.form.get("current_password") or ""
    new, confirm = request.form.get("new_password") or "", request.form.get("confirm_password") or ""
    if not accounts.password_is_right(g.user.id, current):
        throttle.record_failure(key, window, now)
        return _page(["The current password is not correct."], 400)
    errors = []
    if not forms.MIN_PASSWORD <= len(new) <= 128:
        errors.append(f"Choose a new password of {forms.MIN_PASSWORD} to 128 characters.")
    elif new != confirm:
        errors.append("The new password and its confirmation do not match.")
    elif new == current:
        errors.append("Choose a different password from the current one.")
    if errors:
        return _page(errors, 400)

    throttle.clear(key)
    accounts.change_password(g.user.id, new)
    start_session(g.user.id)  # a fresh session and CSRF token after a credential change
    flash("Your password has been changed.", "success")
    return redirect(url_for("account.password"))
