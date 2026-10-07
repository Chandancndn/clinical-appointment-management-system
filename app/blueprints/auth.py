"""Register (patients only), log in, log out."""
from __future__ import annotations

from datetime import timedelta

from flask import Blueprint, current_app, flash, g, redirect, render_template, request, url_for

from .. import accounts, clock, forms
from ..security import ROLES, home_for, roles_required, start_session
from ..throttle import minutes_text

bp = Blueprint("auth", __name__)


@bp.route("/register", methods=["GET", "POST"])
def register():
    if g.user is not None:
        return redirect(home_for(g.user))
    errors, status = [], 200
    if request.method == "POST":
        data, errors = forms.parse_registration(request.form, clock.now().date())
        status = 400
        if not errors:
            try:
                accounts.register_patient(**data)
            except accounts.EmailTaken:
                errors, status = ["That email is already registered. Try logging in instead."], 409
            else:
                flash("Account created. Please log in.", "success")
                return redirect(url_for("auth.login"))
    return render_template("auth/register.html", form=request.form, errors=errors), status


def _limits():
    """(failures per email and address, failures per address, window) from the config."""
    config = current_app.config
    return config["LOGIN_MAX_FAILURES"], config["LOGIN_MAX_FAILURES_PER_ADDRESS"], timedelta(minutes=config["LOGIN_WINDOW_MINUTES"])


@bp.route("/login", methods=["GET", "POST"])
def login():
    if g.user is not None:
        return redirect(home_for(g.user))
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        address = request.remote_addr or "unknown"
        pair, whole = ("login", address, email), ("login-address", address)
        per_pair, per_address, window = _limits()
        throttle, now = current_app.extensions["throttle"], clock.now()
        waits = [w for w in (throttle.blocked_for(pair, per_pair, window, now),
                             throttle.blocked_for(whole, per_address, window, now)) if w is not None]
        if waits:  # refused without looking at the password, so a blocked try cannot extend the block
            return render_template("auth/login.html", form=request.form,
                                   errors=[f"Too many failed attempts. Try again in {minutes_text(max(waits))}."]), 429
        user = accounts.authenticate(email, request.form.get("password") or "")
        if user is not None:
            throttle.clear(pair)
            start_session(user.id)  # new session on login: no fixation, fresh CSRF token
            return redirect(home_for(user))
        throttle.record_failure(pair, window, now)
        throttle.record_failure(whole, window, now)
        return render_template("auth/login.html", form=request.form,
                               errors=["Invalid email or password."]), 401
    return render_template("auth/login.html", form={}, errors=[])


@bp.post("/logout")
@roles_required(*ROLES)
def logout():
    start_session()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))
