"""Register (patients only), log in, log out."""
from __future__ import annotations

from flask import Blueprint, flash, g, redirect, render_template, request, url_for

from .. import accounts, clock, forms
from ..security import ROLES, home_for, roles_required, start_session

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


@bp.route("/login", methods=["GET", "POST"])
def login():
    if g.user is not None:
        return redirect(home_for(g.user))
    if request.method == "POST":
        user = accounts.authenticate(request.form.get("email") or "", request.form.get("password") or "")
        if user is not None:
            start_session(user.id)  # new session on login: no fixation, fresh CSRF token
            return redirect(home_for(user))
        return render_template("auth/login.html", form=request.form,
                               errors=["Invalid email or password."]), 401
    return render_template("auth/login.html", form={}, errors=[])


@bp.post("/logout")
@roles_required(*ROLES)
def logout():
    start_session()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))
