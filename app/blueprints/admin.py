"""Admin pages: overview, add doctors, generate slots for a date range, see and cancel any booking."""
from __future__ import annotations

from flask import Blueprint, abort, flash, g, redirect, render_template, request, url_for

from .. import accounts, clock, forms, queries, risk, scheduling, services
from ..security import roles_required

bp = Blueprint("admin", __name__, url_prefix="/admin")
admin_only = roles_required("admin")

PER_PAGE = 25
STATUSES = ("confirmed", "completed", "no_show", "cancelled")


@bp.get("/")
@admin_only
def dashboard():
    now = clock.now()
    upcoming = queries.upcoming_bookings(25, now)
    return render_template("admin/dashboard.html", counts=queries.admin_counts(now), doctors=queries.list_doctors(),
                           upcoming=upcoming, now=now, flags=risk.flags_for(b.id for b in upcoming))


@bp.get("/bookings")
@admin_only
def bookings():
    status = request.args.get("status") if request.args.get("status") in STATUSES else None
    page = forms.parse_int(request.args.get("page"), low=1) or 1
    rows, total = queries.bookings_page(page, PER_PAGE, status)
    return render_template("admin/bookings.html", rows=rows, total=total, page=page, per_page=PER_PAGE,
                           status=status, statuses=STATUSES, now=clock.now(),
                           last_page=max(1, -(-total // PER_PAGE)),
                           flags=risk.flags_for(r.id for r in rows if r.status != "cancelled"))


@bp.post("/bookings/<int:booking_id>/cancel")
@admin_only
def cancel(booking_id):
    queries.booking_row(booking_id) or abort(404)
    try:
        services.cancel(booking_id, g.user.id)
    except services.NotFound:
        abort(404)
    except services.InvalidState as error:
        flash(str(error), "error")
    else:
        flash("Booking cancelled; the slot is free again.", "success")
    return redirect(url_for("admin.bookings"))


@bp.route("/doctors/new", methods=["GET", "POST"])
@admin_only
def new_doctor():
    errors, status = [], 200
    if request.method == "POST":
        data, errors = forms.parse_new_doctor(request.form)
        status = 400
        if not errors:
            try:
                doctor = accounts.create_doctor(**data)
            except accounts.EmailTaken:
                errors, status = ["That email is already registered."], 409
            else:
                flash(f"Doctor account created for {data['name']}. Now generate their slots.", "success")
                return redirect(url_for("admin.generate_slots", doctor_id=doctor.id))
    return render_template("admin/new_doctor.html", form=request.form, errors=errors), status


@bp.route("/doctors/<int:doctor_id>/slots", methods=["GET", "POST"])
@admin_only
def generate_slots(doctor_id):
    doctor = queries.get_doctor(doctor_id) or abort(404)
    errors, status = [], 200
    if request.method == "POST":
        first, last = forms.parse_date(request.form.get("date_from")), forms.parse_date(request.form.get("date_to"))
        start, end, errors = forms.parse_time_window(request.form)
        if first is None or last is None:
            errors.append("Enter valid first and last dates.")
        status = 400
        if not errors:
            try:
                result = scheduling.generate_slots(doctor_id, first, last, start, end,
                                                   skip_weekends=request.form.get("skip_weekends") == "on")
            except scheduling.ScheduleError as error:
                errors = [str(error)]
            else:
                flash(f"Created {result.created} new slot(s); {result.existing} already existed.", "success")
                return redirect(url_for("admin.dashboard"))
    return render_template("admin/generate_slots.html", doctor=doctor, form=request.form,
                           errors=errors, today=clock.now().date()), status
