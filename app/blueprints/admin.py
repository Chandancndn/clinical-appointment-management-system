"""Admin pages: overview, add doctors, generate slots for a date range, see and cancel any booking."""
from __future__ import annotations

from flask import Blueprint, abort, current_app, flash, g, redirect, render_template, request, send_file, url_for

from .. import accounts, clock, forms, queries, research, risk, scheduling, services, standby
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
                           upcoming=upcoming, now=now, flags=risk.flags_for(b.id for b in upcoming),
                           risk_status=risk.status(), risk_monitor=risk.monitor(), standby_count=standby.count_waiting(now))


@bp.get("/bookings")
@admin_only
def bookings():
    status = request.args.get("status") if request.args.get("status") in STATUSES else None
    page = forms.parse_int(request.args.get("page"), low=1, high=forms.MAX_PAGE) or 1
    rows, total = queries.bookings_page(page, PER_PAGE, status)
    return render_template("admin/bookings.html", rows=rows, total=total, page=page, per_page=PER_PAGE,
                           status=status, statuses=STATUSES, now=clock.now(),
                           last_page=max(1, -(-total // PER_PAGE)),
                           flags=risk.flags_for(r.id for r in rows if r.status != "cancelled"))


@bp.post("/bookings/<dbid:booking_id>/cancel")
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


@bp.route("/doctors/<dbid:doctor_id>/slots", methods=["GET", "POST"])
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
                           errors=errors, leave_errors=[], today=clock.now().date()), status


@bp.post("/doctors/<dbid:doctor_id>/leave")
@admin_only
def leave(doctor_id):
    """Close or reopen a doctor's slots for a date range (the doctor's leave)."""
    from .doctor import leave_message

    doctor = queries.get_doctor(doctor_id) or abort(404)
    first, last, action, errors = forms.parse_leave(request.form)
    if not errors:
        try:
            flash(leave_message(action, first, last, doctor_id, g.user.id), "success")
            return redirect(url_for("admin.generate_slots", doctor_id=doctor_id))
        except scheduling.ScheduleError as error:
            errors = [str(error)]
    return render_template("admin/generate_slots.html", doctor=doctor, form={}, errors=[], leave_errors=errors,
                           today=clock.now().date()), 400


@bp.get("/research")
@admin_only
def research_page():
    config = current_app.config
    return render_template("admin/research.html",
                           page=research.load(config["RESULTS_DIR"], config["RISK_ARTIFACTS_DIR"], config["FIGURES_DIR"]))


@bp.get("/research/figures/<slug>.png")
@admin_only
def research_figure(slug):
    path = research.figure_path(slug, current_app.config["FIGURES_DIR"], current_app.config["RESULTS_DIR"]) or abort(404)
    return send_file(path, mimetype="image/png", max_age=300)
