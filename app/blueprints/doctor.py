"""Doctor pages: own schedule, create slots for a day, mark outcomes, cancel on own schedule.

The outcome route is shared with admins (PLAN section 2): a doctor can only touch bookings on their
own schedule (anything else is a 404), an admin can mark any booking.
"""
from __future__ import annotations

from datetime import timedelta

from flask import Blueprint, abort, flash, g, redirect, render_template, request, url_for

from .. import clock, forms, queries, risk, scheduling, services
from ..security import roles_required

bp = Blueprint("doctor", __name__, url_prefix="/doctor")
doctor_only = roles_required("doctor")


def _my_profile():
    return queries.doctor_for_user(g.user.id) or abort(403)


def _schedule_page(doctor, day, errors=(), status=200, form=None):
    now = clock.now()
    rows = queries.doctor_schedule(doctor.id, day)
    return render_template(
        "doctor/schedule.html", doctor=doctor, day=day, now=now, errors=list(errors), form=form or {},
        previous_day=day - timedelta(days=1), next_day=day + timedelta(days=1),
        rows=rows, week=queries.week_summary(doctor.id, day), outcomes=services.OUTCOMES,
        flags=risk.flags_for(r.booking_id for r in rows if r.booking_id)), status  # advisory badges, staff only


@bp.get("/")
@doctor_only
def schedule():
    day = forms.parse_date(request.args.get("date")) or clock.now().date()
    return _schedule_page(_my_profile(), day)


@bp.post("/slots")
@doctor_only
def generate():
    doctor = _my_profile()
    day = forms.parse_date(request.form.get("date"))
    start, end, errors = forms.parse_time_window(request.form)
    if day is None:
        errors.append("Enter a valid date.")
    elif day < clock.now().date():
        errors.append("You can only create slots for today or a later date.")
    if errors:
        return _schedule_page(doctor, day or clock.now().date(), errors, 400, request.form)
    try:
        result = scheduling.generate_slots(doctor.id, day, day, start, end)
    except scheduling.ScheduleError as error:
        return _schedule_page(doctor, day, [str(error)], 400, request.form)
    flash(f"Created {result.created} new slot(s); {result.existing} already existed.", "success")
    return redirect(url_for("doctor.schedule", date=day.isoformat()))


@bp.post("/bookings/<int:booking_id>/outcome")
@roles_required("doctor", "admin")
def outcome(booking_id):
    is_admin = g.user.role == "admin"
    if is_admin:
        booking = queries.booking_row(booking_id)
    else:
        booking = queries.booking_for_doctor(booking_id, _my_profile().id)
    booking or abort(404)
    try:
        services.mark_outcome(booking_id, request.form.get("outcome", ""))
    except services.InvalidOutcome:
        abort(400)
    except services.NotFound:
        abort(404)
    except services.InvalidState as error:
        flash(str(error), "error")
    else:
        flash("Appointment marked.", "success")
    if is_admin:
        return redirect(url_for("admin.bookings"))
    return redirect(url_for("doctor.schedule", date=booking.slot_date.isoformat()))


@bp.post("/bookings/<int:booking_id>/cancel")
@doctor_only
def cancel(booking_id):
    booking = queries.booking_for_doctor(booking_id, _my_profile().id) or abort(404)
    try:
        services.cancel(booking_id, g.user.id)
    except services.NotFound:
        abort(404)
    except services.InvalidState as error:
        flash(str(error), "error")
    else:
        flash("Appointment cancelled; the slot is free again.", "success")
    return redirect(url_for("doctor.schedule", date=booking.slot_date.isoformat()))
