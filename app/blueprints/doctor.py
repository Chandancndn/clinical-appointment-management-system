"""Doctor pages: own schedule, create slots for a day, close and reopen slots (leave), mark outcomes, cancel on own schedule.

The outcome route is shared with admins (PLAN section 2): a doctor can only touch bookings on their
own schedule (anything else is a 404), an admin can mark any booking.
"""
from __future__ import annotations

from datetime import timedelta

from flask import Blueprint, abort, flash, g, redirect, render_template, request, url_for

from .. import clock, forms, queries, risk, scheduling, services, standby
from ..security import roles_required

bp = Blueprint("doctor", __name__, url_prefix="/doctor")
doctor_only = roles_required("doctor")


def _my_profile():
    return queries.doctor_for_user(g.user.id) or abort(403)


def _schedule_page(doctor, day, errors=(), status=200, form=None, leave_errors=()):
    now = clock.now()
    rows = queries.doctor_schedule(doctor.id, day)
    appointments = [r.booking_id for r in rows if r.booking_id and r.status != services.CLOSED]  # a closed slot is never scored
    return render_template(
        "doctor/schedule.html", doctor=doctor, day=day, now=now, errors=list(errors), form=form or {},
        leave_errors=list(leave_errors), previous_day=day - timedelta(days=1), next_day=day + timedelta(days=1),
        rows=rows, week=queries.week_summary(doctor.id, day), outcomes=services.OUTCOMES,
        waiting=standby.for_doctor_day(doctor.id, day),
        flags=risk.flags_for(appointments)), status  # advisory badges, staff only


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


@bp.post("/bookings/<dbid:booking_id>/outcome")
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


@bp.post("/bookings/<dbid:booking_id>/cancel")
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


def _my_slot(slot_id):
    """The slot if it is on this doctor's own schedule, else a 404 (never says whether it exists)."""
    slot = queries.slot_info(slot_id)
    if slot is None or slot.doctor_id != _my_profile().id:
        abort(404)
    return slot


@bp.post("/slots/<dbid:slot_id>/close")
@doctor_only
def close_slot(slot_id):
    slot = _my_slot(slot_id)
    try:
        services.close_slot(slot_id, g.user.id)
    except services.SlotTaken:
        flash("That slot has just been booked or is already closed, so nothing changed.", "error")
    except services.SlotInPast:
        flash("That time has already passed.", "error")
    except services.NotFound:
        abort(404)
    else:
        flash("Slot closed. Patients can no longer book it.", "success")
    return redirect(url_for("doctor.schedule", date=slot.slot_date.isoformat()))


@bp.post("/slots/<dbid:slot_id>/reopen")
@doctor_only
def reopen_slot(slot_id):
    slot = _my_slot(slot_id)
    try:
        services.reopen_slot(slot_id, g.user.id)
    except services.NotFound:
        abort(404)
    except services.InvalidState:
        flash("That slot is not closed.", "error")
    else:
        flash("Slot reopened. Patients can book it again.", "success")
    return redirect(url_for("doctor.schedule", date=slot.slot_date.isoformat()))


def leave_message(action, first, last, doctor_id, actor_id) -> str:
    """Close or reopen the doctor's slots in the range and say what happened. Shared with the admin page."""
    if action == "close":
        result = scheduling.close_free_slots(doctor_id, first, last, actor_id)
        message = f"Closed {result.closed} free slot(s)."
        if result.booked:
            message += (f" {result.booked} booked appointment(s) in that range are unchanged: "
                        "closing never cancels a patient, so cancel those yourself if needed.")
        return message
    return f"Reopened {scheduling.reopen_slots(doctor_id, first, last, actor_id)} slot(s)."


@bp.post("/leave")
@doctor_only
def leave():
    doctor = _my_profile()
    first, last, action, errors = forms.parse_leave(request.form)
    if not errors:
        try:
            flash(leave_message(action, first, last, doctor.id, g.user.id), "success")
            return redirect(url_for("doctor.schedule", date=first.isoformat()))
        except scheduling.ScheduleError as error:
            errors = [str(error)]
    return _schedule_page(doctor, first or clock.now().date(), status=400, leave_errors=errors)
