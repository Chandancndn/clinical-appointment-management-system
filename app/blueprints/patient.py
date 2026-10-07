"""Patient pages: browse doctors, book, view, cancel and reschedule own appointments."""
from __future__ import annotations

from datetime import datetime, timedelta

from flask import Blueprint, abort, flash, g, redirect, render_template, request, url_for

from .. import clock, forms, queries, risk, services
from ..security import roles_required

bp = Blueprint("patient", __name__, url_prefix="/patient")
patient_only = roles_required("patient")

DAYS_SHOWN = 14
_CLASH = "You already have an appointment at that time with {doctor}. Cancel or move it first, or pick another time."


def _slot_page(template: str, doctor, day, error=None, status=200, **context):
    """Render a doctor's slots for one day, always from fresh data."""
    now = clock.now()
    return render_template(
        template, doctor=doctor, day=day, error=error, today=now.date(),
        slots=queries.day_slots(doctor.id, day, now),
        days=queries.upcoming_days(doctor.id, now.date(), DAYS_SHOWN, now),
        previous_day=day - timedelta(days=1), next_day=day + timedelta(days=1), **context), status


def _chosen_day(doctor_id: int):
    """The ?date= value, else the first day that still has a free slot, else today."""
    day = forms.parse_date(request.args.get("date"))
    if day is not None:
        return day
    now = clock.now()
    first_free = next((d.day for d in queries.upcoming_days(doctor_id, now.date(), DAYS_SHOWN, now) if d.free), None)
    return first_free or now.date()


def _bookings_page(error=None, status=200):
    now = clock.now()
    rows = queries.patient_bookings(g.user.id)
    upcoming = [r for r in rows if r.status == "confirmed" and datetime.combine(r.slot_date, r.slot_time) > now]
    history = [r for r in reversed(rows) if r not in upcoming]
    return render_template("patient/bookings.html", upcoming=upcoming, history=history,
                           now=now, error=error), status


@bp.get("/")
@patient_only
def index():
    return render_template("patient/doctors.html", doctors=queries.list_doctors())


@bp.get("/doctors/<dbid:doctor_id>")
@patient_only
def doctor_slots(doctor_id):
    doctor = queries.get_doctor(doctor_id) or abort(404)
    return _slot_page("patient/doctor_slots.html", doctor, _chosen_day(doctor_id))


@bp.post("/book")
@patient_only
def book():
    slot_id = forms.parse_int(request.form.get("slot_id"), low=1)
    if slot_id is None:
        abort(400)
    slot = queries.slot_info(slot_id) or abort(404)
    reason = (request.form.get("reason") or "").strip()[:500] or None
    clash = queries.patient_clash(g.user.id, slot_id)
    if clash is not None:
        return _slot_page("patient/doctor_slots.html", queries.get_doctor(slot.doctor_id), slot.slot_date, status=409,
                          error=_CLASH.format(doctor=clash.doctor_name))
    try:
        booking = services.book(slot_id, g.user.id, reason)
    except services.SlotTaken:
        doctor = queries.get_doctor(slot.doctor_id)
        return _slot_page("patient/doctor_slots.html", doctor, slot.slot_date, status=409,
                          error="That slot was just taken. The list below is up to date; please pick another.")
    except services.SlotInPast:
        doctor = queries.get_doctor(slot.doctor_id)
        return _slot_page("patient/doctor_slots.html", doctor, slot.slot_date, status=409,
                          error="That time has already passed. Please pick a later slot.")
    except services.NotFound:
        abort(404)
    risk.score_after_commit(booking)  # advisory and staff-only: runs after the commit, can never change the outcome
    flash("Your appointment is booked.", "success")
    return redirect(url_for("patient.bookings"))


@bp.get("/bookings")
@patient_only
def bookings():
    return _bookings_page()


@bp.post("/bookings/<dbid:booking_id>/cancel")
@patient_only
def cancel(booking_id):
    queries.booking_for_patient(booking_id, g.user.id) or abort(404)
    try:
        services.cancel(booking_id, g.user.id)
    except services.NotFound:
        abort(404)
    except services.InvalidState as error:
        return _bookings_page(error=str(error), status=409)
    flash("Your appointment is cancelled.", "success")
    return redirect(url_for("patient.bookings"))


def _reschedule_page(booking, day=None, error=None, status=200):
    doctor = queries.get_doctor(booking.doctor_id)
    return _slot_page("patient/reschedule.html", doctor, day or booking.slot_date, error=error,
                      status=status, booking=booking)


@bp.get("/bookings/<dbid:booking_id>/reschedule")
@patient_only
def reschedule_form(booking_id):
    booking = queries.booking_for_patient(booking_id, g.user.id) or abort(404)
    return _reschedule_page(booking, forms.parse_date(request.args.get("date")))


@bp.post("/bookings/<dbid:booking_id>/reschedule")
@patient_only
def reschedule(booking_id):
    booking = queries.booking_for_patient(booking_id, g.user.id) or abort(404)
    new_slot_id = forms.parse_int(request.form.get("new_slot_id"), low=1)
    if new_slot_id is None:
        abort(400)
    clash = queries.patient_clash(g.user.id, new_slot_id, ignoring_booking_id=booking_id)
    if clash is not None:
        return _reschedule_page(booking, status=409, error=_CLASH.format(doctor=clash.doctor_name))
    try:
        moved = services.reschedule(booking_id, new_slot_id, g.user.id)
    except services.SlotTaken:
        return _reschedule_page(booking, status=409,
                                error="That slot was just taken. The list below is up to date; please pick another.")
    except services.SlotInPast:
        return _reschedule_page(booking, status=409, error="That time has already passed.")
    except services.NotFound:
        abort(404)
    except services.InvalidState as error:
        return _reschedule_page(booking, status=409, error=str(error))
    risk.score_after_commit(moved)  # the new booking, scored after its transaction committed
    flash("Your appointment has been moved.", "success")
    return redirect(url_for("patient.bookings"))
