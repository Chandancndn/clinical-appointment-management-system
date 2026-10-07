"""db/seed_synthetic.py: synthetic data only, repeatable, and never reads the datasets (hard rule 4)."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select
from werkzeug.security import check_password_hash

from app import clock
from app.extensions import db
from app.models import Booking, Doctor, PatientProfile, Slot, Standby, User
from db.seed_synthetic import DOCTORS, PATIENTS, WEEKDAYS, AlreadySeeded, seed

SEED_SOURCE = Path(__file__).resolve().parents[1] / "db" / "seed_synthetic.py"
MONDAY_8AM = datetime(2030, 1, 7, 8, 0)


@pytest.fixture
def seeded(app, monkeypatch):
    monkeypatch.setattr(clock, "now", lambda: MONDAY_8AM)
    return seed("a-demo-password")


def count(model):
    return db.session.execute(select(func.count()).select_from(model)).scalar_one()


def test_the_seed_never_touches_the_datasets():
    source = SEED_SOURCE.read_text()
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    imports = re.findall(r"^\s*(?:import|from)\s+(\S+)", code, re.M)
    assert not {"pandas", "numpy", "csv", "sklearn", "openml"} & set(imports)
    for forbidden in ("read_csv", "KaggleV2", "Hangu", "noshowappointments", "open(", "Path("):
        assert forbidden not in code, forbidden
    # the only mentions of data/raw are the promises in the docstring, not code
    assert "data/raw" not in code.split('"""', 2)[2]


def test_the_seed_creates_the_documented_people(seeded):
    assert seeded["doctors"] == len(DOCTORS) and seeded["patients"] == PATIENTS
    roles = dict(db.session.execute(select(User.role, func.count()).group_by(User.role)).all())
    assert roles == {"admin": 1, "doctor": len(DOCTORS), "patient": PATIENTS}
    assert count(PatientProfile) == PATIENTS and count(Doctor) == len(DOCTORS)
    specializations = {d.specialization for d in db.session.query(Doctor)}
    assert len(specializations) == len(DOCTORS)  # all different
    names = [u.name for u in db.session.query(User).filter_by(role="patient")]
    assert all("Demo-" in n for n in names)  # obviously fake
    assert all(u.email.endswith("@cams-demo.test") for u in db.session.query(User))


def test_the_demo_logins_work_and_passwords_are_hashed(seeded):
    for role in ("admin", "doctor", "patient"):
        user = db.session.query(User).filter_by(email=seeded[role]).one()
        assert user.role == role
        assert user.password_hash != "a-demo-password"
        assert check_password_hash(user.password_hash, "a-demo-password")


def test_ages_and_sex_come_from_the_stated_distributions(seeded):
    profiles = db.session.query(PatientProfile).all()
    ages = [(MONDAY_8AM.date() - p.date_of_birth).days / 365.25 for p in profiles]
    assert all(0.9 <= a <= 92 for a in ages)
    assert {p.sex for p in profiles} == {"F", "M"}


def test_slots_cover_14_weekdays_ahead_and_behind_in_each_doctors_slot_length(seeded):
    per_day = sum(240 // minutes for _, _, minutes in DOCTORS)
    assert seeded["slots"] == per_day * WEEKDAYS * 2
    assert count(Slot) == seeded["slots"]
    days = {d for (d,) in db.session.execute(select(Slot.slot_date).distinct())}
    assert len(days) == WEEKDAYS * 2 and all(d.weekday() < 5 for d in days)


def test_future_bookings_are_confirmed_or_cancelled_and_past_ones_have_outcomes(seeded):
    now = MONDAY_8AM
    rows = db.session.execute(
        select(Booking.status, Slot.slot_date, Slot.slot_time, Booking.created_at)
        .join(Slot, Slot.id == Booking.slot_id)).all()
    ahead = [r for r in rows if datetime.combine(r.slot_date, r.slot_time) > now]
    future = [r for r in ahead if r.status != "closed"]  # a slot closed for leave is not an appointment
    past = [r for r in rows if datetime.combine(r.slot_date, r.slot_time) <= now]
    assert {r.status for r in future} == {"confirmed", "cancelled"}
    assert {r.status for r in past} == {"completed", "no_show", "cancelled"}
    assert len(future) == seeded["future_bookings"] + seeded["full_day_bookings"] and len(past) == seeded["past_bookings"]
    assert len(ahead) - len(future) == seeded["closed_slots"]
    future_slots = sum(1 for r in db.session.execute(select(Slot.slot_date, Slot.slot_time))
                       if datetime.combine(*r) > now)
    assert 0.25 < len(future) / future_slots < 0.42  # about a third
    outcome_rate = sum(r.status == "no_show" for r in past) / sum(r.status != "cancelled" for r in past)
    assert 0.05 < outcome_rate < 0.40  # a plausible no-show share, drawn from propensities
    assert all(r.created_at < datetime.combine(r.slot_date, r.slot_time) for r in past)  # booked before the visit


def test_the_one_active_booking_per_slot_rule_holds_for_all_seeded_data(seeded):
    most = db.session.execute(
        select(func.count()).select_from(Booking).where(Booking.status != "cancelled").group_by(Booking.slot_id)
        .order_by(func.count().desc()).limit(1)).scalar_one()
    assert most == 1


def booking_snapshot():
    """Bookings by natural keys (who, when, outcome), so row ids, which MySQL never reuses, don't matter."""
    from sqlalchemy.orm import aliased
    doctor_user, patient_user = aliased(User), aliased(User)
    return sorted(db.session.execute(
        select(doctor_user.email, Slot.slot_date, Slot.slot_time, patient_user.email, Booking.status, Booking.reason)
        .join(Slot, Slot.id == Booking.slot_id)
        .join(Doctor, Doctor.id == Slot.doctor_id)
        .join(doctor_user, doctor_user.id == Doctor.user_id)
        .join(patient_user, patient_user.id == Booking.patient_id)).all())


def test_the_same_seed_gives_the_same_data(app, monkeypatch):
    monkeypatch.setattr(clock, "now", lambda: MONDAY_8AM)
    first = seed("pw-one-pw", rng_seed=7)
    snapshot = booking_snapshot()
    again = seed("pw-two-pw", rng_seed=7, reset=True)
    assert len(snapshot) > 100
    assert booking_snapshot() == snapshot
    assert {k: first[k] for k in ("slots", "future_bookings", "past_bookings")} == \
           {k: again[k] for k in ("slots", "future_bookings", "past_bookings")}
    seed("pw-three-pw", rng_seed=8, reset=True)
    assert booking_snapshot() != snapshot  # a different seed really does give different data


def test_seeding_twice_needs_an_explicit_reset(seeded):
    with pytest.raises(AlreadySeeded):
        seed("another-password")
    assert seed("another-password", reset=True)["patients"] == PATIENTS
    assert count(User) == 1 + len(DOCTORS) + PATIENTS


# ---- leave and standby, so the demo shows both ---------------------------------------------------------------------
def day_states(doctor_email, day):
    """(slot time, status of the row holding it or None) for a doctor's day."""
    return db.session.execute(
        select(Slot.slot_time, Booking.status).select_from(Slot)
        .join(Doctor, Doctor.id == Slot.doctor_id).join(User, User.id == Doctor.user_id)
        .outerjoin(Booking, Booking.confirmed_slot_id == Slot.id)
        .where(User.email == doctor_email, Slot.slot_date == day).order_by(Slot.slot_time)).all()


def test_one_doctor_has_a_day_closed_for_leave_and_nobody_was_cancelled_for_it(seeded):
    doctor_email, day = seeded["leave"]
    states = day_states(doctor_email, day)
    assert states and all(status in ("closed", "confirmed") for _, status in states)  # nothing left free
    assert sum(status == "closed" for _, status in states) == seeded["closed_slots"] > 0
    assert count(Booking) - db.session.execute(select(func.count()).select_from(Booking).where(Booking.status != "closed")).scalar_one() \
        == seeded["closed_slots"]  # closed rows exist only on that day
    closed_by_doctor = db.session.execute(
        select(func.count()).select_from(Booking).join(User, User.id == Booking.patient_id)
        .where(Booking.status == "closed", User.email == doctor_email)).scalar_one()
    assert closed_by_doctor == seeded["closed_slots"]  # held in the doctor's own name


def test_one_day_is_fully_booked_and_has_a_standby_list_with_the_demo_patient(seeded):
    doctor_email, day = seeded["full_day"]
    states = day_states(doctor_email, day)
    assert states and all(status == "confirmed" for _, status in states)  # every slot is an appointment
    waiting = db.session.execute(
        select(User.email).select_from(Standby).join(User, User.id == Standby.patient_id)
        .where(Standby.slot_date == day).order_by(Standby.id)).scalars().all()
    assert len(waiting) == seeded["standby"] == 3 and seeded["patient"] in waiting
    assert (doctor_email, day) != seeded["leave"]


def test_nobody_on_standby_already_has_an_appointment_with_that_doctor_that_day(seeded):
    doctor_email, day = seeded["full_day"]
    clash = db.session.execute(
        select(func.count()).select_from(Standby)
        .join(Slot, (Slot.doctor_id == Standby.doctor_id) & (Slot.slot_date == Standby.slot_date))
        .join(Booking, (Booking.confirmed_slot_id == Slot.id) & (Booking.patient_id == Standby.patient_id))).scalar_one()
    assert clash == 0


def test_resetting_also_clears_the_standby_list(seeded):
    assert count(Standby) == 3
    again = seed("another-password", reset=True)  # would fail on a foreign key if standby rows were left behind
    assert count(Standby) == again["standby"] == 3
