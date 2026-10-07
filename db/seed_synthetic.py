"""Fill the database with SYNTHETIC demo data, and print demo logins.

    python -m db.seed_synthetic             # seed an empty database
    python -m db.seed_synthetic --reset     # wipe every table first, then seed

Everything here is invented from the distributions stated below. This script never reads data/raw/
or any public dataset (CLAUDE.md hard rule 4), and a test checks its source for that. A fixed random
seed makes the data repeatable.

What it creates (PLAN section 3):
  * 1 admin, 5 doctors with different specializations, 40 patients with obviously fake names
  * all accounts share one dev password: $SEED_PASSWORD from .env, or a random one that is printed
  * slots 09:00-13:00 (each doctor's own slot length) for the next 14 weekdays, and the previous 14
  * about a third of the future slots booked at random through the real booking service, some cancelled
  * about 40% of the past slots booked, with outcomes (completed / no_show / cancelled) drawn from a
    per-patient no-show propensity, so there is history for the risk flag later
  * age (normal, mean 38, sd 18, clipped to 1-90) and sex (55% F) for each patient
  * future bookings are back-dated: a third were made just now, the rest up to a month ago, so lead times vary
  * every non-cancelled booking is scored by the SAVED risk model (ml/artifacts), so the demo shows Low, Medium and
    High badges to staff; without a model file the seed still works and simply has no flags
"""
from __future__ import annotations

import argparse
import os
import random
import secrets
import sys
from datetime import date, datetime, time, timedelta

import sqlalchemy as sa

from app import accounts, clock, create_app, risk, scheduling, services
from app.extensions import db
from app.models import Booking, Doctor, PatientProfile, RiskScore, Slot, User

SEED = 20261006
DOMAIN = "cams-demo.test"  # .test is a reserved TLD: these addresses can never be real

DOCTORS = [  # (first name, specialization, appointment minutes)
    ("Meera", "General Medicine", 15),
    ("Arjun", "Pediatrics", 20),
    ("Kavya", "Dermatology", 15),
    ("Rohan", "Orthopedics", 20),
    ("Sana", "Cardiology", 30),
]
FIRST_NAMES = [
    "Asha", "Bharat", "Chitra", "Deepak", "Esha", "Farhan", "Gita", "Harish", "Indu", "Jayant",
    "Kiran", "Lata", "Manoj", "Nisha", "Omkar", "Pooja", "Qadir", "Rekha", "Suresh", "Tara",
    "Uday", "Vani", "Wasim", "Yamini", "Zoya", "Anil", "Bina", "Chetan", "Divya", "Girish",
]
REASONS = ["Check-up", "Follow-up", "Fever and cold", "Skin rash", "Back pain", "Chest discomfort",
           "Vaccination advice", "Routine review", "Headache", "Joint pain"]
PATIENTS = 40
WEEKDAYS = 14
SESSION = (time(9, 0), time(13, 0))
FUTURE_BOOKED_SHARE = 1 / 3
FUTURE_CANCELLED_SHARE = 0.12
PAST_BOOKED_SHARE = 0.40
PAST_CANCELLED_SHARE = 0.10


class AlreadySeeded(RuntimeError):
    pass


def _weekdays(start: date, count: int, step: int) -> list[date]:
    """`count` weekdays beginning at `start` (inclusive), walking forward (step=1) or backward (-1)."""
    days, day = [], start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=step)
    return sorted(days)


def _wipe() -> None:
    for model in (RiskScore, Booking, Slot, PatientProfile, Doctor, User):  # children before parents
        db.session.execute(sa.delete(model))
    db.session.commit()


def _create_people(rng: random.Random, password_hash: str, today: date):
    admin = User(name="Admin Demo", email=f"admin@{DOMAIN}", password_hash=password_hash, role="admin")
    db.session.add(admin)

    doctors = []
    for first, specialization, minutes in DOCTORS:
        user = User(name=f"Dr. {first} Demo", email=f"dr.{first.lower()}@{DOMAIN}",
                    password_hash=password_hash, role="doctor")
        db.session.add(user)
        db.session.flush()
        doctor = Doctor(user_id=user.id, specialization=specialization, slot_minutes=minutes)
        db.session.add(doctor)
        doctors.append(doctor)

    patients = []  # (user, no-show propensity)
    for n in range(1, PATIENTS + 1):
        first = FIRST_NAMES[(n - 1) % len(FIRST_NAMES)]
        user = User(name=f"{first} Demo-{n:02d}", email=f"{first.lower()}.demo{n:02d}@{DOMAIN}",
                    password_hash=password_hash, role="patient")
        db.session.add(user)
        db.session.flush()
        age = min(90, max(1, round(rng.gauss(38, 18))))
        born = today - timedelta(days=int(age * 365.25) + rng.randint(0, 364))
        db.session.add(PatientProfile(user_id=user.id, date_of_birth=born, sex="F" if rng.random() < 0.55 else "M"))
        patients.append((user, min(0.6, rng.betavariate(1.2, 5.0))))
    db.session.commit()
    return admin, doctors, patients


def _generate_slots(doctors, days: list[date]) -> int:
    created = 0
    for doctor in doctors:
        created += scheduling.generate_slots(doctor.id, days[0], days[-1], *SESSION, skip_weekends=True).created
    return created


def _booked_at(rng: random.Random, now: datetime) -> datetime:
    """When a future booking was made: a third just now (so some are same-day), the rest up to a month ago.
    Never after `now`, so never after the appointment."""
    if rng.random() < 0.33:
        return now - timedelta(minutes=rng.randint(1, 120))
    return now - timedelta(days=min(30.0, rng.expovariate(1 / 7)), hours=rng.uniform(0, 8))


def _book_future(rng: random.Random, patients, now: datetime) -> tuple[int, int]:
    """Book about a third of the future slots through the real service; cancel some. Returns (booked, cancelled)."""
    slots = db.session.execute(
        sa.select(Slot.id, Slot.slot_date, Slot.slot_time).order_by(Slot.slot_date, Slot.slot_time, Slot.id)).all()
    booked = cancelled = 0
    for slot in slots:
        if datetime.combine(slot.slot_date, slot.slot_time) <= now or rng.random() >= FUTURE_BOOKED_SHARE:
            continue
        patient = rng.choice(patients)[0]
        booking = services.book(slot.id, patient.id, reason=rng.choice(REASONS))
        db.session.execute(sa.update(Booking).where(Booking.id == booking.id).values(created_at=_booked_at(rng, now)))
        db.session.commit()
        booked += 1
        if rng.random() < FUTURE_CANCELLED_SHARE:
            services.cancel(booking.id, actor_id=patient.id)
            cancelled += 1
    return booked, cancelled


def _book_past(rng: random.Random, patients, now: datetime) -> int:
    """Past appointments with outcomes, inserted directly: the service refuses to book the past."""
    slots = db.session.execute(sa.select(Slot.id, Slot.slot_date, Slot.slot_time)).all()
    rows = []
    for slot in slots:
        start = datetime.combine(slot.slot_date, slot.slot_time)
        if start > now or rng.random() >= PAST_BOOKED_SHARE:
            continue
        patient, propensity = rng.choice(patients)
        lead = timedelta(days=min(30, int(rng.expovariate(1 / 7))), hours=rng.randint(0, 20))
        created = min(start - lead, start - timedelta(hours=1))
        if rng.random() < PAST_CANCELLED_SHARE:
            status, cancelled_at, cancelled_by = "cancelled", start - timedelta(hours=rng.randint(2, 48)), patient.id
            cancelled_at = max(cancelled_at, created)
        else:
            status = "no_show" if rng.random() < propensity else "completed"
            cancelled_at = cancelled_by = None
        rows.append(Booking(slot_id=slot.id, patient_id=patient.id, status=status, reason=rng.choice(REASONS),
                            created_at=created, cancelled_at=cancelled_at, cancelled_by=cancelled_by))
    db.session.add_all(rows)
    db.session.commit()
    return len(rows)


def seed(password: str, rng_seed: int = SEED, reset: bool = False) -> dict:
    """Seed the database in the current app context. Returns a summary including the demo logins."""
    if reset:
        _wipe()
    elif db.session.execute(sa.select(sa.func.count()).select_from(User)).scalar_one():
        raise AlreadySeeded("The database already has users. Use --reset to wipe it and seed again.")

    rng = random.Random(rng_seed)
    now = clock.now()
    today = now.date()
    password_hash = accounts.hash_password(password)  # one hash shared by all demo accounts

    admin, doctors, patients = _create_people(rng, password_hash, today)
    future_days = _weekdays(today, WEEKDAYS, 1)
    past_days = _weekdays(today - timedelta(days=1), WEEKDAYS, -1)
    future_slots = _generate_slots(doctors, future_days)
    past_slots = _generate_slots(doctors, past_days)
    future_booked, future_cancelled = _book_future(rng, patients, now)
    past_booked = _book_past(rng, patients, now)
    flags = risk.score_many(db.session.execute(sa.select(Booking.id).where(Booking.status != "cancelled")).scalars().all())

    doctor_user = db.session.get(User, doctors[0].user_id)
    return {
        "admin": admin.email, "doctor": doctor_user.email, "patient": patients[0][0].email,
        "doctors": len(doctors), "patients": len(patients),
        "slots": future_slots + past_slots, "future_slots": future_slots, "past_slots": past_slots,
        "future_bookings": future_booked, "future_cancelled": future_cancelled, "past_bookings": past_booked,
        "flags": flags,  # {"scored", "low", "medium", "high"} or None when there is no risk model
        "future_days": (future_days[0], future_days[-1]), "past_days": (past_days[0], past_days[-1]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Seed the CAMS database with synthetic demo data.")
    parser.add_argument("--reset", action="store_true", help="delete every row first (synthetic data only!)")
    args = parser.parse_args(argv)
    try:
        app = create_app()
    except RuntimeError as exc:  # DATABASE_URL or SECRET_KEY missing
        print(exc)
        return 1
    password = os.environ.get("SEED_PASSWORD") or secrets.token_urlsafe(9)
    with app.app_context():
        try:
            summary = seed(password, reset=args.reset)
        except AlreadySeeded as exc:
            print(exc)
            return 1
    print("Seeded synthetic data (nothing is read from any dataset).")
    print(f"  {summary['doctors']} doctors, {summary['patients']} patients, {summary['slots']} slots "
          f"({summary['future_days'][0]} to {summary['future_days'][1]} ahead, "
          f"{summary['past_days'][0]} to {summary['past_days'][1]} behind)")
    print(f"  {summary['future_bookings']} future bookings ({summary['future_cancelled']} cancelled), "
          f"{summary['past_bookings']} past appointments with outcomes")
    flags = summary["flags"]
    if flags:
        print(f"  risk flags from the saved model (advisory, staff pages only): {flags['low']} Low, "
              f"{flags['medium']} Medium, {flags['high']} High")
    else:
        print("  no risk model found in ml/artifacts, so no risk flags (booking is unaffected)")
    print("\nDemo logins (one shared password for every demo account):")
    print(f"  admin    {summary['admin']}")
    print(f"  doctor   {summary['doctor']}")
    print(f"  patient  {summary['patient']}")
    print(f"  password {password}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
