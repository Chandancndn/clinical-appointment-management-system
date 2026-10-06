"""SQLAlchemy models. They mirror db/schema.sql; tests/test_schema.py keeps the two in step.

The double-booking guarantee lives in two constraints (CLAUDE.md hard rule 1):
  * slots: UNIQUE (doctor_id, slot_date, slot_time)
  * bookings.confirmed_slot_id: a generated column equal to slot_id unless the booking is
    cancelled (then NULL), with its own UNIQUE constraint.
"""
from __future__ import annotations

import sqlalchemy as sa

from .extensions import db

ROLES = ("patient", "doctor", "admin")
SEXES = ("F", "M")
BOOKING_STATUSES = ("confirmed", "completed", "no_show", "cancelled")


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.Enum(*ROLES, name="user_role", create_constraint=True), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, server_default=sa.func.now())

    __table_args__ = (sa.UniqueConstraint("email", name="uq_users_email"),)


class PatientProfile(db.Model):
    __tablename__ = "patient_profiles"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    sex = db.Column(db.Enum(*SEXES, name="patient_sex", create_constraint=True), nullable=False)

    __table_args__ = (sa.UniqueConstraint("user_id", name="uq_patient_profiles_user"),)


class Doctor(db.Model):
    __tablename__ = "doctors"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    specialization = db.Column(db.String(100), nullable=False)
    slot_minutes = db.Column(db.Integer, nullable=False)

    __table_args__ = (sa.UniqueConstraint("user_id", name="uq_doctors_user"),)


class Slot(db.Model):
    __tablename__ = "slots"

    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("doctors.id"), nullable=False)
    slot_date = db.Column(db.Date, nullable=False)
    slot_time = db.Column(db.Time, nullable=False)

    __table_args__ = (
        sa.UniqueConstraint("doctor_id", "slot_date", "slot_time", name="uq_slots_doctor_date_time"),
    )


class Booking(db.Model):
    __tablename__ = "bookings"

    id = db.Column(db.Integer, primary_key=True)
    slot_id = db.Column(db.Integer, db.ForeignKey("slots.id"), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    status = db.Column(
        db.Enum(*BOOKING_STATUSES, name="booking_status", create_constraint=True),
        nullable=False,
        default="confirmed",
        server_default="confirmed",
    )
    reason = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, nullable=False, server_default=sa.func.now())
    cancelled_at = db.Column(db.DateTime)
    cancelled_by = db.Column(db.Integer, db.ForeignKey("users.id"))
    # slot_id while the booking is active (confirmed, completed, no_show); NULL once cancelled.
    # Computed by the database; the app never writes it.
    confirmed_slot_id = db.Column(
        db.Integer,
        sa.Computed("CASE WHEN status = 'cancelled' THEN NULL ELSE slot_id END", persisted=True),
    )

    __table_args__ = (sa.UniqueConstraint("confirmed_slot_id", name="uq_bookings_confirmed_slot"),)
