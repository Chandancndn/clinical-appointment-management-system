"""Accounts: registration, doctor creation and password checks. Passwords are only ever stored hashed."""
from __future__ import annotations

from datetime import date
from typing import Optional

import sqlalchemy as sa
from flask import current_app
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

from .dberrors import is_duplicate_key
from .extensions import db
from .models import Doctor, PatientProfile, User


class AccountError(Exception):
    http_status = 400


class EmailTaken(AccountError):
    """An account with this email already exists."""

    http_status = 409


def hash_password(password: str) -> str:
    return generate_password_hash(password, method=current_app.config["PASSWORD_HASH_METHOD"])


def get_user(user_id: int) -> Optional[User]:
    return db.session.get(User, user_id)


_dummy_hash: Optional[str] = None


def _burn_time(password: str) -> None:
    """Hash-check against a dummy so an unknown email takes as long as a wrong password."""
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password("not-a-real-password")
    check_password_hash(_dummy_hash, password)


def authenticate(email: str, password: str) -> Optional[User]:
    """The user with this email and password, or None. Never says which of the two was wrong."""
    user = db.session.execute(sa.select(User).where(User.email == email.strip().lower())).scalar_one_or_none()
    if user is None:
        _burn_time(password)
        return None
    return user if check_password_hash(user.password_hash, password) else None


def _save_user(user: User, make_profile):
    """Insert the user and its profile row in one transaction; returns the profile row.
    A taken email becomes EmailTaken.

    The database's UNIQUE key on users.email decides, not a lookup first: two people registering the
    same email at once cannot both succeed.
    """
    db.session.add(user)
    try:
        db.session.flush()  # INSERT the user: this is where a duplicate email is detected
        profile = make_profile(user)
        db.session.add(profile)
        db.session.commit()
        return profile
    except IntegrityError as exc:
        db.session.rollback()
        if is_duplicate_key(exc):
            raise EmailTaken(f"{user.email} is already registered") from exc
        raise
    except Exception:
        db.session.rollback()
        raise


def register_patient(name: str, email: str, password: str, date_of_birth: date, sex: str) -> User:
    """Create a patient: the users row and the patient_profiles row in one transaction."""
    user = User(name=name.strip(), email=email.strip().lower(), password_hash=hash_password(password),
                role="patient")
    _save_user(user, lambda u: PatientProfile(user_id=u.id, date_of_birth=date_of_birth, sex=sex))
    return user


def create_doctor(name: str, email: str, password: str, specialization: str, slot_minutes: int) -> Doctor:
    """Create a doctor: the users row (role doctor) and the doctors row in one transaction."""
    user = User(name=name.strip(), email=email.strip().lower(), password_hash=hash_password(password),
                role="doctor")
    return _save_user(user, lambda u: Doctor(user_id=u.id, specialization=specialization.strip(),
                                             slot_minutes=slot_minutes))


def password_is_right(user_id: int, password: str) -> bool:
    user = db.session.get(User, user_id)
    return user is not None and check_password_hash(user.password_hash, password)


def change_password(user_id: int, new_password: str) -> None:
    """Store a new hash for the user. The caller has already checked the current password."""
    user = db.session.get(User, user_id)
    user.password_hash = hash_password(new_password)
    db.session.commit()
