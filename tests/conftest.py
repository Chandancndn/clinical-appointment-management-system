"""Test fixtures: a fresh, empty database for every test, on SQLite and/or real MySQL.

    pytest                  SQLite only (default)
    pytest --engine mysql   MySQL, using TEST_DATABASE_URL (or DATABASE_URL) from .env
    pytest --engine both    both

The MySQL run builds its tables from db/schema.sql, the file that ships, so the generated
column and its UNIQUE key are proven on the real engine. The SQLite run builds them from the
SQLAlchemy models.
"""
from __future__ import annotations

import itertools
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv
from sqlalchemy import inspect
from sqlalchemy.engine import make_url

from helpers import FAST_HASH, PASSWORD, csrf_token

ROOT = Path(__file__).resolve().parents[1]
ENGINES = ("sqlite", "mysql")


def pytest_addoption(parser):
    parser.addoption(
        "--engine",
        choices=("sqlite", "mysql", "both"),
        default="sqlite",
        help="database engine(s) the tests run on (default: sqlite)",
    )


def pytest_generate_tests(metafunc):
    if "engine_name" in metafunc.fixturenames:
        choice = metafunc.config.getoption("--engine")
        names = ENGINES if choice == "both" else (choice,)
        metafunc.parametrize("engine_name", names, indirect=True)


@pytest.fixture
def engine_name(request):
    return request.param


def _mysql_test_url() -> str:
    """The MySQL URL for tests. Fails (never skips) if it is missing or could hit real data."""
    load_dotenv(ROOT / ".env")
    url = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.fail("--engine mysql needs TEST_DATABASE_URL (or DATABASE_URL) in .env; see README.md")
    parsed = make_url(url)
    if parsed.get_backend_name() != "mysql":
        pytest.fail(f"--engine mysql needs a MySQL URL, got backend '{parsed.get_backend_name()}'")
    if not (parsed.database or "").endswith("_test"):
        pytest.fail(
            f"refusing to run on database '{parsed.database}': the tests drop every table in it. "
            "Create a database whose name ends in _test (README.md, MySQL setup) and set TEST_DATABASE_URL."
        )
    return url


def _create_schema(engine_name: str) -> None:
    from app.extensions import db
    from db import schema_statements

    if engine_name == "sqlite":
        db.create_all()
        return
    with db.engine.begin() as conn:
        conn.exec_driver_sql("SET FOREIGN_KEY_CHECKS=0")
        for table in inspect(conn).get_table_names():
            conn.exec_driver_sql(f"DROP TABLE IF EXISTS `{table}`")
        conn.exec_driver_sql("SET FOREIGN_KEY_CHECKS=1")
        for statement in schema_statements():
            conn.exec_driver_sql(statement)


@pytest.fixture
def app(engine_name, tmp_path, request):
    """A fresh app and database. By default there is NO risk model (RISK_ARTIFACTS_DIR points at nothing), so the
    booking and role tests run exactly the no-model path; a test module opts in with APP_CONFIG = {...}."""
    from app import create_app, risk
    from app.extensions import db

    if engine_name == "sqlite":
        url = f"sqlite:///{tmp_path / 'cams_test.sqlite3'}"
    else:
        url = _mysql_test_url()
    risk.reset_cache()
    config = {
        "SQLALCHEMY_DATABASE_URI": url,
        "TESTING": True,
        "SECRET_KEY": "test-secret",
        "PASSWORD_HASH_METHOD": "pbkdf2:sha256:1000",  # fast hashing for tests
        "RISK_ARTIFACTS_DIR": tmp_path / "no_model_here",
    }
    config.update(getattr(request.module, "APP_CONFIG", {}))
    flask_app = create_app(config)
    with flask_app.app_context():
        _create_schema(engine_name)
        yield flask_app
        db.session.remove()
        db.engine.dispose()


@pytest.fixture
def client(app):
    return app.test_client()


class World:
    """Builds the rows a test needs. Every method commits and returns an id."""

    def __init__(self):
        self._n = itertools.count(1)
        self.emails: dict[int, str] = {}
        self.doctor_user_ids: dict[int, int] = {}

    def user(self, role: str, name: str | None = None) -> int:
        from app.extensions import db
        from app.models import User

        n = next(self._n)
        email = f"{role}{n}@example.test"
        user = User(name=name or f"Test {role} {n}", email=email, password_hash=FAST_HASH, role=role)
        db.session.add(user)
        db.session.commit()
        self.emails[user.id] = email
        return user.id

    def patient(self, name: str | None = None) -> int:
        return self.user("patient", name)

    def admin(self) -> int:
        return self.user("admin")

    def doctor(self, specialization: str = "General Medicine", slot_minutes: int = 15) -> int:
        from app.extensions import db
        from app.models import Doctor

        user_id = self.user("doctor")
        doctor = Doctor(user_id=user_id, specialization=specialization, slot_minutes=slot_minutes)
        db.session.add(doctor)
        db.session.commit()
        self.doctor_user_ids[doctor.id] = user_id
        return doctor.id

    def slots(self, doctor_id: int, count: int, day: date = date(2030, 1, 7)) -> list[int]:
        """`count` consecutive 15-minute slots starting 09:00 on `day`."""
        from app.extensions import db
        from app.models import Slot

        start = datetime.combine(day, time(9, 0))
        rows = [Slot(doctor_id=doctor_id, slot_date=day, slot_time=(start + timedelta(minutes=15 * i)).time())
                for i in range(count)]
        db.session.add_all(rows)
        db.session.commit()
        return [row.id for row in rows]

    def profile(self, user_id: int, date_of_birth: date = date(1990, 6, 15), sex: str = "F") -> int:
        from app.extensions import db
        from app.models import PatientProfile

        row = PatientProfile(user_id=user_id, date_of_birth=date_of_birth, sex=sex)
        db.session.add(row)
        db.session.commit()
        return row.id

    def booking(self, slot_id: int, patient_id: int, status: str = "confirmed") -> int:
        """Insert a booking directly, bypassing the service rules (needed to build past appointments)."""
        from app.extensions import db
        from app.models import Booking

        booking = Booking(slot_id=slot_id, patient_id=patient_id, status=status)
        db.session.add(booking)
        db.session.commit()
        return booking.id

    def login(self, client, user_id: int) -> None:
        """Log `client` in through the real login form."""
        response = client.post("/login", data={
            "email": self.emails[user_id], "password": PASSWORD, "_csrf": csrf_token(client)})
        assert response.status_code == 302, f"login failed: {response.status_code}"


@pytest.fixture
def world(app):
    return World()


@pytest.fixture
def scene(world):
    """One doctor with four slots on 2030-01-07, a second doctor, two patients, an admin, and
    a confirmed booking by `alice` in the doctor's first slot."""
    doctor = world.doctor()
    slots = world.slots(doctor, 4)
    alice, bob = world.patient("Alice Patient"), world.patient("Bob Patient")
    return SimpleNamespace(
        doctor=doctor, doctor_user=world.doctor_user_ids[doctor], slots=slots,
        other_doctor=world.doctor("Dermatology"),
        alice=alice, bob=bob, admin=world.admin(),
        booking=world.booking(slots[0], alice),
        day="2030-01-07",
    )
