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

import pytest
from dotenv import load_dotenv
from sqlalchemy import inspect
from sqlalchemy.engine import make_url

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
def app(engine_name, tmp_path):
    from app import create_app
    from app.extensions import db

    if engine_name == "sqlite":
        url = f"sqlite:///{tmp_path / 'cams_test.sqlite3'}"
    else:
        url = _mysql_test_url()
    flask_app = create_app({"SQLALCHEMY_DATABASE_URI": url, "TESTING": True, "SECRET_KEY": "test-secret"})
    with flask_app.app_context():
        _create_schema(engine_name)
        yield flask_app
        db.session.remove()
        db.engine.dispose()


class World:
    """Builds the rows a booking test needs. Every method commits and returns an id."""

    def __init__(self):
        self._n = itertools.count(1)

    def user(self, role: str) -> int:
        from app.extensions import db
        from app.models import User

        n = next(self._n)
        user = User(name=f"Test {role} {n}", email=f"{role}{n}@example.test",
                    password_hash="not-a-real-hash", role=role)
        db.session.add(user)
        db.session.commit()
        return user.id

    def patient(self) -> int:
        return self.user("patient")

    def doctor(self) -> int:
        from app.extensions import db
        from app.models import Doctor

        doctor = Doctor(user_id=self.user("doctor"), specialization="General Medicine", slot_minutes=15)
        db.session.add(doctor)
        db.session.commit()
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


@pytest.fixture
def world(app):
    return World()
