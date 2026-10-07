"""CAMS Flask application factory."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask
from werkzeug.routing import BaseConverter, ValidationError
from sqlalchemy import event
from sqlalchemy.engine import Engine

from .extensions import db

ROOT = Path(__file__).resolve().parents[1]


@event.listens_for(Engine, "connect")
def _configure_sqlite(dbapi_connection, _connection_record):
    """SQLite ignores foreign keys by default and fails fast on lock contention; fix both.

    Only applies to SQLite connections (used by the test suite); MySQL is untouched.
    """
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()


class DbIdConverter(BaseConverter):
    """A URL id that can exist in the database (1 to the largest INT). Anything else is a 404, never a server error."""

    regex = r"\d{1,10}"

    def to_python(self, value):
        number = int(value)
        if not 1 <= number <= 2**31 - 1:
            raise ValidationError()
        return number

    def to_url(self, value):
        return str(int(value))


def create_app(config: dict | None = None) -> Flask:
    """Build the app. Settings come from .env / environment, then `config` overrides them."""
    load_dotenv(ROOT / ".env")
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", ""),
        SQLALCHEMY_DATABASE_URI=os.environ.get("DATABASE_URL"),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        PASSWORD_HASH_METHOD="pbkdf2:sha256:600000",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
        MAX_CONTENT_LENGTH=1024 * 1024,
        SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "").lower() in ("1", "true", "yes"),  # set behind HTTPS
        LOGIN_MAX_FAILURES=5,              # wrong passwords for one email from one address before it is blocked
        LOGIN_MAX_FAILURES_PER_ADDRESS=20,  # wrong passwords from one address, whatever the email
        LOGIN_WINDOW_MINUTES=15,           # how long failures count, and how long a block lasts at most
        RISK_ARTIFACTS_DIR=ROOT / "ml" / "artifacts",  # the saved risk model and its card (advisory flag, M7)
        RESULTS_DIR=ROOT / "results",  # read by the admin "Models and simulation" page
        FIGURES_DIR=ROOT / "docs" / "report_assets" / "figures",
    )
    if config:
        app.config.update(config)
    if not app.config.get("SQLALCHEMY_DATABASE_URI"):
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
    if not app.config.get("SECRET_KEY"):
        raise RuntimeError("SECRET_KEY is not set. Copy .env.example to .env and fill it in.")

    app.url_map.converters["dbid"] = DbIdConverter
    db.init_app(app)
    from . import models  # noqa: F401  (registers the tables on db.metadata)
    from .blueprints import account, admin, auth, doctor, main, patient
    from .throttle import Throttle
    from .errors import init_errors
    from .security import init_security

    app.extensions["throttle"] = Throttle()
    init_security(app)
    init_errors(app)
    for module in (main, auth, account, patient, doctor, admin):
        app.register_blueprint(module.bp)
    _register_template_helpers(app)
    return app


def _register_template_helpers(app: Flask) -> None:
    from . import clock

    @app.template_filter("date_long")
    def date_long(value):
        return value.strftime("%a %d %b %Y")

    @app.template_filter("time_short")
    def time_short(value):
        return value.strftime("%H:%M")

    @app.template_filter("datetime_short")
    def datetime_short(value):
        return value.strftime("%d %b %Y, %H:%M")

    @app.template_filter("status_label")
    def status_label(value):
        return {"no_show": "No-show"}.get(value, value.capitalize())

    @app.template_global("has_started")
    def has_started(slot_date, slot_time):
        return datetime.combine(slot_date, slot_time) <= clock.now()
