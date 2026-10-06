"""CAMS Flask application factory."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask
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


def create_app(config: dict | None = None) -> Flask:
    """Build the app. Settings come from .env / environment, then `config` overrides them."""
    load_dotenv(ROOT / ".env")
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", ""),
        SQLALCHEMY_DATABASE_URI=os.environ.get("DATABASE_URL"),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    if config:
        app.config.update(config)
    if not app.config.get("SQLALCHEMY_DATABASE_URI"):
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
    db.init_app(app)
    from . import models  # noqa: F401  (registers the tables on db.metadata)

    return app
