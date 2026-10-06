"""Check that DATABASE_URL works and the server can enforce the no-double-booking rule.

    python -m db.check_connection

Connects, prints the server version, then builds a throwaway temporary table with the same
generated column + UNIQUE key as bookings.confirmed_slot_id and checks it behaves. If this fails
on your MySQL, tell the team: CLAUDE.md describes a slot_holds fallback table.
"""
from __future__ import annotations

import sys

import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError

from app import create_app
from app.extensions import db

PROBE_TABLE = """
CREATE TEMPORARY TABLE cams_probe (
  id      INT PRIMARY KEY,
  slot_id INT NOT NULL,
  status  VARCHAR(9) NOT NULL,
  held    INT GENERATED ALWAYS AS (CASE WHEN status = 'cancelled' THEN NULL ELSE slot_id END) STORED,
  UNIQUE (held)
)
"""


def probe_generated_unique(conn) -> str:
    """Return '' if one active row per slot is enforced and cancelled rows are free, else the problem."""
    try:
        conn.exec_driver_sql(PROBE_TABLE)
    except DBAPIError as exc:
        return f"cannot create a stored generated column: {exc.orig}"
    insert = sa.text("INSERT INTO cams_probe (id, slot_id, status) VALUES (:id, 7, :status)")
    conn.execute(insert, {"id": 1, "status": "confirmed"})
    try:
        conn.execute(insert, {"id": 2, "status": "confirmed"})
        return "a second active row for the same slot was accepted"
    except IntegrityError:
        pass
    conn.execute(insert, {"id": 3, "status": "cancelled"})
    conn.execute(insert, {"id": 4, "status": "cancelled"})
    return ""


def main() -> int:
    try:
        app = create_app()
    except RuntimeError as exc:  # DATABASE_URL missing
        print(exc)
        return 1
    with app.app_context():
        url = db.engine.url.render_as_string(hide_password=True)
        try:
            with db.engine.connect() as conn:
                version_sql = "SELECT sqlite_version()" if db.engine.dialect.name == "sqlite" else "SELECT VERSION()"
                version = conn.execute(sa.text(version_sql)).scalar_one()
                print(f"Connected to {url}")
                print(f"Server: {db.engine.dialect.name} {version}")
                problem = probe_generated_unique(conn)
        except DBAPIError as exc:
            print(f"Could not connect to {url}\n{exc.orig}")
            return 1
    if problem:
        print(f"Generated column + UNIQUE: FAILED ({problem})")
        return 1
    print("Generated column + UNIQUE: OK (one active booking per slot, cancelled rows free the slot)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
