"""Bring a MySQL database made by an earlier db/schema.sql up to date. Run by python -m db.init_db.

Each step says whether it is needed by looking at the database, so applying them twice changes nothing, and none of them
drops or rewrites data. A brand-new database (created from the current schema.sql) needs none of them.

  1. bookings.status gains 'closed' (a slot the doctor closed for leave). The stored generated column and its unique
     key, which are what forbid double-booking, are not touched.
  2. the standby table (patients waiting for a slot on a full day).
"""
from __future__ import annotations

import re

from app.models import BOOKING_STATUSES

from . import schema_statements


def _status_type(conn) -> str:
    return conn.exec_driver_sql(
        "SELECT COLUMN_TYPE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE() "
        "AND TABLE_NAME = 'bookings' AND COLUMN_NAME = 'status'").scalar_one()


def _has_table(conn, name: str) -> bool:
    return conn.exec_driver_sql(
        "SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s", (name,)
    ).scalar_one() > 0


def _create_statement(table: str) -> str:
    return next(s for s in schema_statements() if re.match(rf"CREATE TABLE\s+{table}\b", s, re.I))


def apply(conn) -> list:
    """Apply every step the database still needs; returns their descriptions (empty when it is up to date)."""
    applied = []
    if _has_table(conn, "bookings") and any(f"'{status}'" not in _status_type(conn) for status in BOOKING_STATUSES):
        values = ",".join(f"'{status}'" for status in BOOKING_STATUSES)
        conn.exec_driver_sql(f"ALTER TABLE bookings MODIFY COLUMN status ENUM({values}) NOT NULL DEFAULT 'confirmed'")
        applied.append("bookings.status: added 'closed' (slots closed for leave)")
    if _has_table(conn, "bookings") and not _has_table(conn, "standby"):
        conn.exec_driver_sql(_create_statement("standby"))
        applied.append("standby: table created")
    return applied
