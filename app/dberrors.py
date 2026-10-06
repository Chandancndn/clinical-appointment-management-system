"""Recognise database constraint errors on both MySQL (PyMySQL) and SQLite."""
from __future__ import annotations

from sqlalchemy.exc import IntegrityError

MYSQL_DUPLICATE_ENTRY = 1062
MYSQL_NO_REFERENCED_ROW = 1452


def _code(exc: IntegrityError):
    args = getattr(exc.orig, "args", None)
    return args[0] if args else None


def is_duplicate_key(exc: IntegrityError) -> bool:
    return _code(exc) == MYSQL_DUPLICATE_ENTRY or "UNIQUE constraint failed" in str(exc.orig)


def is_missing_reference(exc: IntegrityError) -> bool:
    return _code(exc) == MYSQL_NO_REFERENCED_ROW or "FOREIGN KEY constraint failed" in str(exc.orig)
