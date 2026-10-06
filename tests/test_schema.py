"""db/schema.sql (what MySQL runs) and the SQLAlchemy models (what SQLite tests run) must agree."""
from __future__ import annotations

import re

from app.extensions import db
from app import models  # noqa: F401  (registers the tables on db.metadata)
from db import schema_statements

NOT_COLUMNS = {"PRIMARY", "UNIQUE", "KEY", "CONSTRAINT", "FOREIGN", "INDEX", "CHECK"}


def tables_in_schema_sql() -> dict[str, set[str]]:
    """Table name -> column names, read from db/schema.sql (one column or constraint per line)."""
    tables = {}
    for statement in schema_statements():
        match = re.match(r"CREATE TABLE\s+(\w+)\s*\((.*)\)\s*ENGINE", statement, re.S | re.I)
        assert match, f"cannot parse statement: {statement[:60]!r}"
        columns = set()
        for line in match.group(2).splitlines():
            line = line.strip().rstrip(",")
            if line and line.split()[0].upper() not in NOT_COLUMNS:
                columns.add(line.split()[0].strip("`"))
        tables[match.group(1)] = columns
    return tables


def test_schema_sql_has_the_same_tables_and_columns_as_the_models():
    from_models = {name: {c.name for c in table.columns} for name, table in db.metadata.tables.items()}
    assert tables_in_schema_sql() == from_models


def test_the_five_m1_tables_exist():
    assert set(db.metadata.tables) == {"users", "patient_profiles", "doctors", "slots", "bookings"}
