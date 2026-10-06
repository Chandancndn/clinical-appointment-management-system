"""Database helpers shared by the db scripts and the test suite."""
from __future__ import annotations

import re
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def schema_statements() -> list[str]:
    """The statements in schema.sql, comments removed, without the trailing semicolons."""
    sql = re.sub(r"--[^\n]*", "", SCHEMA_PATH.read_text())
    return [statement.strip() for statement in sql.split(";") if statement.strip()]
