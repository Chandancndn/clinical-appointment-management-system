"""Create the CAMS tables in the MySQL database named by DATABASE_URL.

    python -m db.init_db

Applies db/schema.sql. Tables that already exist are left alone, so it is safe to re-run; it
never drops anything.
"""
from __future__ import annotations

import re
import sys

import sqlalchemy as sa

from app import create_app
from app.extensions import db
from db import schema_statements


def main() -> int:
    try:
        app = create_app()
    except RuntimeError as exc:  # DATABASE_URL missing
        print(exc)
        return 1
    with app.app_context():
        if db.engine.dialect.name != "mysql":
            print(f"db/schema.sql is MySQL DDL but DATABASE_URL uses '{db.engine.dialect.name}'.")
            return 1
        existing = set(sa.inspect(db.engine).get_table_names())
        with db.engine.begin() as conn:
            for statement in schema_statements():
                table = re.match(r"CREATE TABLE\s+(\w+)", statement, re.I).group(1)
                if table in existing:
                    print(f"  {table}: already exists, left alone")
                    continue
                conn.exec_driver_sql(statement)
                print(f"  {table}: created")
    return 0


if __name__ == "__main__":
    sys.exit(main())
