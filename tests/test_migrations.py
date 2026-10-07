"""A database created by an earlier db/schema.sql is brought up to date by python -m db.init_db, without losing a row.

MySQL only: db/schema.sql is MySQL DDL, and the SQLite test databases are always built fresh from the models.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app import services
from app.extensions import db
from app.models import BOOKING_STATUSES

OLD_STATUS = "ENUM('confirmed','completed','no_show','cancelled') NOT NULL DEFAULT 'confirmed'"


def status_type() -> str:
    with db.engine.connect() as conn:
        return conn.execute(text("SELECT COLUMN_TYPE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE() "
                                 "AND TABLE_NAME = 'bookings' AND COLUMN_NAME = 'status'")).scalar_one()


@pytest.fixture
def old_database(app, engine_name, scene):
    """The test database as an older schema.sql made it: no 'closed' status, no standby table, one existing booking."""
    if engine_name != "mysql":
        pytest.skip("migrations apply to MySQL databases")
    db.session.rollback()  # release the session's metadata locks, or the DDL below waits for them
    db.session.close()
    with db.engine.begin() as conn:
        conn.exec_driver_sql("SET SESSION lock_wait_timeout = 20")  # fail fast instead of hanging if something holds a lock
        conn.exec_driver_sql("DROP TABLE standby")
        conn.exec_driver_sql(f"ALTER TABLE bookings MODIFY COLUMN status {OLD_STATUS}")
    assert "closed" not in status_type()
    return scene


def test_an_old_database_cannot_hold_a_closed_slot(old_database):
    with pytest.raises(Exception):
        services.close_slot(old_database.slots[1], old_database.doctor_user)
    db.session.rollback()


def test_migrating_adds_the_status_and_the_table_and_keeps_the_data(old_database):
    from db import migrations

    with db.engine.begin() as conn:
        applied = migrations.apply(conn)
    assert len(applied) == 2 and all(f"'{s}'" in status_type() for s in BOOKING_STATUSES)
    with db.engine.connect() as conn:
        assert conn.execute(text("SELECT status FROM bookings WHERE id = :b"), {"b": old_database.booking}).scalar_one() == "confirmed"
        assert conn.execute(text("SELECT COUNT(*) FROM standby")).scalar_one() == 0
    assert services.close_slot(old_database.slots[1], old_database.doctor_user).status == "closed"


def test_the_unique_key_still_guards_the_slot_after_migrating(old_database):
    from db import migrations

    with db.engine.begin() as conn:
        migrations.apply(conn)
    with pytest.raises(services.SlotTaken):
        services.book(old_database.slots[0], old_database.bob)  # Alice's booking from before the migration still holds it
    with pytest.raises(IntegrityError):
        db.session.execute(text("INSERT INTO bookings (slot_id, patient_id, status) VALUES (:s, :p, 'closed')"),
                           {"s": old_database.slots[0], "p": old_database.doctor_user})
        db.session.commit()
    db.session.rollback()


def test_migrating_twice_changes_nothing(old_database):
    from db import migrations

    with db.engine.begin() as conn:
        migrations.apply(conn)
    with db.engine.begin() as conn:
        assert migrations.apply(conn) == []


def test_a_fresh_database_needs_no_migration(app, engine_name):
    if engine_name != "mysql":
        pytest.skip("migrations apply to MySQL databases")
    from db import migrations

    with db.engine.begin() as conn:
        assert migrations.apply(conn) == []
