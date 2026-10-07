"""The standby list: a patient waiting for a slot on a day that is full (PLAN section 6, the stretch item).

Standby never books, holds or moves anything. A patient on standby for a doctor and a day is simply told, on their own
pages, when a slot on that day is free again; they then book it the normal way, and whoever books first gets it, because
the database's one-booking-per-slot rule is the only thing that hands out slots. Staff see who is waiting. Nothing in
app/services.py knows the list exists.
"""
from __future__ import annotations

import ast
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app import clock, services, standby
from app.extensions import db
from helpers import post
from helpers import text as body

DAY = date(2030, 1, 7)


def waiting(patient_id=None) -> int:
    sql = "SELECT COUNT(*) FROM standby" + (" WHERE patient_id = :p" if patient_id else "")
    with db.engine.connect() as conn:
        return conn.execute(text(sql), {"p": patient_id} if patient_id else {}).scalar_one()


def fill_day(world, scene) -> list:
    """Book the three slots Alice does not hold, with three new patients; returns their booking ids."""
    return [services.book(slot, world.patient()).id for slot in scene.slots[1:]]


def join(client, scene, doctor=None, day=None):
    return post(client, "/patient/standby", doctor_id=doctor or scene.doctor, date=day or scene.day)


# ---- joining ----------------------------------------------------------------------------------------------------
def test_a_patient_can_join_standby_for_a_full_day(scene, world, client):
    fill_day(world, scene)
    world.login(client, scene.bob)
    assert "Join the standby list" in body(client.get(f"/patient/doctors/{scene.doctor}?date={scene.day}"))
    response = join(client, scene)
    assert response.status_code == 302 and waiting(scene.bob) == 1
    mine = body(client.get("/patient/bookings"))
    assert "Standby" in mine and "Test doctor" in mine


def test_a_day_with_a_free_slot_has_no_standby(scene, world, client):
    world.login(client, scene.bob)
    assert "Join the standby list" not in body(client.get(f"/patient/doctors/{scene.doctor}?date={scene.day}"))
    response = join(client, scene)
    assert response.status_code == 409 and "free" in body(response).lower() and waiting() == 0


def test_joining_twice_keeps_one_entry(scene, world, client):
    fill_day(world, scene)
    world.login(client, scene.bob)
    assert join(client, scene).status_code == 302
    assert join(client, scene).status_code == 302
    assert waiting(scene.bob) == 1


def test_the_database_refuses_a_duplicate_entry(scene, world):
    insert = "INSERT INTO standby (patient_id, doctor_id, slot_date) VALUES (:p, :d, :day)"
    db.session.execute(text(insert), {"p": scene.bob, "d": scene.doctor, "day": DAY})
    db.session.commit()
    with pytest.raises(IntegrityError):
        db.session.execute(text(insert), {"p": scene.bob, "d": scene.doctor, "day": DAY})
        db.session.commit()
    db.session.rollback()


def test_there_is_nothing_to_wait_for_on_a_day_without_slots_or_in_the_past(scene, world, client, monkeypatch):
    fill_day(world, scene)
    world.login(client, scene.bob)
    assert join(client, scene, day="2030-01-09").status_code == 409  # the doctor has no slots that day
    assert join(client, scene, doctor=999_999).status_code == 404
    assert join(client, scene, day="not a date").status_code == 400
    monkeypatch.setattr(clock, "now", lambda: datetime(2030, 1, 7, 14, 0))  # every slot of the day has started
    assert join(client, scene).status_code == 409
    monkeypatch.setattr(clock, "now", lambda: datetime(2030, 1, 8, 9, 0))
    assert join(client, scene).status_code == 409
    assert waiting() == 0


def test_a_patient_can_wait_for_at_most_ten_days_at_once(scene, world, client):
    world.login(client, scene.bob)
    for i in range(standby.MAX_WAITING + 1):
        day = DAY + timedelta(days=10 + i)
        for slot in world.slots(scene.doctor, 1, day):
            services.book(slot, world.patient())
        response = join(client, scene, day=day.isoformat())
        assert response.status_code == (302 if i < standby.MAX_WAITING else 409), i
    assert waiting(scene.bob) == standby.MAX_WAITING


def test_a_day_closed_for_leave_can_be_waited_for(scene, world, client):
    for slot in scene.slots[1:]:
        services.close_slot(slot, scene.doctor_user)  # Alice's slot is booked, the rest are closed: nothing is free
    world.login(client, scene.bob)
    assert join(client, scene).status_code == 302
    assert "A slot is free" not in body(client.get("/patient/bookings"))
    services.reopen_slot(scene.slots[1], scene.doctor_user)
    assert "A slot is free" in body(client.get("/patient/bookings"))


# ---- leaving ------------------------------------------------------------------------------------------------------
def test_a_patient_can_leave_and_only_their_own_entry(scene, world, client, app):
    fill_day(world, scene)
    world.login(client, scene.bob)
    join(client, scene)
    with db.engine.connect() as conn:
        entry = conn.execute(text("SELECT id FROM standby")).scalar_one()
    stranger = app.test_client()
    world.login(stranger, world.patient())
    assert post(stranger, f"/patient/standby/{entry}/leave").status_code == 404 and waiting() == 1
    assert post(client, f"/patient/standby/{entry}/leave").status_code == 302 and waiting() == 0
    assert post(client, f"/patient/standby/{entry}/leave").status_code == 404


# ---- being told ----------------------------------------------------------------------------------------------------
def test_the_patient_is_told_when_a_slot_frees_and_not_before(scene, world, client):
    booked = fill_day(world, scene)
    world.login(client, scene.bob)
    join(client, scene)
    assert "A slot is free" not in body(client.get("/patient/bookings"))
    assert "A slot is free" not in body(client.get("/patient/"))
    services.cancel(booked[0], scene.admin)
    for url in ("/patient/bookings", "/patient/"):
        page = body(client.get(url))
        assert "A slot is free" in page and f"/patient/doctors/{scene.doctor}?date={scene.day}" in page, url


def test_the_notice_goes_away_when_someone_else_takes_the_slot_and_the_entry_stays(scene, world, client):
    booked = fill_day(world, scene)
    world.login(client, scene.bob)
    join(client, scene)
    services.cancel(booked[0], scene.admin)
    services.book(scene.slots[1], world.patient())  # standby gives no priority: first to book gets it
    page = body(client.get("/patient/bookings"))
    assert "A slot is free" not in page and waiting(scene.bob) == 1


def test_booking_with_that_doctor_on_that_day_ends_the_wait(scene, world, client):
    booked = fill_day(world, scene)
    world.login(client, scene.bob)
    join(client, scene)
    services.cancel(booked[0], scene.admin)
    assert post(client, "/patient/book", slot_id=scene.slots[1]).status_code == 302
    assert waiting(scene.bob) == 0


def test_a_past_day_drops_off_the_list(scene, world, client, monkeypatch):
    fill_day(world, scene)
    world.login(client, scene.bob)
    join(client, scene)
    assert len(standby.for_patient(scene.bob, datetime(2030, 1, 7, 8, 0))) == 1
    monkeypatch.setattr(clock, "now", lambda: datetime(2030, 1, 9, 9, 0))
    assert standby.for_patient(scene.bob, clock.now()) == []
    assert "Leave standby" not in body(client.get("/patient/bookings"))


# ---- who sees what -------------------------------------------------------------------------------------------------
def test_staff_see_who_is_waiting_in_the_order_they_joined(scene, world, client, app):
    fill_day(world, scene)
    first, second = world.patient("First Waiting"), world.patient("Second Waiting")
    for patient in (first, second):
        other = app.test_client()
        world.login(other, patient)
        join(other, scene)
    world.login(client, scene.doctor_user)
    page = body(client.get(f"/doctor/?date={scene.day}"))
    assert "On standby for this day" in page
    assert page.index("First Waiting") < page.index("Second Waiting")
    assert "On standby for this day" not in body(client.get("/doctor/?date=2030-01-08"))
    admin = app.test_client()
    world.login(admin, scene.admin)
    assert "on standby" in body(admin.get("/admin/")).lower()


def test_a_patient_never_sees_who_else_is_waiting(scene, world, client, app):
    fill_day(world, scene)
    other = app.test_client()
    world.login(other, world.patient("Someone Private"))
    join(other, scene)
    world.login(client, scene.bob)
    join(client, scene)
    for url in ("/patient/", "/patient/bookings", f"/patient/doctors/{scene.doctor}?date={scene.day}"):
        assert "Someone Private" not in body(client.get(url)), url


def test_another_doctor_does_not_see_this_doctors_standby_list(scene, world, client, app):
    fill_day(world, scene)
    world.login(client, scene.bob)
    join(client, scene)
    other = app.test_client()
    world.login(other, world.doctor_user_ids[scene.other_doctor])
    assert "Bob Patient" not in body(other.get(f"/doctor/?date={scene.day}"))


def test_only_patients_can_join(scene, world, client, app):
    fill_day(world, scene)
    assert join(client, scene).status_code == 302  # anonymous: to the login page
    for who in (scene.doctor_user, scene.admin):
        other = app.test_client()
        world.login(other, who)
        assert join(other, scene).status_code == 403
    assert waiting() == 0


# ---- it never touches booking -------------------------------------------------------------------------------------
def test_standby_gives_no_right_to_the_slot(scene, world, client):
    booked = fill_day(world, scene)
    world.login(client, scene.bob)
    join(client, scene)
    services.cancel(booked[0], scene.admin)
    assert services.book(scene.slots[1], world.patient()).status == "confirmed"  # anyone may take it
    assert post(client, "/patient/book", slot_id=scene.slots[1]).status_code == 409  # Bob, on standby, was too late


def test_the_booking_service_does_not_know_the_standby_list_exists():
    tree = ast.parse(Path(services.__file__).read_text())
    names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)] + [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    names += [a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names]
    assert not [n for n in names if "standby" in n.lower()]


def test_the_schema_has_the_standby_table_in_both_places():
    from db import schema_statements

    assert "standby" in db.metadata.tables
    assert any(s.startswith("CREATE TABLE standby") for s in schema_statements())
    assert [c.name for c in db.metadata.tables["standby"].columns] == ["id", "patient_id", "doctor_id", "slot_date", "created_at"]
