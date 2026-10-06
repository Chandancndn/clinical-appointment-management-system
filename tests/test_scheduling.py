"""Slot generation: idempotent, weekends skippable, windows validated; doctors create their own day."""
from __future__ import annotations

from datetime import date, time

import pytest

from app import scheduling
from app.extensions import db
from app.models import Slot
from helpers import post, text

FRI, MON = date(2030, 1, 4), date(2030, 1, 7)


def slot_count(doctor_id=None):
    db.session.expire_all()
    query = db.session.query(Slot)
    return (query.filter_by(doctor_id=doctor_id) if doctor_id else query).count()


def test_generating_twice_creates_no_duplicates(world):
    doctor = world.doctor(slot_minutes=15)
    first = scheduling.generate_slots(doctor, MON, MON, time(9, 0), time(13, 0))
    assert (first.created, first.existing) == (16, 0)  # 4 hours of 15-minute slots
    second = scheduling.generate_slots(doctor, MON, MON, time(9, 0), time(13, 0))
    assert (second.created, second.existing) == (0, 16)
    assert slot_count(doctor) == 16


def test_a_wider_second_run_only_adds_the_missing_slots(world):
    doctor = world.doctor(slot_minutes=30)
    scheduling.generate_slots(doctor, MON, MON, time(9, 0), time(11, 0))  # 4 slots
    result = scheduling.generate_slots(doctor, MON, MON, time(9, 0), time(12, 0))  # 6 slots, 4 exist
    assert (result.created, result.existing) == (2, 4)


def test_weekends_are_skipped_when_asked(world):
    doctor = world.doctor(slot_minutes=60)
    result = scheduling.generate_slots(doctor, FRI, MON, time(9, 0), time(10, 0), skip_weekends=True)
    assert result.created == 2  # Friday and Monday only
    db.session.expire_all()
    days = {s.slot_date for s in db.session.query(Slot).filter_by(doctor_id=doctor)}
    assert days == {FRI, MON}
    scheduling.generate_slots(doctor, FRI, MON, time(9, 0), time(10, 0), skip_weekends=False)
    assert slot_count(doctor) == 4  # now the weekend too


def test_a_partial_last_slot_is_not_created(world):
    doctor = world.doctor(slot_minutes=20)
    scheduling.generate_slots(doctor, MON, MON, time(9, 0), time(10, 10))
    db.session.expire_all()
    times = sorted(s.slot_time for s in db.session.query(Slot).filter_by(doctor_id=doctor))
    assert times == [time(9, 0), time(9, 20), time(9, 40)]  # 10:00 would end at 10:20, after 10:10


@pytest.mark.parametrize("args, message", [
    ((MON, FRI, time(9), time(13)), "before the first"),
    ((MON, MON, time(9), time(9, 10)), "shorter than one"),
    ((MON, date(2030, 6, 1), time(9), time(13)), "at most"),
])
def test_nonsense_ranges_are_refused(world, args, message):
    doctor = world.doctor(slot_minutes=15)
    with pytest.raises(scheduling.ScheduleError, match=message):
        scheduling.generate_slots(doctor, *args)
    assert slot_count(doctor) == 0


def test_slots_of_other_doctors_are_untouched(world):
    one, two = world.doctor(), world.doctor()
    scheduling.generate_slots(one, MON, MON, time(9), time(10))
    scheduling.generate_slots(two, MON, MON, time(9), time(10))
    assert slot_count(one) == slot_count(two) == 4


def test_a_doctor_creates_slots_for_a_day_and_running_again_adds_none(scene, world, client):
    world.login(client, scene.doctor_user)
    day = "2031-03-03"
    response = post(client, "/doctor/slots", date=day, start_time="10:00", end_time="11:00")
    assert response.status_code == 302 and response.headers["Location"].endswith(f"date={day}")
    assert slot_count(scene.doctor) == 4 + 4
    page = text(client.get(f"/doctor/?date={day}"))
    assert "10:00" in page and "10:45" in page
    post(client, "/doctor/slots", date=day, start_time="10:00", end_time="11:00")
    assert slot_count(scene.doctor) == 4 + 4  # nothing duplicated


def test_a_doctor_cannot_create_slots_in_the_past_or_with_a_bad_window(scene, world, client):
    world.login(client, scene.doctor_user)
    past = post(client, "/doctor/slots", date="2001-01-01", start_time="09:00", end_time="10:00")
    assert past.status_code == 400 and "today or a later date" in text(past)
    backwards = post(client, "/doctor/slots", date="2031-03-03", start_time="11:00", end_time="10:00")
    assert backwards.status_code == 400 and "after the start" in text(backwards)
    assert slot_count(scene.doctor) == 4
