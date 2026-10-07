"""Closing slots (a doctor's leave) without ever weakening the no-double-booking rule (CLAUDE.md hard rule 1).

A closed slot is a row in `bookings` with status 'closed', held in the doctor's own name. It occupies the slot through
the SAME unique key (bookings.confirmed_slot_id) that stops two patients sharing a slot, so the database decides every
race between "a patient books it" and "the doctor closes it": exactly one wins, and services.book() is still a plain
INSERT with no extra check. Reopening cancels that row, which frees the slot like any cancellation.
"""
from __future__ import annotations

import threading
from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app import clock, scheduling, services
from app.extensions import db
from app.models import BOOKING_STATUSES, Booking
from db import SCHEMA_PATH
from helpers import post, slot_ids
from helpers import text as body

DAY = date(2030, 1, 7)


def scalar(sql: str, **params):
    with db.engine.connect() as conn:  # a fresh connection: never a stale snapshot
        return conn.execute(text(sql), params).scalar_one()


def active_rows(slot_id) -> int:
    return scalar("SELECT COUNT(*) FROM bookings WHERE slot_id = :s AND status <> 'cancelled'", s=slot_id)


def status_of(slot_id):
    return scalar("SELECT status FROM bookings WHERE slot_id = :s AND status <> 'cancelled'", s=slot_id)


def run_concurrently(app, jobs):
    barrier = threading.Barrier(len(jobs))
    results = [None] * len(jobs)

    def target(i, job):
        with app.app_context():
            barrier.wait(timeout=30)
            results[i] = job()

    threads = [threading.Thread(target=target, args=(i, job)) for i, job in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads), "a thread hung"
    return results


def try_book(slot_id, patient_id) -> str:
    try:
        services.book(slot_id, patient_id)
        return "booked"
    except services.SlotTaken:
        return "slot_taken"
    except Exception as exc:  # noqa: BLE001 - the test must see anything unexpected
        return f"error: {exc!r}"


def try_close(slot_id, actor_id) -> str:
    try:
        services.close_slot(slot_id, actor_id)
        return "closed"
    except services.SlotTaken:
        return "slot_taken"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc!r}"


# ---- the schema knows the new status ---------------------------------------------------------------------------
def test_closed_is_a_booking_status_in_the_models_and_in_schema_sql():
    assert "closed" in BOOKING_STATUSES
    line = next(l for l in SCHEMA_PATH.read_text().splitlines() if l.strip().startswith("status "))
    assert all(f"'{s}'" in line for s in BOOKING_STATUSES), line


# ---- the service -------------------------------------------------------------------------------------------------
def test_closing_a_free_slot_holds_it_and_a_patient_can_no_longer_book_it(scene, world):
    row = services.close_slot(scene.slots[1], scene.doctor_user)
    assert row.status == "closed" and row.patient_id == scene.doctor_user  # held in the doctor's own name
    with pytest.raises(services.SlotTaken):
        services.book(scene.slots[1], scene.bob)
    assert active_rows(scene.slots[1]) == 1 and status_of(scene.slots[1]) == "closed"


def test_an_admin_closing_a_slot_still_holds_it_in_the_doctors_name(scene, world):
    row = services.close_slot(scene.slots[1], scene.admin)
    assert row.patient_id == scene.doctor_user


def test_a_booked_slot_cannot_be_closed_and_nothing_changes(scene, world):
    with pytest.raises(services.SlotTaken):
        services.close_slot(scene.slots[0], scene.doctor_user)  # Alice holds it
    assert status_of(scene.slots[0]) == "confirmed" and active_rows(scene.slots[0]) == 1


def test_a_slot_cannot_be_closed_twice(scene, world):
    services.close_slot(scene.slots[1], scene.doctor_user)
    with pytest.raises(services.SlotTaken):
        services.close_slot(scene.slots[1], scene.doctor_user)
    assert active_rows(scene.slots[1]) == 1


def test_reopening_frees_the_slot_for_patients(scene, world):
    services.close_slot(scene.slots[1], scene.doctor_user)
    services.reopen_slot(scene.slots[1], scene.doctor_user)
    assert active_rows(scene.slots[1]) == 0
    assert services.book(scene.slots[1], scene.bob).status == "confirmed"


def test_reopening_a_slot_that_is_not_closed_is_refused(scene, world):
    with pytest.raises(services.InvalidState):
        services.reopen_slot(scene.slots[1], scene.doctor_user)  # free
    with pytest.raises(services.InvalidState):
        services.reopen_slot(scene.slots[0], scene.doctor_user)  # booked by Alice: reopening must never cancel a patient
    assert status_of(scene.slots[0]) == "confirmed"


def test_unknown_slots_are_not_found(scene, world):
    with pytest.raises(services.NotFound):
        services.close_slot(999_999, scene.doctor_user)
    with pytest.raises(services.NotFound):
        services.reopen_slot(999_999, scene.doctor_user)


def test_a_slot_that_has_started_cannot_be_closed(scene, world, monkeypatch):
    monkeypatch.setattr(clock, "now", lambda: datetime(2030, 1, 7, 9, 20))  # 09:00 and 09:15 have started
    with pytest.raises(services.SlotInPast):
        services.close_slot(scene.slots[1], scene.doctor_user)
    assert services.close_slot(scene.slots[2], scene.doctor_user).status == "closed"


def test_a_closed_row_is_not_an_appointment(scene, world, monkeypatch):
    closed = services.close_slot(scene.slots[1], scene.doctor_user).id
    with pytest.raises(services.InvalidState):
        services.cancel(closed, scene.doctor_user)  # reopened, not cancelled like an appointment
    with pytest.raises(services.InvalidState):
        services.reschedule(closed, scene.slots[2], scene.doctor_user)
    monkeypatch.setattr(clock, "now", lambda: datetime(2030, 1, 8, 9, 0))
    with pytest.raises(services.InvalidState):
        services.mark_outcome(closed, "no_show")
    assert status_of(scene.slots[1]) == "closed"


# ---- the database decides every race ----------------------------------------------------------------------------
def test_a_patient_booking_and_a_doctor_closing_the_same_slot_give_exactly_one_winner(app, world):
    rounds = 20
    doctor = world.doctor()
    slots, patient, doctor_user = world.slots(doctor, rounds), world.patient(), world.doctor_user_ids[doctor]
    winners = set()
    for slot in slots:
        outcomes = run_concurrently(app, [lambda s=slot: try_book(s, patient), lambda s=slot: try_close(s, doctor_user)])
        assert sorted(outcomes) in (["booked", "slot_taken"], ["closed", "slot_taken"]), outcomes
        assert active_rows(slot) == 1
        winners.add(next(o for o in outcomes if o != "slot_taken"))
    assert winners <= {"booked", "closed"}


def test_many_patients_and_a_closing_doctor_across_25_slots_give_one_holder_per_slot(app, world):
    doctor = world.doctor()
    slots, doctor_user = world.slots(doctor, 25), world.doctor_user_ids[doctor]
    patients = [world.patient() for _ in range(5)]
    jobs = [lambda p=p: [try_book(s, p) for s in slots] for p in patients]
    jobs.append(lambda: [try_close(s, doctor_user) for s in slots])
    outcomes = [o for per_actor in run_concurrently(app, jobs) for o in per_actor]
    assert not [o for o in outcomes if o.startswith("error")], outcomes
    assert outcomes.count("booked") + outcomes.count("closed") == 25
    assert outcomes.count("slot_taken") == 6 * 25 - 25
    assert all(active_rows(s) == 1 for s in slots)


INSERT = "INSERT INTO bookings (slot_id, patient_id, status) VALUES (:s, :p, :status)"


def test_direct_sql_cannot_put_a_closed_row_beside_an_active_booking(scene, world):
    with pytest.raises(IntegrityError):
        db.session.execute(text(INSERT), {"s": scene.slots[0], "p": scene.doctor_user, "status": "closed"})
        db.session.commit()
    db.session.rollback()
    assert active_rows(scene.slots[0]) == 1


@pytest.mark.parametrize("status", ["confirmed", "completed", "no_show", "closed"])
def test_direct_sql_cannot_put_anything_active_beside_a_closed_row(scene, world, status):
    services.close_slot(scene.slots[1], scene.doctor_user)
    with pytest.raises(IntegrityError):
        db.session.execute(text(INSERT), {"s": scene.slots[1], "p": scene.bob, "status": status})
        db.session.commit()
    db.session.rollback()
    assert active_rows(scene.slots[1]) == 1


# ---- a day or a range at once -----------------------------------------------------------------------------------
def test_closing_a_day_closes_the_free_slots_and_leaves_the_booked_ones(scene, world):
    result = scheduling.close_free_slots(scene.doctor, DAY, DAY, scene.doctor_user)
    assert (result.closed, result.booked) == (3, 1)
    assert status_of(scene.slots[0]) == "confirmed"
    assert [status_of(s) for s in scene.slots[1:]] == ["closed"] * 3
    again = scheduling.close_free_slots(scene.doctor, DAY, DAY, scene.doctor_user)
    assert (again.closed, again.booked) == (0, 1)  # safe to run twice


def test_reopening_a_range_frees_only_the_closed_slots(scene, world):
    scheduling.close_free_slots(scene.doctor, DAY, DAY, scene.doctor_user)
    assert scheduling.reopen_slots(scene.doctor, DAY, DAY, scene.doctor_user) == 3
    assert status_of(scene.slots[0]) == "confirmed"  # Alice's appointment is untouched
    assert all(active_rows(s) == 0 for s in scene.slots[1:])
    assert scheduling.reopen_slots(scene.doctor, DAY, DAY, scene.doctor_user) == 0


def test_only_slots_that_have_not_started_are_closed(scene, world, monkeypatch):
    monkeypatch.setattr(clock, "now", lambda: datetime(2030, 1, 7, 9, 20))
    result = scheduling.close_free_slots(scene.doctor, DAY, DAY, scene.doctor_user)
    assert result.closed == 2  # 09:30 and 09:45; 09:15 has started
    assert active_rows(scene.slots[1]) == 0


def test_a_range_covers_several_days_and_only_this_doctor(scene, world):
    next_day = world.slots(scene.doctor, 2, DAY + timedelta(days=1))
    other = world.slots(scene.other_doctor, 2, DAY)
    result = scheduling.close_free_slots(scene.doctor, DAY, DAY + timedelta(days=1), scene.doctor_user)
    assert result.closed == 5
    assert all(status_of(s) == "closed" for s in next_day) and all(active_rows(s) == 0 for s in other)


def test_a_range_must_be_in_order_and_not_too_long(scene, world):
    with pytest.raises(scheduling.ScheduleError):
        scheduling.close_free_slots(scene.doctor, DAY, DAY - timedelta(days=1), scene.doctor_user)
    with pytest.raises(scheduling.ScheduleError):
        scheduling.close_free_slots(scene.doctor, DAY, DAY + timedelta(days=scheduling.MAX_DAYS), scene.doctor_user)
    with pytest.raises(scheduling.ScheduleError):
        scheduling.reopen_slots(scene.doctor, DAY, DAY - timedelta(days=1), scene.doctor_user)


# ---- the pages ----------------------------------------------------------------------------------------------------
def test_a_doctor_closes_and_reopens_one_slot_from_the_schedule(scene, world, client):
    world.login(client, scene.doctor_user)
    page = body(client.get(f"/doctor/?date={scene.day}"))
    assert f"/doctor/slots/{scene.slots[1]}/close" in page  # a free slot offers Close
    assert f"/doctor/slots/{scene.slots[0]}/close" not in page  # a booked one does not
    assert post(client, f"/doctor/slots/{scene.slots[1]}/close").status_code == 302
    page = body(client.get(f"/doctor/?date={scene.day}"))
    assert "Closed" in page and f"/doctor/slots/{scene.slots[1]}/reopen" in page
    assert post(client, f"/doctor/slots/{scene.slots[1]}/reopen").status_code == 302
    assert active_rows(scene.slots[1]) == 0


def test_the_leave_form_closes_and_reopens_a_range(scene, world, client):
    world.login(client, scene.doctor_user)
    response = post(client, "/doctor/leave", date_from=scene.day, date_to=scene.day, action="close")
    assert response.status_code == 302
    page = body(client.get(f"/doctor/?date={scene.day}"))
    assert "Closed 3 free slot" in page and "1 booked appointment" in page
    assert post(client, "/doctor/leave", date_from=scene.day, date_to=scene.day, action="reopen").status_code == 302
    assert "Reopened 3 slot" in body(client.get(f"/doctor/?date={scene.day}"))


@pytest.mark.parametrize("form", [{"date_from": "nonsense", "date_to": "2030-01-07", "action": "close"},
                                  {"date_from": "2030-01-08", "date_to": "2030-01-07", "action": "close"},
                                  {"date_from": "2030-01-07", "date_to": "2030-01-07", "action": "explode"}])
def test_a_bad_leave_form_is_refused_with_a_message(form, scene, world, client):
    world.login(client, scene.doctor_user)
    response = post(client, "/doctor/leave", **form)
    assert response.status_code == 400
    assert all(active_rows(s) == 0 for s in scene.slots[1:])


def test_a_doctor_cannot_close_another_doctors_slot(scene, world, client):
    [theirs] = world.slots(scene.other_doctor, 1)
    world.login(client, scene.doctor_user)
    assert post(client, f"/doctor/slots/{theirs}/close").status_code == 404
    assert post(client, f"/doctor/slots/{theirs}/reopen").status_code == 404
    assert active_rows(theirs) == 0


def test_an_admin_closes_and_reopens_any_doctors_slots(scene, world, client):
    world.login(client, scene.admin)
    assert "Close free slots" in body(client.get(f"/admin/doctors/{scene.doctor}/slots"))
    response = post(client, f"/admin/doctors/{scene.doctor}/leave", date_from=scene.day, date_to=scene.day, action="close")
    assert response.status_code == 302 and [status_of(s) for s in scene.slots[1:]] == ["closed"] * 3
    assert post(client, f"/admin/doctors/{scene.doctor}/leave", date_from=scene.day, date_to=scene.day, action="reopen").status_code == 302
    assert all(active_rows(s) == 0 for s in scene.slots[1:])
    assert post(client, "/admin/doctors/999999/leave", date_from=scene.day, date_to=scene.day, action="close").status_code == 404


def test_patients_and_anonymous_visitors_cannot_close_anything(scene, world, client, app):
    assert post(client, f"/doctor/slots/{scene.slots[1]}/close").status_code == 302  # anonymous: sent to the login page
    world.login(client, scene.bob)
    for url in (f"/doctor/slots/{scene.slots[1]}/close", "/doctor/leave", f"/admin/doctors/{scene.doctor}/leave"):
        assert post(client, url, date_from=scene.day, date_to=scene.day, action="close").status_code == 403
    assert all(active_rows(s) == 0 for s in scene.slots[1:])


def test_a_patient_sees_a_closed_slot_as_closed_and_cannot_book_it(scene, world, client):
    services.close_slot(scene.slots[1], scene.doctor_user)
    world.login(client, scene.bob)
    page = body(client.get(f"/patient/doctors/{scene.doctor}?date={scene.day}"))
    assert scene.slots[1] not in slot_ids(page) and "Closed" in page
    assert scene.slots[2] in slot_ids(page)
    response = post(client, "/patient/book", slot_id=scene.slots[1])  # a stale page, or a hand-made request
    assert response.status_code == 409 and status_of(scene.slots[1]) == "closed"


def test_a_closed_slot_is_not_offered_when_rescheduling(scene, world, client):
    services.close_slot(scene.slots[1], scene.doctor_user)
    world.login(client, scene.alice)
    page = body(client.get(f"/patient/bookings/{scene.booking}/reschedule?date={scene.day}"))
    assert scene.slots[1] not in slot_ids(page)
    assert post(client, f"/patient/bookings/{scene.booking}/reschedule", new_slot_id=scene.slots[1]).status_code == 409
    db.session.expire_all()
    assert db.session.get(Booking, scene.booking).status == "confirmed"


def test_closed_slots_are_not_appointments_anywhere_staff_look(scene, world, client, app):
    world.login(client, scene.admin)
    before = body(client.get("/admin/"))
    services.close_slot(scene.slots[1], scene.doctor_user)
    services.close_slot(scene.slots[2], scene.doctor_user)
    after = body(client.get("/admin/"))
    numbers = lambda page: __import__("re").findall(r'<div class="stat"><strong>(\d+)</strong>', page)  # noqa: E731
    assert numbers(before)[:3] == numbers(after)[:3]  # patients, doctors, upcoming appointments
    listing = body(client.get("/admin/bookings"))
    assert "1 booking(s)" in listing  # Alice's only
    doctor = app.test_client()
    world.login(doctor, scene.doctor_user)
    assert "1/4 booked" in body(doctor.get(f"/doctor/?date={scene.day}"))  # closed slots are not counted as booked


def test_the_outcome_and_cancel_buttons_do_not_work_on_a_closed_row(scene, world, client, monkeypatch):
    closed = services.close_slot(scene.slots[1], scene.doctor_user).id
    world.login(client, scene.doctor_user)
    assert post(client, f"/doctor/bookings/{closed}/cancel").status_code == 302  # refused with a message, not an error
    assert status_of(scene.slots[1]) == "closed"
    monkeypatch.setattr(clock, "now", lambda: datetime(2030, 1, 8, 9, 0))
    assert post(client, f"/doctor/bookings/{closed}/outcome", outcome="no_show").status_code == 302
    assert status_of(scene.slots[1]) == "closed"


def test_a_closed_row_is_never_scored_and_never_counts_as_history(scene, world, client, app):
    from pathlib import Path

    from app import risk

    real = Path(__file__).resolve().parents[1] / "ml" / "artifacts"
    if not (real / "risk_model.joblib").exists():
        pytest.skip("ml/artifacts/risk_model.joblib missing")
    app.config["RISK_ARTIFACTS_DIR"] = real
    risk.reset_cache()
    closed = services.close_slot(scene.slots[1], scene.doctor_user).id
    world.login(client, scene.doctor_user)
    assert client.get(f"/doctor/?date={scene.day}").status_code == 200
    assert scalar("SELECT COUNT(*) FROM risk_scores WHERE booking_id = :b", b=closed) == 0
    assert risk._history(scene.doctor_user, 0) == []
    risk.reset_cache()
