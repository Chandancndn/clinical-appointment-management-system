"""PLAN section 2: a slot that has started cannot be booked; a started appointment cannot be cancelled
or moved, and can only be marked completed or no_show. The clock is replaced so no test depends on today."""
from __future__ import annotations

from datetime import datetime

import pytest

from app import clock, services
from app.extensions import db
from app.models import Booking
from helpers import post, slot_ids, text

NOW = datetime(2030, 1, 7, 9, 30)  # slots are 09:00, 09:15, 09:30, 09:45 on 2030-01-07


@pytest.fixture
def at_0930(monkeypatch):
    monkeypatch.setattr(clock, "now", lambda: NOW)


def test_a_slot_that_has_started_cannot_be_booked(scene, at_0930):
    # 09:00 and 09:15 are past; 09:30 starts exactly now, which counts as started
    for slot in scene.slots[:3]:
        with pytest.raises(services.SlotInPast):
            services.book(slot, scene.bob)
    assert services.book(scene.slots[3], scene.bob).status == "confirmed"  # 09:45 is still ahead


def test_booking_a_started_slot_over_http_gives_409_and_a_message(scene, world, client, at_0930):
    world.login(client, scene.bob)
    response = post(client, "/patient/book", slot_id=scene.slots[1])
    assert response.status_code == 409
    assert "already passed" in text(response)


def test_past_slots_are_shown_as_past_and_not_offered(scene, world, client, at_0930):
    world.login(client, scene.bob)
    page = text(client.get(f"/patient/doctors/{scene.doctor}?date={scene.day}"))
    offered = slot_ids(page)
    assert offered == [scene.slots[3]]
    assert page.count('class="slot slot-past"') == 3  # 09:00, 09:15, 09:30


def test_a_started_appointment_cannot_be_cancelled(scene, at_0930):
    # alice's booking is the 09:00 slot, which started 30 minutes ago
    with pytest.raises(services.AppointmentStarted):
        services.cancel(scene.booking, actor_id=scene.alice)
    db.session.expire_all()
    assert db.session.get(Booking, scene.booking).status == "confirmed"


def test_a_started_appointment_cannot_be_moved(scene, at_0930):
    with pytest.raises(services.AppointmentStarted):
        services.reschedule(scene.booking, scene.slots[3], actor_id=scene.alice)


def test_moving_into_a_past_slot_is_refused_and_changes_nothing(scene, world, monkeypatch):
    monkeypatch.setattr(clock, "now", lambda: datetime(2030, 1, 7, 8, 0))  # everything still ahead
    mine = services.book(scene.slots[3], scene.bob)
    monkeypatch.setattr(clock, "now", lambda: NOW)  # now 09:00-09:30 have passed, 09:45 not
    with pytest.raises(services.SlotInPast):
        services.reschedule(mine.id, scene.slots[1], actor_id=scene.bob)
    db.session.expire_all()
    assert db.session.get(Booking, mine.id).status == "confirmed"


def test_an_outcome_can_only_be_recorded_once_the_appointment_started(scene, monkeypatch):
    monkeypatch.setattr(clock, "now", lambda: datetime(2030, 1, 7, 8, 0))
    with pytest.raises(services.NotStarted):
        services.mark_outcome(scene.booking, "completed")
    monkeypatch.setattr(clock, "now", lambda: NOW)
    assert services.mark_outcome(scene.booking, "completed").status == "completed"
    assert services.mark_outcome(scene.booking, "no_show").status == "no_show"  # a correction is allowed
    with pytest.raises(services.InvalidOutcome):
        services.mark_outcome(scene.booking, "cancelled")  # cancelling is not an outcome


def test_a_cancelled_booking_cannot_be_marked(scene, monkeypatch):
    monkeypatch.setattr(clock, "now", lambda: datetime(2030, 1, 7, 8, 0))
    services.cancel(scene.booking, actor_id=scene.alice)
    monkeypatch.setattr(clock, "now", lambda: NOW)
    with pytest.raises(services.InvalidState):
        services.mark_outcome(scene.booking, "completed")
