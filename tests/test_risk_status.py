"""M7+: staff can see whether the flag is on, which model it uses, and how it is doing on recorded outcomes.

The panel is for admins only (patients and doctors never get it), is advisory, and an off or failing model only turns the
panel into an explanation: nothing else on the page or in booking changes.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app import risk
from helpers import text

REAL = Path(__file__).resolve().parents[1] / "ml" / "artifacts"
APP_CONFIG = {"RISK_ARTIFACTS_DIR": REAL}
pytestmark = pytest.mark.skipif(not (REAL / "risk_model.joblib").exists(), reason="ml/artifacts/risk_model.joblib missing")
CARD = json.loads((REAL / "model_card.json").read_text()) if (REAL / "model_card.json").exists() else {}


@pytest.fixture(autouse=True)
def fresh_cache():
    risk.reset_cache()
    yield
    risk.reset_cache()


def past_booking(world, scene, outcome, probability, day):
    """A finished appointment for a fresh patient with a stored score."""
    slot = world.slots(scene.doctor, 1, day)[0]
    booking = world.booking(slot, world.patient(), outcome)
    risk._write_score(booking, probability, CARD["version"])
    return booking


def test_status_describes_the_model_in_use(app):
    status = risk.status()
    assert status["on"] is True and status["version"] == CARD["version"]
    assert status["medium"] == CARD["thresholds"]["medium"]["threshold"] and status["high"] == CARD["thresholds"]["high"]["threshold"]
    assert status["trained_with"]["scikit-learn"] == CARD["environment"]["scikit-learn"]
    assert status["running"]["scikit-learn"]


def test_status_explains_why_the_flag_is_off(app, tmp_path):
    app.config["RISK_ARTIFACTS_DIR"] = tmp_path / "empty"
    risk.reset_cache()
    status = risk.status()
    assert status["on"] is False and "not found" in status["reason"].lower()


def test_a_card_that_does_not_match_the_model_file_is_explained(app, tmp_path):
    copy = tmp_path / "artifacts"
    shutil.copytree(REAL, copy)
    card = json.loads((copy / "model_card.json").read_text())
    card["artifact_sha256"] = "0" * 64
    (copy / "model_card.json").write_text(json.dumps(card))
    app.config["RISK_ARTIFACTS_DIR"] = copy
    risk.reset_cache()
    status = risk.status()
    assert status["on"] is False and "hash" in status["reason"].lower()


def test_a_different_scikit_learn_version_is_reported_but_does_not_switch_the_flag_off(app, monkeypatch):
    monkeypatch.setattr(risk, "_runtime_versions", lambda: {**CARD["environment"], "scikit-learn": "0.0.1"})
    risk.reset_cache()
    status = risk.status()
    assert status["on"] is True and "0.0.1" in status["note"] and CARD["environment"]["scikit-learn"] in status["note"]


def test_the_same_versions_give_no_note(app, monkeypatch):
    monkeypatch.setattr(risk, "_runtime_versions", lambda: dict(CARD["environment"]))
    risk.reset_cache()
    assert risk.status()["on"] is True and risk.status()["note"] is None


def test_the_monitor_counts_outcomes_by_band(scene, world, app):
    from datetime import date, timedelta

    base = date(2029, 1, 1)
    medium, high = CARD["thresholds"]["medium"]["threshold"], CARD["thresholds"]["high"]["threshold"]
    for i, (outcome, p) in enumerate([("no_show", high + 0.05), ("no_show", high + 0.01), ("completed", high + 0.2),
                                       ("completed", medium + 0.01), ("no_show", medium + 0.01),
                                       ("completed", 0.05), ("completed", 0.06), ("completed", 0.07), ("no_show", 0.01)]):
        past_booking(world, scene, outcome, p, base + timedelta(days=i))
    world.booking(world.slots(scene.doctor, 1, date(2029, 3, 1))[0], world.patient(), "confirmed")  # no outcome yet: not counted
    rows = {r["band"]: r for r in risk.monitor()}
    assert [r["band"] for r in risk.monitor()] == ["low", "medium", "high"]
    assert (rows["high"]["bookings"], rows["high"]["no_shows"]) == (3, 2) and rows["high"]["rate"] == pytest.approx(2 / 3)
    assert (rows["medium"]["bookings"], rows["medium"]["no_shows"]) == (2, 1) and rows["medium"]["rate"] == pytest.approx(0.5)
    assert (rows["low"]["bookings"], rows["low"]["no_shows"]) == (4, 1) and rows["low"]["rate"] == pytest.approx(0.25)


def test_the_monitor_ignores_scores_from_another_model_version(scene, world, app):
    from datetime import date

    slot = world.slots(scene.doctor, 1, date(2029, 1, 1))[0]
    booking = world.booking(slot, world.patient(), "no_show")
    risk._write_score(booking, 0.9, "some-older-version")
    assert all(r["bookings"] == 0 for r in risk.monitor())


def test_the_monitor_with_nothing_to_count_has_no_rates(app):
    rows = risk.monitor()
    assert [r["band"] for r in rows] == ["low", "medium", "high"] and all(r["rate"] is None for r in rows)


def test_the_monitor_is_none_without_a_model(app, tmp_path):
    app.config["RISK_ARTIFACTS_DIR"] = tmp_path / "empty"
    risk.reset_cache()
    assert risk.monitor() is None


def test_admins_see_the_panel_and_nobody_else_does(scene, world, client, app):
    world.login(client, scene.admin)
    page = text(client.get("/admin/"))
    assert "Risk flag status" in page and CARD["version"] in page and "advisory" in page.lower()
    for url in ("/doctor/", "/patient/", "/patient/bookings"):
        other = app.test_client()
        world.login(other, scene.doctor_user if url == "/doctor/" else scene.alice)
        assert "Risk flag status" not in text(other.get(url))


def test_the_panel_explains_an_off_flag_and_the_page_still_works(scene, world, client, app, tmp_path):
    app.config["RISK_ARTIFACTS_DIR"] = tmp_path / "empty"
    risk.reset_cache()
    world.login(client, scene.admin)
    response = client.get("/admin/")
    page = text(response)
    assert response.status_code == 200 and "Risk flag status" in page and "off" in page.lower()
