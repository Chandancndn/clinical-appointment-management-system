"""M7: the advisory no-show risk flag (CLAUDE.md hard rules 1 and 2, PLAN section 6).

(a) booking works with the model file deleted        (b) booking works when scoring raises an error
(c) patients never see risk fields in any page       (d) the flag matches the model card thresholds
(e) the existing booking and role tests still pass (the whole suite; the shared app fixture runs them with NO model)
Plus: the booking logic never reads the score, scoring starts only after the booking is committed, the flag never
changes which slots are free, the model is loaded once, and a card that does not fit the model disables the flag.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sqlalchemy import text

from app import risk, services
from app.extensions import db
from app.models import Booking, RiskScore
from helpers import form_token, post, slot_ids
from helpers import text as body
from ml.src import features

REAL = Path(__file__).resolve().parents[1] / "ml" / "artifacts"
APP_CONFIG = {"RISK_ARTIFACTS_DIR": REAL}  # this module's app uses the real saved model unless a test says otherwise
pytestmark = pytest.mark.skipif(not (REAL / "risk_model.joblib").exists(), reason="ml/artifacts/risk_model.joblib missing")

CARD = json.loads((REAL / "model_card.json").read_text()) if (REAL / "model_card.json").exists() else {}


@pytest.fixture(autouse=True)
def fresh_cache():
    risk.reset_cache()
    yield
    risk.reset_cache()


def booking_row(booking_id):
    db.session.expire_all()
    return db.session.get(Booking, booking_id)


def score_rows():
    return db.session.execute(text("SELECT COUNT(*) FROM risk_scores")).scalar_one()


def no_model(app, tmp_path):
    app.config["RISK_ARTIFACTS_DIR"] = tmp_path / "deleted"
    risk.reset_cache()


def set_score(booking_id, probability, version=None):
    risk._write_score(booking_id, probability, version or CARD["version"])


def badges(page: str) -> list[str]:
    return re.findall(r'class="risk risk-(\w+)"', page)


# ---- (a) the model file is gone ----------------------------------------------------------------------
def test_booking_works_with_the_model_file_deleted(scene, world, client, app, tmp_path):
    no_model(app, tmp_path)
    assert risk.get_model() is None
    world.profile(scene.bob)
    world.login(client, scene.bob)
    assert post(client, "/patient/book", slot_id=scene.slots[1]).status_code == 302
    booked = db.session.query(Booking).filter_by(slot_id=scene.slots[1]).one()
    assert booked.status == "confirmed" and score_rows() == 0

    staff = app.test_client()
    world.login(staff, scene.doctor_user)
    page = body(staff.get(f"/doctor/?date={scene.day}"))
    assert badges(page) == [] and "Bob Patient" in page  # the schedule still works, just without badges
    # the service layer is untouched: the same slot is still taken
    with pytest.raises(services.SlotTaken):
        services.book(scene.slots[1], scene.alice)


def test_reschedule_and_cancel_also_work_without_a_model(scene, world, client, app, tmp_path):
    no_model(app, tmp_path)
    world.login(client, scene.alice)
    assert post(client, f"/patient/bookings/{scene.booking}/reschedule", new_slot_id=scene.slots[2]).status_code == 302
    mine = db.session.query(Booking).filter_by(patient_id=scene.alice, status="confirmed").one()
    assert post(client, f"/patient/bookings/{mine.id}/cancel").status_code == 302


# ---- (b) scoring raises an error ---------------------------------------------------------------------------
class ExplodingModel:
    def predict_proba(self, X):
        raise RuntimeError("the model blew up")


@pytest.mark.parametrize("failure", ["predict", "feature_function", "storing_the_score", "reading_the_database"])
def test_booking_works_when_scoring_raises_an_error(scene, world, client, app, monkeypatch, failure):
    world.profile(scene.bob)
    if failure == "predict":
        real = risk.get_model()
        monkeypatch.setattr(risk, "get_model", lambda *a, **k: type(real)(**{**real.__dict__, "estimator": ExplodingModel()}))
    elif failure == "feature_function":
        monkeypatch.setattr(features, "deployable_features", lambda **k: (_ for _ in ()).throw(ValueError("bad features")))
    elif failure == "storing_the_score":
        monkeypatch.setattr(risk, "_write_score", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cannot write")))
    else:
        monkeypatch.setattr(risk, "_history", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("database trouble")))
    world.login(client, scene.bob)
    response = post(client, "/patient/book", slot_id=scene.slots[1])
    assert response.status_code == 302 and response.headers["Location"].endswith("/patient/bookings")
    assert booking_row(db.session.query(Booking).filter_by(slot_id=scene.slots[1]).one().id).status == "confirmed"
    assert score_rows() == 0
    staff = app.test_client()
    world.login(staff, scene.doctor_user)
    assert staff.get(f"/doctor/?date={scene.day}").status_code == 200  # staff pages survive a failing model too


def test_a_failure_while_scoring_cannot_roll_the_booking_back(scene, world, client, monkeypatch):
    """The booking is committed before scoring starts, and the scorer runs on its own transaction."""
    world.profile(scene.bob)
    seen = {}

    def spy(booking_id):
        with db.engine.connect() as other:  # a separate connection sees only COMMITTED rows
            seen["visible"] = other.execute(text("SELECT COUNT(*) FROM bookings WHERE id = :i"), {"i": booking_id}).scalar_one()
        raise RuntimeError("scoring fails after the booking is already safe")

    monkeypatch.setattr(risk, "probability_for", spy)
    world.login(client, scene.bob)
    assert post(client, "/patient/book", slot_id=scene.slots[1]).status_code == 302
    assert seen["visible"] == 1  # already committed and visible to everybody when scoring began
    assert db.session.query(Booking).filter_by(slot_id=scene.slots[1], status="confirmed").count() == 1


# ---- hard rules: the booking logic never reads the score ---------------------------------------------------------
def test_the_booking_services_do_not_know_the_risk_module_exists():
    """No import of, and no reference to, anything about risk in the booking code (prose in docstrings is fine)."""
    import ast

    tree = ast.parse(Path(services.__file__).read_text())
    imported = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    imported += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names]
    identifiers = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)] + [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    assert not [name for name in imported + identifiers if "risk" in name.lower()]
    assert not hasattr(services, "risk") and not hasattr(services, "RiskScore")


def test_the_flag_never_changes_which_slots_are_free_or_who_can_book(scene, world, client):
    world.login(client, scene.bob)
    day = f"/patient/doctors/{scene.doctor}?date={scene.day}"
    before = slot_ids(body(client.get(day)))
    for booking in (scene.booking,):
        set_score(booking, 0.99)  # a High flag on alice's booking
    assert slot_ids(body(client.get(day))) == before  # same free slots
    with pytest.raises(services.SlotTaken):  # and the slot is exactly as taken as it was
        services.book(scene.slots[0], scene.bob)
    assert post(client, "/patient/book", slot_id=scene.slots[3]).status_code == 302  # others book normally
    # removing every score changes nothing either
    db.session.execute(text("DELETE FROM risk_scores"))
    db.session.commit()
    assert slot_ids(body(client.get(day))) == [s for s in before if s != scene.slots[3]]


def test_a_high_risk_booking_is_never_refused_or_moved(scene, world, client):
    world.profile(scene.bob)
    world.login(client, scene.bob)
    assert post(client, "/patient/book", slot_id=scene.slots[1]).status_code == 302
    new = db.session.query(Booking).filter_by(slot_id=scene.slots[1]).one()
    set_score(new.id, 0.95)
    # cancelling and rebooking are unaffected by a High score
    assert post(client, f"/patient/bookings/{new.id}/cancel").status_code == 302
    assert post(client, "/patient/book", slot_id=scene.slots[1]).status_code == 302


# ---- (c) patients never see risk --------------------------------------------------------------------------------
PATIENT_TEMPLATES = ["patient", "auth", "errors"]


def test_no_patient_template_mentions_risk():
    templates = Path(services.__file__).resolve().parent / "templates"
    texts = [p.read_text() for d in PATIENT_TEMPLATES for p in (templates / d).rglob("*.html")] + [(templates / "base.html").read_text()]
    assert texts and not any(re.search(r"risk|advisory", t, re.I) for t in texts)


def test_patients_never_see_risk_fields_in_any_page(scene, world, client):
    for booking in (scene.booking,):
        set_score(booking, 0.97)  # alice's own booking carries the highest-risk flag there is
    world.login(client, scene.alice)
    pages = [
        client.get("/patient/"),
        client.get(f"/patient/doctors/{scene.doctor}?date={scene.day}"),
        client.get("/patient/bookings"),
        client.get(f"/patient/bookings/{scene.booking}/reschedule"),
    ]
    booked = post(client, "/patient/book", slot_id=scene.slots[1])  # a new booking is scored behind the scenes
    pages += [client.get(booked.headers["Location"]), client.get("/patient/bookings")]
    for response in pages:
        assert response.status_code == 200
        page = body(response)
        assert not re.search(r"risk|advisory|probab|no[- ]show risk", page, re.I), page[:200]
        assert "97%" not in page and "0.97" not in page and CARD["version"] + "</" not in page


def test_staff_do_see_the_flag_so_the_patient_test_is_not_vacuous(scene, world, client, app):
    set_score(scene.booking, 0.97)
    world.login(client, scene.doctor_user)
    page = body(client.get(f"/doctor/?date={scene.day}"))
    assert badges(page) == ["high"] and "advisory only" in page.lower()


# ---- (d) the flag is what the model card thresholds say -----------------------------------------------------------
def test_the_thresholds_come_from_the_model_card_not_from_the_code(app):
    model = risk.get_model()
    assert (model.medium, model.high) == (CARD["thresholds"]["medium"]["threshold"], CARD["thresholds"]["high"]["threshold"])
    assert model.version == CARD["version"]
    assert "0.19" not in Path(risk.__file__).read_text() and "0.38" not in Path(risk.__file__).read_text()


def test_the_band_of_a_probability_follows_the_card_thresholds_exactly(app):
    model = risk.get_model()
    medium, high = CARD["thresholds"]["medium"]["threshold"], CARD["thresholds"]["high"]["threshold"]
    cases = [(0.0, "low"), (medium - 1e-9, "low"), (medium, "medium"), ((medium + high) / 2, "medium"),
             (high - 1e-9, "medium"), (high, "high"), (0.99, "high")]
    for probability, expected in cases:
        assert risk.band_for(probability, model) == expected, probability


def test_the_badges_on_the_doctor_schedule_are_the_bands_the_card_gives(scene, world, client):
    medium, high = CARD["thresholds"]["medium"]["threshold"], CARD["thresholds"]["high"]["threshold"]
    probabilities = [0.01, medium - 1e-6, medium, (medium + high) / 2, high - 1e-6, high, 0.9]
    day = date(2031, 2, 3)
    slots = world.slots(scene.doctor, len(probabilities), day=day)
    for slot, p in zip(slots, probabilities):
        patient = world.patient()
        booking_id = world.booking(slot, patient)
        set_score(booking_id, p)
    world.login(client, scene.doctor_user)
    shown = badges(body(client.get(f"/doctor/?date={day.isoformat()}")))
    assert shown == ["low", "low", "medium", "medium", "medium", "high", "high"]


def test_admins_see_the_badges_on_the_overview_and_the_booking_list(scene, world, client):
    set_score(scene.booking, 0.9)
    other = world.booking(scene.slots[1], scene.bob)
    set_score(other, 0.25)
    world.login(client, scene.admin)
    for url in ("/admin/", "/admin/bookings"):
        page = body(client.get(url))
        assert set(badges(page)) >= {"high", "medium"}, url
        assert "advisory only" in page.lower(), url


def test_cancelled_bookings_get_no_badge(scene, world, client):
    set_score(scene.booking, 0.9)
    services.cancel(scene.booking, actor_id=scene.alice)
    world.login(client, scene.admin)
    assert badges(body(client.get("/admin/bookings"))) == []


# ---- scoring itself ------------------------------------------------------------------------------------------------
def test_a_booking_is_scored_with_exactly_the_app_feature_function_and_the_saved_model(scene, world, app):
    patient = world.patient()
    world.profile(patient, date_of_birth=date(1990, 6, 15), sex="F")
    history_slots = world.slots(scene.doctor, 3, day=date(2020, 1, 6))
    for slot, status in zip(history_slots, ("no_show", "completed", "no_show")):
        world.booking(slot, patient, status=status)
    target = Booking(slot_id=scene.slots[2], patient_id=patient, status="confirmed", created_at=datetime(2030, 1, 1, 10, 0))
    db.session.add(target)
    db.session.commit()

    p = risk.score_booking(target.id)

    served = features.deployable_features(age=39, sex="F", booked_on=date(2030, 1, 1), appointment_date=date(2030, 1, 7),
                                          history=[(date(2020, 1, 6), 1), (date(2020, 1, 6), 0), (date(2020, 1, 6), 1)])
    assert served["prev_appointments"] == 3 and served["lead_days"] == 6 and served["appt_weekday"] == 0
    expected = joblib.load(REAL / "risk_model.joblib").predict_proba(pd.DataFrame([served])[features.DEPLOYABLE_FEATURES])[:, 1][0]
    assert p == pytest.approx(expected, abs=1e-12)
    stored = db.session.get(RiskScore, target.id)
    assert stored.no_show_probability == pytest.approx(expected, abs=1e-12) and stored.model_version == CARD["version"]
    assert stored.scored_at is not None


def test_rescoring_replaces_the_row_there_is_one_score_per_booking(scene, world):
    patient = world.patient()
    world.profile(patient)
    booking_id = world.booking(scene.slots[1], patient)
    risk.score_booking(booking_id)
    risk.score_booking(booking_id)
    assert db.session.query(RiskScore).filter_by(booking_id=booking_id).count() == 1


def test_without_a_profile_or_with_impossible_dates_there_is_simply_no_score(scene, world):
    no_profile = world.booking(scene.slots[1], world.patient())
    assert risk.score_booking(no_profile) is None and score_rows() == 0
    patient = world.patient()
    world.profile(patient)
    backwards = Booking(slot_id=scene.slots[2], patient_id=patient, status="confirmed", created_at=datetime(2031, 1, 1))  # booked after the visit
    db.session.add(backwards)
    db.session.commit()
    assert risk.score_booking(backwards.id) is None and score_rows() == 0


def test_a_booking_shown_to_staff_without_a_score_is_scored_then(scene, world, client, app):
    patient = world.patient("Pat Unscored")
    world.profile(patient)
    booking_id = world.booking(scene.slots[1], patient)
    assert score_rows() == 0
    world.login(client, scene.doctor_user)
    page = body(client.get(f"/doctor/?date={scene.day}"))
    assert "Pat Unscored" in page and len(badges(page)) >= 1
    assert db.session.query(RiskScore).filter_by(booking_id=booking_id).count() == 1


def test_a_score_from_an_older_model_version_is_replaced(scene, world, client):
    patient = world.patient()
    world.profile(patient)
    booking_id = world.booking(scene.slots[1], patient)
    set_score(booking_id, 0.5, version="0.0.1-old")
    world.login(client, scene.doctor_user)
    client.get(f"/doctor/?date={scene.day}")
    db.session.expire_all()
    assert db.session.get(RiskScore, booking_id).model_version == CARD["version"]


def test_a_failing_model_on_a_staff_page_shows_no_badge_and_the_page_still_loads(scene, world, client, monkeypatch):
    patient = world.patient()
    world.profile(patient)
    world.booking(scene.slots[1], patient)
    real = risk.get_model()
    monkeypatch.setattr(risk, "get_model", lambda *a, **k: type(real)(**{**real.__dict__, "estimator": ExplodingModel()}))
    world.login(client, scene.doctor_user)
    response = client.get(f"/doctor/?date={scene.day}")
    assert response.status_code == 200 and badges(body(response)) == []


def test_the_model_is_loaded_once(app, monkeypatch):
    calls = []
    real_load = joblib.load
    monkeypatch.setattr(joblib, "load", lambda *a, **k: calls.append(a) or real_load(*a, **k))
    risk.reset_cache()
    first = risk.get_model()
    assert risk.get_model() is first and risk.get_model() is first
    assert len(calls) == 1


# ---- a card or model that does not fit disables the flag instead of misleading staff ---------------------------------
def broken_copy(tmp_path, edit):
    target = tmp_path / "artifacts"
    shutil.copytree(REAL, target)
    card = json.loads((target / "model_card.json").read_text())
    edit(card, target)
    (target / "model_card.json").write_text(json.dumps(card))
    return target


@pytest.mark.parametrize("name, edit", [
    ("no thresholds", lambda card, d: card.update(thresholds=None)),
    ("a different feature list", lambda card, d: card["features"].pop()),
    ("a hash that does not match the file", lambda card, d: card.update(artifact_sha256="0" * 64)),
    ("a model file that is not a model", lambda card, d: (d / "risk_model.joblib").write_bytes(b"garbage")),
])
def test_a_card_or_model_that_does_not_fit_disables_the_flag(app, tmp_path, name, edit):
    app.config["RISK_ARTIFACTS_DIR"] = broken_copy(tmp_path, edit)
    risk.reset_cache()
    assert risk.get_model() is None, name


# ---- end to end through the real forms ------------------------------------------------------------------------------
def test_registering_and_booking_through_the_forms_creates_a_score_staff_can_see(scene, world, client, app):
    page = body(client.get("/register"))
    client.post("/register", data={"name": "Flag Tester", "email": "flag@example.test", "password": "correct horse battery",
                                   "date_of_birth": "1988-03-09", "sex": "M", "_csrf": form_token(page)})
    page = body(client.get("/login"))
    client.post("/login", data={"email": "flag@example.test", "password": "correct horse battery", "_csrf": form_token(page)})
    slots = body(client.get(f"/patient/doctors/{scene.doctor}?date={scene.day}"))
    chosen = slot_ids(slots)[0]
    assert client.post("/patient/book", data={"slot_id": chosen, "_csrf": form_token(slots)}).status_code == 302
    booked = db.session.query(Booking).filter_by(slot_id=chosen).one()
    score = db.session.get(RiskScore, booked.id)
    assert score is not None and 0 < score.no_show_probability < 1

    staff = app.test_client()
    world.login(staff, scene.doctor_user)
    assert "Flag Tester" in body(staff.get(f"/doctor/?date={scene.day}"))


# ---- the seed gives a demo with a mix of flags ----------------------------------------------------------------------
def test_the_seeded_demo_shows_a_mix_of_low_medium_and_high(app, monkeypatch):
    from app import clock
    from db.seed_synthetic import seed

    monkeypatch.setattr(clock, "now", lambda: datetime(2030, 1, 7, 8, 0))
    summary = seed("a-demo-password")
    flags = summary["flags"]
    assert flags is not None and flags["scored"] > 100
    assert min(flags["low"], flags["medium"], flags["high"]) >= 0.03 * flags["scored"], flags  # every band is really present
    non_cancelled = db.session.execute(text("SELECT COUNT(*) FROM bookings WHERE status <> 'cancelled'")).scalar_one()
    assert score_rows() == flags["scored"] == non_cancelled
    # seeding again with --reset wipes the scores first (foreign key) and still works
    assert seed("another-password", reset=True)["flags"]["scored"] > 100
