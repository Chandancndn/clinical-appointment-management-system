"""The advisory no-show risk flag (PLAN section 6, CLAUDE.md hard rule 2).

  * The saved deployable model (ml/artifacts/risk_model.joblib) is loaded ONCE, and only if it fits its model card:
    the file hash matches, the feature list is the one ml/src/features.py defines, and the card holds the Medium
    and High thresholds that E10 chose. Otherwise the flag is simply off.
  * A booking's inputs are computed by the SAME function training used, features.deployable_features(), from plain
    values read out of the database (date of birth, sex, booking date, appointment date, earlier outcomes).
  * A booking is scored AFTER it has been committed, in a try/except, so scoring can never roll it back. Every public
    function here returns None or an empty result on ANY failure: a missing model, a bad card, an exception in the
    model, a missing profile, a database hiccup. The booking logic (app/services.py) never imports this module and
    never reads a score, so booking works exactly the same with or without it.
  * Scores are shown to doctors and admins only, as Low, Medium or High, using the thresholds on the model card.
    Patients never see them, and no patient is refused or moved because of one.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import sqlalchemy as sa
from flask import current_app

from . import clock
from .extensions import db
from .models import Booking, PatientProfile, RiskScore, Slot

log = logging.getLogger("cams.risk")

HISTORY_STATUSES = ("completed", "no_show")  # appointments whose outcome is known
MAX_SCORED_PER_PAGE = 200  # unscored bookings scored lazily when a staff page shows them


@dataclass(frozen=True)
class RiskModel:
    estimator: object
    version: str
    medium: float  # from the model card (E10), never typed in here
    high: float


@dataclass(frozen=True)
class Flag:
    probability: float
    band: str  # "low", "medium" or "high"
    version: str


_cache: dict = {}


def reset_cache() -> None:
    """Forget the loaded model (tests; or after re-running E9 and E10 without restarting the app)."""
    _cache.clear()


# ---- loading the model once ------------------------------------------------------------------------
def _load(directory: Path) -> Optional[RiskModel]:
    try:
        import joblib

        from ml.src import features  # the same module the model was trained with

        card = json.loads((directory / "model_card.json").read_text())
        model_path = directory / "risk_model.joblib"
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != card["artifact_sha256"]:
            raise ValueError("risk_model.joblib does not match the hash in model_card.json")
        if [f["name"] for f in card["features"]] != features.DEPLOYABLE_FEATURES:
            raise ValueError("the model card's features are not the ones features.deployable_features() produces")
        thresholds = card["thresholds"]
        medium, high = float(thresholds["medium"]["threshold"]), float(thresholds["high"]["threshold"])
        if not 0 < medium < high < 1:
            raise ValueError(f"unusable thresholds on the model card: medium {medium}, high {high}")
        return RiskModel(joblib.load(model_path), str(card["version"]), medium, high)
    except Exception as exc:  # any problem at all means: no flag
        log.warning("risk flag is off: %s", exc)
        return None


def get_model() -> Optional[RiskModel]:
    """The saved model, loaded once per artifacts directory; None (also remembered) if it cannot be used."""
    directory = Path(current_app.config["RISK_ARTIFACTS_DIR"])
    key = str(directory)
    if key not in _cache:
        _cache[key] = _load(directory)
    return _cache[key]


def band_for(probability: float, model: RiskModel) -> str:
    """Low below the card's Medium threshold, Medium from it, High from the card's High threshold."""
    if probability >= model.high:
        return "high"
    if probability >= model.medium:
        return "medium"
    return "low"


# ---- scoring one booking -------------------------------------------------------------------------------
def _history(patient_id: int, booking_id: int) -> list:
    """(appointment date, 1 if no-show else 0) for the patient's OTHER appointments whose outcome is known.
    deployable_features() keeps only those dated before this booking was made."""
    rows = db.session.execute(
        sa.select(Slot.slot_date, Booking.status).join(Slot, Slot.id == Booking.slot_id)
        .where(Booking.patient_id == patient_id, Booking.id != booking_id, Booking.status.in_(HISTORY_STATUSES))
    ).all()
    return [(day, 1 if status == "no_show" else 0) for day, status in rows]


def probability_for(booking_id: int) -> Optional[float]:
    """The model's no-show probability for a booking, or None for ANY reason it cannot be computed. Writes nothing."""
    try:
        model = get_model()
        if model is None:
            return None
        import pandas as pd

        from ml.src import features

        row = db.session.execute(
            sa.select(Booking.patient_id, Booking.created_at, Slot.slot_date, PatientProfile.date_of_birth, PatientProfile.sex)
            .join(Slot, Slot.id == Booking.slot_id).join(PatientProfile, PatientProfile.user_id == Booking.patient_id)
            .where(Booking.id == booking_id)
        ).one_or_none()
        if row is None:  # no such booking, or the patient has no profile (no date of birth, no sex)
            return None
        values = features.deployable_features(
            age=features.age_in_years(row.date_of_birth, row.slot_date), sex=row.sex, booked_on=row.created_at,
            appointment_date=row.slot_date, history=_history(row.patient_id, booking_id))
        frame = pd.DataFrame([values])[features.DEPLOYABLE_FEATURES]
        probability = float(model.estimator.predict_proba(frame)[:, 1][0])
        if not 0.0 < probability < 1.0:
            raise ValueError(f"the model returned {probability!r}")
        return probability
    except Exception as exc:
        db.session.rollback()  # we only read, so this just clears any failed state
        log.warning("could not score booking %s: %s", booking_id, exc)
        return None


def _write_score(booking_id: int, probability: float, version: str) -> None:
    row = db.session.get(RiskScore, booking_id)
    if row is None:
        db.session.add(RiskScore(booking_id=booking_id, no_show_probability=probability, model_version=version,
                                 scored_at=clock.now()))
    else:
        row.no_show_probability, row.model_version, row.scored_at = probability, version, clock.now()
    db.session.commit()


def score_booking(booking_id: int) -> Optional[float]:
    """Score a booking and store the result (one row per booking, replaced on re-scoring). Never raises."""
    try:
        probability = probability_for(booking_id)
        if probability is None:
            return None
        _write_score(booking_id, probability, get_model().version)
        return probability
    except Exception as exc:
        db.session.rollback()  # a new transaction: the booking itself was committed long before this started
        log.warning("could not store the score of booking %s: %s", booking_id, exc)
        return None


def score_after_commit(booking) -> None:
    """Call right after a booking has been committed. Whatever happens here, the booking stands."""
    try:
        score_booking(booking.id)
    except Exception as exc:  # even reading the id back failed: still nothing to undo
        log.warning("could not score a new booking: %s", exc)


def score_many(booking_ids: Iterable[int]) -> Optional[dict]:
    """Score many bookings in one go (the demo seed). Returns {"scored", "low", "medium", "high"}, or None without a model."""
    try:
        model = get_model()
        if model is None:
            return None
        scores = {}
        for booking_id in booking_ids:
            probability = probability_for(booking_id)
            if probability is not None:
                scores[booking_id] = probability
        db.session.execute(sa.delete(RiskScore).where(RiskScore.booking_id.in_(list(scores) or [0])))
        db.session.add_all(RiskScore(booking_id=i, no_show_probability=p, model_version=model.version, scored_at=clock.now())
                           for i, p in scores.items())
        db.session.commit()
        counts = {"low": 0, "medium": 0, "high": 0}
        for probability in scores.values():
            counts[band_for(probability, model)] += 1
        return {"scored": len(scores), **counts}
    except Exception as exc:
        db.session.rollback()
        log.warning("could not score the bookings: %s", exc)
        return None


# ---- what staff pages show ---------------------------------------------------------------------------------------
def flags_for(booking_ids: Iterable) -> dict:
    """{booking id: Flag} for the bookings a STAFF page is about to show. A booking without a score from the current
    model is scored now. Returns {} (so no badges, and the page still works) on any failure."""
    try:
        ids = list(dict.fromkeys(i for i in booking_ids if i is not None))
        model = get_model()
        if model is None or not ids:
            return {}

        def read() -> dict:
            rows = db.session.execute(sa.select(RiskScore).where(RiskScore.booking_id.in_(ids))).scalars().all()
            return {r.booking_id: r for r in rows if r.model_version == model.version}

        scores = read()
        missing = [i for i in ids if i not in scores]
        for booking_id in missing[:MAX_SCORED_PER_PAGE]:
            score_booking(booking_id)
        if missing:
            scores = read()
        return {i: Flag(r.no_show_probability, band_for(r.no_show_probability, model), r.model_version)
                for i, r in scores.items()}
    except Exception as exc:
        db.session.rollback()
        log.warning("no risk flags on this page: %s", exc)
        return {}
