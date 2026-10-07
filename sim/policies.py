"""Booking policies: which slots of a session get a second patient (PLAN section 5).

  P0  no overbooking
  P1  uniform: every k-th slot gets a second patient (k = 2, 3, 4, 6)
  P2  risk threshold: a slot gets a second patient when the FIRST patient's predicted risk is at least a threshold.
      The grid is the fixed values 0.3 and 0.5 plus the 50th, 70th, 80th, 90th and 95th percentiles of predicted risk,
      so the rule still triggers when base rates are near 20%.
  Ranked (S6 only)  double-book exactly m slots chosen by a ranking: predicted risk, random scores, even spacing, or
      an oracle that knows who will not attend. It is how prediction is compared with its alternatives at the SAME
      number of double-booked slots.

Common random numbers: a replication's draws (patients, attendance uniforms, service times) are made once, and every
policy only chooses which slots use an extra patient. Extra patients are taken IN ORDER from the extra draws, so any
two policies that double-book the same number of slots get exactly the same extra patients; they differ only in
WHERE those patients are placed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

PERCENTILES = (50, 70, 80, 90, 95)
P1_KS = (2, 3, 4, 6)
FIXED_THRESHOLDS = (0.3, 0.5)


@dataclass
class Draw:
    """Everything random about one replication, drawn once and shared by every policy.

    risk / extra_risk   true no-show probability of the primary patient of each slot and of the j-th extra patient
    u_primary / u_extra a patient does NOT attend when its uniform is below its risk
    service_*           consultation time in minutes if the patient attends
    random_scores       uniform scores for the "random" ranking
    """

    risk: np.ndarray
    extra_risk: np.ndarray
    u_primary: np.ndarray
    u_extra: np.ndarray
    service_primary: np.ndarray
    service_extra: np.ndarray
    random_scores: np.ndarray

    @property
    def primary_attends(self) -> np.ndarray:
        return self.u_primary >= self.risk

    @property
    def extra_attends(self) -> np.ndarray:
        return self.u_extra >= self.extra_risk


@dataclass
class Session:
    arrivals: list
    attends: list
    services: list
    double_slots: int  # slots that received a second patient
    both_slots: int  # of those, slots where BOTH patients attended


def build_session(draw: Draw, mask: np.ndarray, slot_minutes: float) -> Session:
    """Queue order: slot by slot, the primary patient first, then the extra patient if the slot is double-booked.
    The j-th double-booked slot (in slot order) receives the j-th extra patient."""
    arrivals, attends, services = [], [], []
    primary_attends, extra_attends = draw.primary_attends, draw.extra_attends
    extra, both = 0, 0
    for slot in range(len(draw.risk)):
        arrival = slot * slot_minutes
        arrivals.append(arrival)
        attends.append(bool(primary_attends[slot]))
        services.append(float(draw.service_primary[slot]))
        if mask[slot]:
            arrivals.append(arrival)
            attends.append(bool(extra_attends[extra]))
            services.append(float(draw.service_extra[extra]))
            both += int(primary_attends[slot] and extra_attends[extra])
            extra += 1
    return Session(arrivals, attends, services, double_slots=extra, both_slots=both)


class NoOverbooking:
    name, family, label, threshold = "P0", "P0", "none", None

    def select(self, draw, visible_risk, matched_count=None) -> np.ndarray:
        return np.zeros(len(visible_risk), dtype=bool)


class Uniform:
    family = "P1"

    def __init__(self, k: int) -> None:
        self.k, self.label, self.threshold = k, f"k={k}", None
        self.name = f"P1 k={k}"

    def select(self, draw, visible_risk, matched_count=None) -> np.ndarray:
        return (np.arange(len(visible_risk)) + 1) % self.k == 0  # the k-th, 2k-th, ... slot


class RiskThreshold:
    family = "P2"

    def __init__(self, threshold: float, label: str) -> None:
        self.threshold, self.label = float(threshold), label
        self.name = f"P2 {label}"

    def select(self, draw, visible_risk, matched_count=None) -> np.ndarray:
        return np.asarray(visible_risk) >= self.threshold


class Ranked:
    """Double-book exactly m slots, chosen by `selector`. m is fixed or supplied per session (matched_count)."""

    family = "ranked"
    SELECTORS = ("risk", "random", "uniform", "oracle")

    def __init__(self, selector: str, m: Optional[int] = None) -> None:
        if selector not in self.SELECTORS:
            raise ValueError(f"unknown selector {selector!r}")
        self.selector, self.m, self.threshold = selector, m, None
        self.label = selector
        self.name = f"ranked {selector}"

    def select(self, draw, visible_risk, matched_count=None) -> np.ndarray:
        m = self.m if self.m is not None else matched_count
        if m is None:
            raise ValueError("a Ranked policy needs a count: pass m or matched_count")
        n = len(visible_risk)
        mask = np.zeros(n, dtype=bool)
        if m <= 0:
            return mask
        if self.selector == "uniform":
            mask[np.floor((np.arange(m) + 0.5) * n / m).astype(int)] = True  # evenly spread through the session
            return mask
        if self.selector == "risk":
            scores = np.asarray(visible_risk, dtype=float)
        elif self.selector == "random":
            scores = draw.random_scores
        else:  # oracle: slots whose first patient will not attend come first, ties broken by predicted risk
            scores = (~draw.primary_attends).astype(float) * 10.0 + np.asarray(visible_risk, dtype=float)
        mask[np.argsort(-scores, kind="stable")[:m]] = True
        return mask


def p1_policies() -> list:
    return [Uniform(k) for k in P1_KS]


def percentile_thresholds(pool, percentiles=PERCENTILES) -> dict:
    """{ 'p50': value, ... }: percentiles of the predicted-risk distribution, so about (100 - q)% of first patients
    reach the p-q threshold."""
    return {f"p{q}": float(np.percentile(pool, q)) for q in percentiles}


def p2_policies(pool) -> list:
    grid = [RiskThreshold(t, f"{t:g}") for t in FIXED_THRESHOLDS]
    grid += [RiskThreshold(value, label) for label, value in percentile_thresholds(pool).items()]
    return grid
