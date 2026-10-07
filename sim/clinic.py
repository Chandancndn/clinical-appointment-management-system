"""One provider, one half-day session: a SimPy discrete-event model (PLAN section 5).

Patients arrive exactly at their slot time and are seen first come, first served; two patients booked into the same
slot arrive together and the second waits behind the first. A patient who does not attend uses no doctor time.
Service starts when the patient has arrived and the doctor is free. Every attending patient is eventually seen, so
a late session simply runs over (overtime).

Metrics for one session:
  served     patients who attended and were seen
  mean_wait  mean of (service start - arrival) over attending patients; NaN if nobody attended
  overtime   minutes the doctor works past the session length T
  idle       minutes inside [0, T] with the doctor not consulting (T if nobody attends)

`simulate_session_recurrence` computes the same thing from start_i = max(arrival_i, end_{i-1}); it exists only so a test
can check the SimPy model against an independent formula.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import simpy


@dataclass(frozen=True)
class SessionResult:
    served: int
    mean_wait: float
    overtime: float
    idle: float
    waits: tuple
    starts: tuple
    ends: tuple


def _check(arrivals, attends, services) -> None:
    if not (len(arrivals) == len(attends) == len(services)):
        raise ValueError("arrivals, attends and services must have the same length")
    if any(b < a for a, b in zip(arrivals, list(arrivals)[1:])):
        raise ValueError("patients must be listed in arrival (queue) order")


def _summarise(waits: list, starts: list, ends: list, session_minutes: float) -> SessionResult:
    busy_inside = sum(max(0.0, min(end, session_minutes) - min(start, session_minutes)) for start, end in zip(starts, ends))
    overtime = max(0.0, max(ends) - session_minutes) if ends else 0.0
    return SessionResult(
        served=len(waits), mean_wait=float(np.mean(waits)) if waits else math.nan, overtime=overtime,
        idle=session_minutes - busy_inside, waits=tuple(waits), starts=tuple(starts), ends=tuple(ends))


def simulate_session(arrivals: Sequence[float], attends: Sequence[bool], services: Sequence[float],
                     session_minutes: float) -> SessionResult:
    """Run one session in SimPy. `arrivals` must be in queue order (non-decreasing); ties keep list order."""
    _check(arrivals, attends, services)
    env = simpy.Environment()
    doctor = simpy.Resource(env, capacity=1)
    waits, starts, ends = [], [], []

    def patient(arrival: float, service: float):
        yield env.timeout(arrival)  # arrives exactly on time
        with doctor.request() as turn:
            yield turn  # first come, first served
            started = env.now
            waits.append(started - arrival)
            starts.append(started)
            yield env.timeout(service)
            ends.append(env.now)

    for arrival, attended, service in zip(arrivals, attends, services):
        if attended:  # a no-show never joins the queue and uses no doctor time
            env.process(patient(float(arrival), float(service)))
    env.run()
    return _summarise(waits, starts, ends, session_minutes)


def simulate_session_recurrence(arrivals, attends, services, session_minutes: float) -> SessionResult:
    """The same session from the single-server recurrence start_i = max(arrival_i, end_{i-1}). For testing only."""
    _check(arrivals, attends, services)
    waits, starts, ends = [], [], []
    free_at = 0.0
    for arrival, attended, service in zip(arrivals, attends, services):
        if not attended:
            continue
        start = max(float(arrival), free_at)
        free_at = start + float(service)
        waits.append(start - float(arrival))
        starts.append(start)
        ends.append(free_at)
    return _summarise(waits, starts, ends, session_minutes)
