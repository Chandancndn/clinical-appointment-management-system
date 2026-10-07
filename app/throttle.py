"""Brute-force protection for logins and password checks.

Failures are remembered in memory, per process, for a sliding window. Once a key has `limit` failures inside the window it
is blocked until the oldest of those failures ages out, and a blocked try is refused without checking the password (and
without extending the block). A success clears the key. An email that does not exist is throttled exactly like one that
does, so the block reveals nothing about which accounts exist. Behind several worker processes each keeps its own count,
which still bounds an attacker per worker; a shared store would be the next step for a real deployment.
"""
from __future__ import annotations

import math
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

MAX_KEYS = 10_000  # bound on memory: when exceeded, keys with no recent failures are dropped


class Throttle:
    def __init__(self) -> None:
        self._failures: dict = {}
        self._lock = threading.Lock()

    def blocked_for(self, key, limit: int, window: timedelta, now: datetime) -> Optional[timedelta]:
        """How long `key` must still wait, or None if it may try."""
        with self._lock:
            failures = self._recent(key, window, now)
            if len(failures) >= limit:
                return failures[-limit] + window - now
            return None

    def record_failure(self, key, window: timedelta, now: datetime) -> None:
        with self._lock:
            self._recent(key, window, now).append(now)
            if len(self._failures) > MAX_KEYS:
                self._prune(window, now)

    def clear(self, key) -> None:
        with self._lock:
            self._failures.pop(key, None)

    def _recent(self, key, window: timedelta, now: datetime) -> deque:
        failures = self._failures.setdefault(key, deque())
        while failures and failures[0] <= now - window:
            failures.popleft()
        return failures

    def _prune(self, window: timedelta, now: datetime) -> None:
        for key in list(self._failures):
            self._recent(key, window, now)
            if not self._failures[key]:
                del self._failures[key]


def minutes_text(wait: timedelta) -> str:
    minutes = max(1, math.ceil(wait.total_seconds() / 60))
    return f"{minutes} minute{'s' if minutes != 1 else ''}"
