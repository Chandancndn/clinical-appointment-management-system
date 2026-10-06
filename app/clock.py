"""The one place the app asks what time it is, so tests can replace it."""
from __future__ import annotations

from datetime import datetime


def now() -> datetime:
    """Current clinic-local time, naive. Slot dates and times are stored in the same local time."""
    return datetime.now()
