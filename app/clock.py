"""The one place the app asks what time it is, so tests can replace it."""
from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def clinic_timezone() -> ZoneInfo | None:
    """The clinic's time zone from CLINIC_TIMEZONE (e.g. Asia/Kolkata), or None to use the machine's local time.

    A hosted server runs on UTC, so a deployment sets this; a laptop leaves it unset.
    """
    name = os.environ.get("CLINIC_TIMEZONE", "").strip()
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise RuntimeError(f"CLINIC_TIMEZONE={name!r} is not a known time zone (for example Asia/Kolkata).") from error


def now() -> datetime:
    """Current clinic-local time, naive. Slot dates and times are stored in the same local time."""
    zone = clinic_timezone()
    return datetime.now(zone).replace(tzinfo=None) if zone else datetime.now()
