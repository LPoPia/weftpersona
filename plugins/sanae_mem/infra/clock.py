from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo

    SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
except Exception:  # Windows may not have tzdata installed.
    SHANGHAI_TZ = timezone(timedelta(hours=8))

SLEEP_DAY_BOUNDARY_HOURS = 4


def shanghai_datetime(value: datetime | float | None = None) -> datetime:
    """Return a Shanghai-aware datetime; naive datetimes are interpreted there."""
    if value is None:
        return datetime.now(SHANGHAI_TZ)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, SHANGHAI_TZ)
    if value.tzinfo is None:
        return value.replace(tzinfo=SHANGHAI_TZ)
    return value.astimezone(SHANGHAI_TZ)


def sleep_day_date(value: datetime | float | None = None) -> date:
    """Return the 04:00 Shanghai sleep day as a date."""
    return (shanghai_datetime(value) - timedelta(hours=SLEEP_DAY_BOUNDARY_HOURS)).date()


def sleep_day_iso(value: datetime | float | None = None) -> str:
    """Return the 04:00 Shanghai sleep day as an ISO date string."""
    return sleep_day_date(value).isoformat()


def sleep_day_window(value: datetime | float | None = None) -> tuple[str, datetime, datetime]:
    """Return the sleep-day label and its Shanghai 04:00 inclusive/exclusive bounds."""
    day = sleep_day_iso(value)
    start = datetime.fromisoformat(f"{day}T04:00:00").replace(tzinfo=SHANGHAI_TZ)
    return day, start, start + timedelta(days=1)
