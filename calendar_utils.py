"""Calendar helpers for the Persian Saturday-to-Friday reporting week."""

from __future__ import annotations

from datetime import date, timedelta


def saturday_week_bounds(anchor: date) -> tuple[date, date]:
    """Return Saturday and Friday containing *anchor*.

    Python's ``date.weekday()`` uses Monday=0 ... Sunday=6.  In the
    application's reporting calendar, Saturday is the first day of the week.
    """
    days_since_saturday = (anchor.weekday() - 5) % 7
    week_start = anchor - timedelta(days=days_since_saturday)
    return week_start, week_start + timedelta(days=6)
