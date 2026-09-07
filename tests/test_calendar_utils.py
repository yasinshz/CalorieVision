from datetime import date

from calendar_utils import saturday_week_bounds


def test_saturday_is_first_day_of_week():
    start, end = saturday_week_bounds(date(2026, 8, 8))  # Saturday
    assert start == date(2026, 8, 8)
    assert end == date(2026, 8, 14)


def test_friday_is_last_day_of_same_week():
    start, end = saturday_week_bounds(date(2026, 8, 14))  # Friday
    assert start == date(2026, 8, 8)
    assert end == date(2026, 8, 14)


def test_monday_maps_back_to_previous_saturday():
    start, end = saturday_week_bounds(date(2026, 8, 10))  # Monday
    assert start == date(2026, 8, 8)
    assert end == date(2026, 8, 14)
