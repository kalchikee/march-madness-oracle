from datetime import date

from madness.config import (
    current_season,
    is_tournament_window,
    season_end_date,
    season_start_date,
    tournament_window,
)


def test_season_start_end():
    assert season_start_date(2026) == date(2025, 11, 1)
    assert season_end_date(2026) == date(2026, 4, 15)


def test_tournament_window():
    start, end = tournament_window(2026)
    assert start < end
    assert start.year == 2026
    assert end.year == 2026


def test_is_tournament_window_march():
    assert is_tournament_window(date(2026, 3, 20)) is True


def test_is_tournament_window_june():
    assert is_tournament_window(date(2026, 6, 15)) is False


def test_current_season_is_int():
    s = current_season()
    assert isinstance(s, int)
    assert 2000 < s < 2100
