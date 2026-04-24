"""Rest-days and site-splits tests."""
from __future__ import annotations

from datetime import date

import pandas as pd

from madness.features.rest import days_rest_before
from madness.features.site_splits import build_site_splits


def _games():
    return pd.DataFrame([
        {"season": 2024, "date": date(2024, 3, 9), "team_a": "A", "team_b": "B",
         "score_a": 80, "score_b": 70, "site": "home"},
        {"season": 2024, "date": date(2024, 3, 10), "team_a": "A", "team_b": "C",
         "score_a": 65, "score_b": 70, "site": "away"},
        {"season": 2024, "date": date(2024, 3, 11), "team_a": "A", "team_b": "D",
         "score_a": 85, "score_b": 80, "site": "neutral"},
        {"season": 2024, "date": date(2024, 2, 1), "team_a": "B", "team_b": "E",
         "score_a": 50, "score_b": 60, "site": "home"},
    ])


def test_days_rest_counts_from_last_game():
    out = days_rest_before(_games(), game_date=date(2024, 3, 14))
    a = out[out["team"] == "A"].iloc[0]
    assert a["days_rest"] == 3  # last played Mar 11
    b = out[out["team"] == "B"].iloc[0]
    assert b["days_rest"] == 5  # last played Mar 9 (as team_b vs A in the first row)


def test_games_in_last_7():
    out = days_rest_before(_games(), game_date=date(2024, 3, 14))
    a = out[out["team"] == "A"].iloc[0]
    assert a["games_in_last_7"] == 3


def test_site_splits_by_site():
    out = build_site_splits(_games(), cutoff=date(2024, 3, 14))
    a = out[out["team"] == "A"].iloc[0]
    # A: home W vs B (margin 10), away L vs C (margin -5), neutral W vs D (margin 5)
    assert a["home_winpct"] == 1.0
    assert a["away_winpct"] == 0.0
    assert a["neutral_winpct"] == 1.0
    assert a["home_margin"] == 10.0
    assert a["away_margin"] == -5.0
    assert a["neutral_margin"] == 5.0
