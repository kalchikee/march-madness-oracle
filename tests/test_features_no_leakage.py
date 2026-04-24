"""Feature-leakage test: team-season features for 2019 use no post-Feb-2019 data."""
from __future__ import annotations

from datetime import date

import pandas as pd

from madness.features.team_season import build_team_season_table


def _games():
    return pd.DataFrame([
        {"season": 2019, "date": date(2019, 1, 1), "team_a": "A", "team_b": "B",
         "score_a": 80, "score_b": 70, "site": "home"},
        {"season": 2019, "date": date(2019, 2, 1), "team_a": "A", "team_b": "C",
         "score_a": 60, "score_b": 75, "site": "away"},
        {"season": 2019, "date": date(2019, 3, 20), "team_a": "A", "team_b": "D",
         "score_a": 100, "score_b": 50, "site": "neutral"},
    ])


def test_cutoff_excludes_post_cutoff_games():
    cutoff = date(2019, 3, 15)
    out = build_team_season_table(_games(), cutoff)
    a = out[out["team"] == "A"].iloc[0]
    # Team A has 1 W (vs B) and 1 L (vs C); the March 20 game must be excluded.
    assert a["wins"] == 1
    assert a["losses"] == 1
    assert a["games_played"] == 2


def test_cutoff_includes_all_when_past_season():
    cutoff = date(2019, 5, 1)
    out = build_team_season_table(_games(), cutoff)
    a = out[out["team"] == "A"].iloc[0]
    assert a["games_played"] == 3
