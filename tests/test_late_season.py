"""Late-season decay feature tests."""
from __future__ import annotations

from datetime import date

import pandas as pd

from madness.features.late_season import time_decay_weighted_features


def _make_games():
    # Three November games (far) and three March games (close to cutoff)
    return pd.DataFrame([
        {"season": 2024, "date": date(2023, 11, 5), "team_a": "A", "team_b": "X",
         "score_a": 60, "score_b": 90, "site": "away"},   # A loses by 30 early
        {"season": 2024, "date": date(2023, 11, 15), "team_a": "A", "team_b": "Y",
         "score_a": 70, "score_b": 80, "site": "away"},   # A loses by 10 early
        {"season": 2024, "date": date(2023, 11, 25), "team_a": "A", "team_b": "Z",
         "score_a": 75, "score_b": 85, "site": "home"},   # A loses by 10 early
        {"season": 2024, "date": date(2024, 2, 20), "team_a": "A", "team_b": "P",
         "score_a": 85, "score_b": 70, "site": "home"},   # A wins by 15 late
        {"season": 2024, "date": date(2024, 3, 1), "team_a": "A", "team_b": "Q",
         "score_a": 90, "score_b": 70, "site": "neutral"},# A wins by 20 late
        {"season": 2024, "date": date(2024, 3, 10), "team_a": "A", "team_b": "R",
         "score_a": 100, "score_b": 80, "site": "home"},  # A wins by 20 late
    ])


def test_decay_weighted_margin_favors_late_games():
    cutoff = date(2024, 3, 14)
    out = time_decay_weighted_features(_make_games(), cutoff, tau_days=20)
    row = out[out["team"] == "A"].iloc[0]
    # Unweighted avg margin ≈ 0 (3 losses of ~17 avg, 3 wins of ~18 avg)
    # Decay-weighted should heavily favor late wins → positive
    assert row["late_weighted_margin"] > 10, (
        f"Expected decay-weighted margin to be positive & large, got {row['late_weighted_margin']}"
    )


def test_feb_mar_winpct_only_counts_recent():
    cutoff = date(2024, 3, 14)
    out = time_decay_weighted_features(_make_games(), cutoff, tau_days=30)
    row = out[out["team"] == "A"].iloc[0]
    assert row["feb_mar_games"] == 3
    assert row["feb_mar_winpct"] == 1.0  # all 3 late games were wins


def test_decay_excludes_post_cutoff():
    games = _make_games()
    games.loc[len(games)] = {
        "season": 2024, "date": date(2024, 3, 20), "team_a": "A", "team_b": "LEAK",
        "score_a": 50, "score_b": 100, "site": "away",
    }
    cutoff = date(2024, 3, 14)
    out = time_decay_weighted_features(games, cutoff, tau_days=30)
    row = out[out["team"] == "A"].iloc[0]
    assert row["feb_mar_games"] == 3
