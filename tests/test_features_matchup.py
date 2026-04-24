"""Matchup feature tests — verify diffs/ratios are emitted correctly."""
from __future__ import annotations

import pandas as pd

from madness.features.matchup import build_matchup_features


def test_matchup_emits_diff_and_ratio():
    games = pd.DataFrame([
        {"season": 2019, "team_a": "A", "team_b": "B", "target": 1},
    ])
    ts = pd.DataFrame([
        {"season": 2019, "team": "A", "win_pct": 0.8, "wins": 20, "losses": 5,
         "point_diff_per_game": 12.0, "points_for_pg": 75, "points_against_pg": 63},
        {"season": 2019, "team": "B", "win_pct": 0.6, "wins": 18, "losses": 12,
         "point_diff_per_game": 4.0, "points_for_pg": 70, "points_against_pg": 66},
    ])
    out = build_matchup_features(games, ts)
    assert out.loc[0, "diff_win_pct"] == 0.8 - 0.6
    assert abs(out.loc[0, "ratio_win_pct"] - (0.8 / 0.6)) < 1e-9


def test_matchup_symmetry_zero_diff():
    games = pd.DataFrame([
        {"season": 2019, "team_a": "A", "team_b": "B", "target": 1},
    ])
    ts = pd.DataFrame([
        {"season": 2019, "team": "A", "win_pct": 0.7, "wins": 20, "losses": 8,
         "point_diff_per_game": 5.0, "points_for_pg": 70, "points_against_pg": 65},
        {"season": 2019, "team": "B", "win_pct": 0.7, "wins": 20, "losses": 8,
         "point_diff_per_game": 5.0, "points_for_pg": 70, "points_against_pg": 65},
    ])
    out = build_matchup_features(games, ts)
    assert out.loc[0, "diff_win_pct"] == 0
    assert out.loc[0, "ratio_win_pct"] == 1.0
