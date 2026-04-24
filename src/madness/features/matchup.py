"""Pairwise matchup features: diffs, ratios, and style-clash signals."""
from __future__ import annotations

import pandas as pd


DIFF_COLUMNS = [
    "win_pct", "point_diff_per_game", "points_for_pg", "points_against_pg",
    "wins", "losses",
]


def build_matchup_features(
    games: pd.DataFrame,
    team_season: pd.DataFrame,
) -> pd.DataFrame:
    """Join team-season features onto game rows and emit pairwise diffs.

    Input `games` schema: season, team_a, team_b, (optional target).
    """
    if games.empty or team_season.empty:
        return pd.DataFrame()

    t_a = team_season.add_prefix("a_").rename(
        columns={"a_season": "season", "a_team": "team_a"}
    )
    t_b = team_season.add_prefix("b_").rename(
        columns={"b_season": "season", "b_team": "team_b"}
    )
    merged = games.merge(t_a, on=["season", "team_a"], how="left")
    merged = merged.merge(t_b, on=["season", "team_b"], how="left")

    for col in DIFF_COLUMNS:
        a, b = f"a_{col}", f"b_{col}"
        if a in merged.columns and b in merged.columns:
            merged[f"diff_{col}"] = merged[a] - merged[b]
            denom = merged[b].replace(0, pd.NA)
            merged[f"ratio_{col}"] = merged[a] / denom

    return merged
