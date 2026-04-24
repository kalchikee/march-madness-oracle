"""Late-season form features — the "hot team" signal.

Hypothesis (one of the novel-signal priorities): Feb/March SoS-weighted
performance is more predictive than November/December, because rotations
stabilize and motivation peaks. This module exposes that as a feature.
"""
from __future__ import annotations

from datetime import date

import pandas as pd


def last_n_form(
    regular_season_games: pd.DataFrame,
    cutoff: date,
    n: int = 10,
) -> pd.DataFrame:
    """Rolling last-N-games record + avg margin + opp quality."""
    if regular_season_games.empty:
        return pd.DataFrame(columns=["season", "team", "last10_wins", "last10_point_diff"])

    df = regular_season_games.copy()
    df = df[df["date"] < cutoff]
    if df.empty:
        return pd.DataFrame(columns=["season", "team", "last10_wins", "last10_point_diff"])

    left = df[["season", "date", "team_a", "team_b", "score_a", "score_b"]].rename(
        columns={"team_a": "team", "team_b": "opp", "score_a": "pf", "score_b": "pa"}
    )
    right = df[["season", "date", "team_b", "team_a", "score_b", "score_a"]].rename(
        columns={"team_b": "team", "team_a": "opp", "score_b": "pf", "score_a": "pa"}
    )
    stacked = pd.concat([left, right], ignore_index=True)
    stacked = stacked.sort_values(["season", "team", "date"])

    stacked["win"] = stacked["pf"] > stacked["pa"]
    stacked["margin"] = stacked["pf"] - stacked["pa"]

    def _tail_n(group: pd.DataFrame) -> pd.Series:
        tail = group.tail(n)
        return pd.Series({
            "last10_wins": int(tail["win"].sum()),
            "last10_point_diff": float(tail["margin"].mean()),
        })

    out = stacked.groupby(["season", "team"]).apply(_tail_n).reset_index()
    return out


def longest_win_streak(regular_season_games: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    if regular_season_games.empty:
        return pd.DataFrame(columns=["season", "team", "longest_win_streak"])

    df = regular_season_games.copy()
    df = df[df["date"] < cutoff]
    left = df[["season", "date", "team_a", "team_b", "score_a", "score_b"]].rename(
        columns={"team_a": "team", "team_b": "opp", "score_a": "pf", "score_b": "pa"}
    )
    right = df[["season", "date", "team_b", "team_a", "score_b", "score_a"]].rename(
        columns={"team_b": "team", "team_a": "opp", "score_b": "pf", "score_a": "pa"}
    )
    stacked = pd.concat([left, right], ignore_index=True).sort_values(["season", "team", "date"])
    stacked["win"] = stacked["pf"] > stacked["pa"]

    def _longest(s: pd.Series) -> int:
        best = cur = 0
        for v in s:
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    out = stacked.groupby(["season", "team"])["win"].apply(_longest).reset_index(
        name="longest_win_streak"
    )
    return out
