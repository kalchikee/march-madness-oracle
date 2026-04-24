"""Rest-days and game-density features.

Novel-signal hypothesis: a team that just played in a conference
championship game on Sunday faces a compressed prep window for a
Thursday/Friday R64 tip. Short-rest teams also run lighter rotations,
which amplifies in later rounds. We compute:

    - days_rest_since_last: days between last played game and tournament game
    - games_in_last_7_days: conf-tourney fatigue signal
    - rest_diff: (a_days_rest - b_days_rest) — the matchup-level delta
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def days_rest_before(
    regular_season_games: pd.DataFrame,
    game_date: date,
) -> pd.DataFrame:
    """For each team, days since their last played game before `game_date`."""
    if regular_season_games.empty:
        return pd.DataFrame(columns=["season", "team", "days_rest", "games_in_last_7"])

    df = regular_season_games.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] < game_date]
    if df.empty:
        return pd.DataFrame(columns=["season", "team", "days_rest", "games_in_last_7"])

    left = df[["season", "date", "team_a"]].rename(columns={"team_a": "team"})
    right = df[["season", "date", "team_b"]].rename(columns={"team_b": "team"})
    stacked = pd.concat([left, right], ignore_index=True)

    last_game = stacked.groupby(["season", "team"])["date"].max().reset_index(
        name="last_played"
    )
    last_game["days_rest"] = last_game["last_played"].apply(
        lambda d: (game_date - d).days
    )

    cutoff_7 = game_date - timedelta(days=7)
    recent = stacked[stacked["date"] >= cutoff_7]
    counts = recent.groupby(["season", "team"]).size().reset_index(name="games_in_last_7")

    out = last_game.merge(counts, on=["season", "team"], how="left")
    out["games_in_last_7"] = out["games_in_last_7"].fillna(0).astype(int)
    return out[["season", "team", "days_rest", "games_in_last_7"]]


def add_rest_matchup(
    games: pd.DataFrame, rest_frame: pd.DataFrame
) -> pd.DataFrame:
    """Join rest features onto matchup rows as a/b columns and emit diff."""
    if games.empty or rest_frame.empty:
        return games
    a = rest_frame.add_prefix("a_").rename(
        columns={"a_season": "season", "a_team": "team_a"}
    )
    b = rest_frame.add_prefix("b_").rename(
        columns={"b_season": "season", "b_team": "team_b"}
    )
    out = games.merge(a, on=["season", "team_a"], how="left")
    out = out.merge(b, on=["season", "team_b"], how="left")
    if "a_days_rest" in out.columns and "b_days_rest" in out.columns:
        out["diff_days_rest"] = out["a_days_rest"] - out["b_days_rest"]
    if "a_games_in_last_7" in out.columns and "b_games_in_last_7" in out.columns:
        out["diff_games_in_last_7"] = out["a_games_in_last_7"] - out["b_games_in_last_7"]
    return out
