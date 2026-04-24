"""Coach tournament-experience computed from tournament history alone.

This is purely a walk-forward accumulation of tournament wins and Final
Four appearances BY TEAM (coach proxy). The "coach" dimension requires
an external coach table to join on; if coaches.csv isn't available, we
fall back to using the school itself as a proxy for "program tournament
experience" — a coarser but still useful signal.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from madness.features.registry import ROUND_INDEX


def build_school_tourney_experience(
    tournament_games: pd.DataFrame,
) -> pd.DataFrame:
    """Per-(season, team): career tournament W, F4, E8 BEFORE this season."""
    if tournament_games.empty:
        return pd.DataFrame(columns=[
            "season", "team",
            "school_career_tourney_wins", "school_career_f4s", "school_career_e8s",
        ])

    df = tournament_games.copy()
    df["round_idx"] = df["round_name"].map(ROUND_INDEX).fillna(-1).astype(int)

    # Every winner of a game gets +1 tourney win
    wins = df.rename(columns={"team_winner": "team"})[[
        "season", "team", "round_idx"
    ]].copy()
    wins["won"] = 1
    wins["reached_f4"] = (wins["round_idx"] >= ROUND_INDEX["Final Four"]).astype(int)
    wins["reached_e8"] = (wins["round_idx"] >= ROUND_INDEX["Elite Eight"]).astype(int)

    wins = wins.sort_values(["team", "season", "round_idx"])
    # Cumulative counts: career total through end of season S
    agg_by_season = wins.groupby(["team", "season"]).agg(
        season_wins=("won", "sum"),
        season_reached_f4=("reached_f4", "max"),
        season_reached_e8=("reached_e8", "max"),
    ).reset_index()
    agg_by_season = agg_by_season.sort_values(["team", "season"])

    # Shift so the feature for season S reflects only S-1 and earlier
    agg_by_season["school_career_tourney_wins"] = (
        agg_by_season.groupby("team")["season_wins"].cumsum()
        - agg_by_season["season_wins"]
    )
    agg_by_season["school_career_f4s"] = (
        agg_by_season.groupby("team")["season_reached_f4"].cumsum()
        - agg_by_season["season_reached_f4"]
    )
    agg_by_season["school_career_e8s"] = (
        agg_by_season.groupby("team")["season_reached_e8"].cumsum()
        - agg_by_season["season_reached_e8"]
    )

    # Recent (5-year rolling)
    def _recent(group: pd.DataFrame) -> pd.DataFrame:
        s = group.set_index("season")["season_wins"]
        out = group.copy()
        out["school_last5_tourney_wins"] = [
            int(s.loc[(s.index >= y - 5) & (s.index < y)].sum()) for y in group["season"]
        ]
        return out

    agg_by_season = agg_by_season.groupby("team", group_keys=False).apply(_recent)

    return agg_by_season[[
        "season", "team",
        "school_career_tourney_wins", "school_career_f4s",
        "school_career_e8s", "school_last5_tourney_wins",
    ]]


def build_coach_tourney_experience(
    tournament_games: pd.DataFrame,
    coaches_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-(season, coach): career tournament wins BEFORE this season.

    Falls back to empty if no coach table is available.
    """
    if tournament_games.empty or coaches_df.empty:
        return pd.DataFrame(columns=[
            "season", "coach_name",
            "coach_career_tourney_wins", "coach_career_f4s", "coach_career_e8s",
        ])
    df = tournament_games.copy()
    df["round_idx"] = df["round_name"].map(ROUND_INDEX).fillna(-1).astype(int)

    # Attach coach to the winner
    c = coaches_df[["season", "team", "coach_name"]].copy()
    wins = df.rename(columns={"team_winner": "team"})[[
        "season", "team", "round_idx",
    ]]
    wins = wins.merge(c, on=["season", "team"], how="left").dropna(subset=["coach_name"])
    wins["won"] = 1
    wins["reached_f4"] = (wins["round_idx"] >= ROUND_INDEX["Final Four"]).astype(int)
    wins["reached_e8"] = (wins["round_idx"] >= ROUND_INDEX["Elite Eight"]).astype(int)

    agg = wins.groupby(["coach_name", "season"]).agg(
        season_wins=("won", "sum"),
        season_f4=("reached_f4", "max"),
        season_e8=("reached_e8", "max"),
    ).reset_index().sort_values(["coach_name", "season"])

    agg["coach_career_tourney_wins"] = (
        agg.groupby("coach_name")["season_wins"].cumsum() - agg["season_wins"]
    )
    agg["coach_career_f4s"] = (
        agg.groupby("coach_name")["season_f4"].cumsum() - agg["season_f4"]
    )
    agg["coach_career_e8s"] = (
        agg.groupby("coach_name")["season_e8"].cumsum() - agg["season_e8"]
    )
    return agg[[
        "season", "coach_name",
        "coach_career_tourney_wins", "coach_career_f4s", "coach_career_e8s",
    ]]


def rolling_seed_upset_rate(
    tournament_games: pd.DataFrame, round_col: str = "round_name"
) -> pd.DataFrame:
    """For each (season, seed_min, seed_max, round), the historical upset
    rate using ONLY prior seasons.

    Returns a lookup table you can left-join onto matchups.
    """
    if tournament_games.empty:
        return pd.DataFrame()

    from madness.features.registry import ROUND_INDEX
    df = tournament_games.copy()
    df["round_idx"] = df[round_col].map(ROUND_INDEX).fillna(-1).astype(int)
    df["seed_min"] = df[["seed_winner", "seed_loser"]].min(axis=1)
    df["seed_max"] = df[["seed_winner", "seed_loser"]].max(axis=1)
    df["upset"] = (df["seed_winner"] > df["seed_loser"]).astype(int)

    # Sort by season then compute "as of prior season" via expanding window
    df = df.sort_values("season")
    rows = []
    seasons = sorted(df["season"].unique())
    for s in seasons:
        history = df[df["season"] < s]
        if history.empty:
            continue
        agg = history.groupby(["round_idx", "seed_min", "seed_max"]).agg(
            hist_upset_rate=("upset", "mean"),
            hist_n=("upset", "size"),
        ).reset_index()
        agg["season"] = s
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
