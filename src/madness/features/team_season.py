"""Team-season aggregates — one row per team per season.

CRITICAL: every aggregator takes `season_cutoff_date`. Features for the
2019 tournament must only use games played BEFORE the 2019 tournament
tipped off. This is the single most important guard against leakage.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import polars as pl


def build_team_season_table(
    regular_season_games: pd.DataFrame,
    season_cutoff_date: date,
) -> pd.DataFrame:
    """Aggregate per-team-per-season basics.

    Input schema: columns season, date, team_a, team_b, score_a, score_b,
                  site ('home'|'away'|'neutral') oriented on team_a.
    """
    if regular_season_games.empty:
        return pd.DataFrame()

    df = pl.from_pandas(regular_season_games).filter(
        pl.col("date") < pl.lit(season_cutoff_date)
    )

    left = df.select([
        pl.col("season"), pl.col("date"), pl.col("team_a").alias("team"),
        pl.col("team_b").alias("opp"),
        pl.col("score_a").alias("points_for"),
        pl.col("score_b").alias("points_against"),
        pl.col("site"),
    ])
    right = df.select([
        pl.col("season"), pl.col("date"), pl.col("team_b").alias("team"),
        pl.col("team_a").alias("opp"),
        pl.col("score_b").alias("points_for"),
        pl.col("score_a").alias("points_against"),
        pl.when(pl.col("site") == "home").then(pl.lit("away"))
        .when(pl.col("site") == "away").then(pl.lit("home"))
        .otherwise(pl.lit("neutral")).alias("site"),
    ])
    stacked = pl.concat([left, right], how="vertical")

    stacked = stacked.with_columns([
        (pl.col("points_for") > pl.col("points_against")).alias("win"),
        (pl.col("points_for") - pl.col("points_against")).alias("margin"),
    ])

    team_season = stacked.group_by(["season", "team"]).agg([
        pl.col("win").sum().alias("wins"),
        (~pl.col("win")).sum().alias("losses"),
        pl.col("margin").mean().alias("point_diff_per_game"),
        pl.col("points_for").mean().alias("points_for_pg"),
        pl.col("points_against").mean().alias("points_against_pg"),
        pl.col("win").count().alias("games_played"),
    ]).with_columns(
        (pl.col("wins") / pl.col("games_played")).alias("win_pct"),
    )

    return team_season.to_pandas()
