"""Home / away / neutral splits as team-season features.

Tournament games are neutral-site. A team that dominates at home but
crumbles on the road in the regular season is a worse bet in the
tournament than a team with balanced home/road splits, even at equal
overall record. This signal is under-exploited by naive models that
only look at overall W/L and point differential.
"""
from __future__ import annotations

from datetime import date

import pandas as pd


def build_site_splits(
    regular_season_games: pd.DataFrame,
    cutoff: date,
) -> pd.DataFrame:
    """Per-team splits by site. Returns wide frame with one row per team."""
    if regular_season_games.empty:
        return pd.DataFrame(columns=[
            "season", "team",
            "home_winpct", "home_margin",
            "away_winpct", "away_margin",
            "neutral_winpct", "neutral_margin",
        ])

    df = regular_season_games.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] < cutoff]
    if df.empty:
        return pd.DataFrame(columns=[
            "season", "team",
            "home_winpct", "home_margin",
            "away_winpct", "away_margin",
            "neutral_winpct", "neutral_margin",
        ])

    left = df[["season", "team_a", "team_b", "score_a", "score_b", "site"]].rename(
        columns={"team_a": "team", "score_a": "pf", "score_b": "pa"}
    )
    right = df.copy()
    right["site"] = right["site"].map(
        {"home": "away", "away": "home", "neutral": "neutral"}
    )
    right = right[["season", "team_b", "team_a", "score_b", "score_a", "site"]].rename(
        columns={"team_b": "team", "score_b": "pf", "score_a": "pa"}
    )
    stacked = pd.concat([left, right], ignore_index=True)
    stacked["win"] = (stacked["pf"] > stacked["pa"]).astype(int)
    stacked["margin"] = stacked["pf"] - stacked["pa"]

    pieces = []
    for site_name in ("home", "away", "neutral"):
        sub = stacked[stacked["site"] == site_name]
        if sub.empty:
            continue
        agg = sub.groupby(["season", "team"]).agg(
            winpct=("win", "mean"), margin=("margin", "mean"),
        ).reset_index()
        agg = agg.rename(columns={
            "winpct": f"{site_name}_winpct", "margin": f"{site_name}_margin"
        })
        pieces.append(agg)

    if not pieces:
        return pd.DataFrame()
    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on=["season", "team"], how="outer")
    return out
