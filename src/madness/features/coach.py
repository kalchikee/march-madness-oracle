"""Coach tournament-experience features.

Novel-signal hypothesis: coaches with prior deep tournament experience
outperform in S16+ rounds vs. tournament-first-timers, independent of
roster talent. We compute cumulative career counts as-of each tournament.
"""
from __future__ import annotations

import pandas as pd

from madness.features.registry import ROUND_INDEX


def build_coach_features(
    tournament_games: pd.DataFrame,
    coaches: pd.DataFrame,
) -> pd.DataFrame:
    """Compute cumulative coach tournament wins / F4s / E8s before each season.

    coaches schema: season, team, coach_name
    tournament_games schema: season, round_name, team_winner, team_loser, ...
    """
    if tournament_games.empty or coaches.empty:
        return pd.DataFrame(columns=[
            "season", "coach_name", "coach_tourney_wins",
            "coach_final_fours", "coach_elite_eights"
        ])

    t = tournament_games.copy()
    t["round_idx"] = t["round_name"].map(ROUND_INDEX).fillna(-1).astype(int)
    wins = t.rename(columns={"team_winner": "team"})[["season", "team", "round_idx"]].copy()
    wins["won"] = 1

    # Join coach by (season, team)
    c = coaches[["season", "team", "coach_name"]].copy()
    joined = wins.merge(c, on=["season", "team"], how="left").dropna(subset=["coach_name"])

    joined = joined.sort_values(["coach_name", "season"])

    # Career cumulative — but only BEFORE current season
    joined["career_wins_prior"] = (
        joined.groupby("coach_name")["won"].cumsum() - joined["won"]
    )
    joined["is_f4"] = (joined["round_idx"] >= ROUND_INDEX["Final Four"]).astype(int)
    joined["is_e8"] = (joined["round_idx"] >= ROUND_INDEX["Elite Eight"]).astype(int)
    joined["career_f4_prior"] = (
        joined.groupby("coach_name")["is_f4"].cumsum() - joined["is_f4"]
    )
    joined["career_e8_prior"] = (
        joined.groupby("coach_name")["is_e8"].cumsum() - joined["is_e8"]
    )

    out = joined.groupby(["season", "coach_name"]).agg(
        coach_tourney_wins=("career_wins_prior", "max"),
        coach_final_fours=("career_f4_prior", "max"),
        coach_elite_eights=("career_e8_prior", "max"),
    ).reset_index()
    return out
