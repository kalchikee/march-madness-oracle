"""Tournament-context features: round, seed matchup upset rate, region."""
from __future__ import annotations

import pandas as pd

from madness.features.registry import ROUND_INDEX


def add_round_index(games: pd.DataFrame, round_col: str = "round_name") -> pd.DataFrame:
    if round_col not in games.columns:
        return games
    games = games.copy()
    games["round"] = games[round_col].map(ROUND_INDEX).fillna(-1).astype(int)
    return games


def add_seed_features(games: pd.DataFrame) -> pd.DataFrame:
    """Requires seed_a and seed_b columns (oriented to symmetric game)."""
    if "seed_a" not in games.columns or "seed_b" not in games.columns:
        return games
    games = games.copy()
    games["seed_diff"] = games["seed_b"] - games["seed_a"]
    games["seed_sum"] = games["seed_a"] + games["seed_b"]
    games["is_upset_potential"] = (games["seed_a"] < games["seed_b"]).astype(int)
    return games


def compute_seed_matchup_upset_rates(historic_games: pd.DataFrame) -> pd.DataFrame:
    """Historical upset rate (higher-seed win) by (seed_a, seed_b, round).

    Returns a table indexable via left-join in predict-time.
    """
    if historic_games.empty:
        return pd.DataFrame(columns=["round", "seed_a", "seed_b", "upset_rate", "n"])
    # Upset = lower-numbered seed lost. Use canonical ordering (min, max).
    df = historic_games.copy()
    df["seed_min"] = df[["seed_winner", "seed_loser"]].min(axis=1)
    df["seed_max"] = df[["seed_winner", "seed_loser"]].max(axis=1)
    df["upset"] = df["seed_winner"] > df["seed_loser"]
    grp = df.groupby(["round", "seed_min", "seed_max"]).agg(
        upset_rate=("upset", "mean"),
        n=("upset", "size"),
    ).reset_index()
    return grp
