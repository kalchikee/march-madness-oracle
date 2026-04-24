"""Late-season-weighted SoS and February/March specific form.

Novel-signal hypothesis: by late February rosters have stabilized
(injuries healed, rotations settled, transfers integrated) and motivation
peaks heading into conference tournaments. Performance in the last 6
weeks of the regular season should predict tournament performance
better than year-long averages.

We implement this as a time-decay weighted aggregate: each game's
contribution is multiplied by exp(-(cutoff - date).days / tau), where
smaller tau = heavier late-season emphasis. The tau parameter itself
becomes a tunable hyperparameter in the feature experiments config.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def time_decay_weighted_features(
    regular_season_games: pd.DataFrame,
    cutoff: date,
    tau_days: int = 30,
) -> pd.DataFrame:
    """Per-team weighted-avg margin, win pct, and opp quality proxy.

    tau_days: half-life for exponential decay. tau=30 means a game 30
    days before cutoff counts half as much as a game on cutoff day.
    """
    if regular_season_games.empty:
        return pd.DataFrame(columns=[
            "season", "team", "late_weighted_margin",
            "late_weighted_winpct", "feb_mar_games", "feb_mar_winpct",
        ])

    df = regular_season_games.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] < cutoff]
    if df.empty:
        return pd.DataFrame(columns=[
            "season", "team", "late_weighted_margin",
            "late_weighted_winpct", "feb_mar_games", "feb_mar_winpct",
        ])

    left = df[["season", "date", "team_a", "team_b", "score_a", "score_b"]].rename(
        columns={"team_a": "team", "team_b": "opp", "score_a": "pf", "score_b": "pa"}
    )
    right = df[["season", "date", "team_b", "team_a", "score_b", "score_a"]].rename(
        columns={"team_b": "team", "team_a": "opp", "score_b": "pf", "score_a": "pa"}
    )
    stacked = pd.concat([left, right], ignore_index=True)
    stacked["win"] = (stacked["pf"] > stacked["pa"]).astype(int)
    stacked["margin"] = stacked["pf"] - stacked["pa"]

    days_back = np.array([(cutoff - d).days for d in stacked["date"]], dtype=float)
    stacked["weight"] = np.exp(-days_back / float(tau_days))

    feb_mar_start = cutoff - timedelta(days=45)
    stacked["is_late"] = stacked["date"] >= feb_mar_start

    def _weighted(group: pd.DataFrame) -> pd.Series:
        w = group["weight"].to_numpy()
        wsum = w.sum()
        if wsum == 0:
            return pd.Series({
                "late_weighted_margin": 0.0,
                "late_weighted_winpct": 0.5,
                "feb_mar_games": 0,
                "feb_mar_winpct": float("nan"),
            })
        late = group[group["is_late"]]
        return pd.Series({
            "late_weighted_margin": float((group["margin"].to_numpy() * w).sum() / wsum),
            "late_weighted_winpct": float((group["win"].to_numpy() * w).sum() / wsum),
            "feb_mar_games": int(len(late)),
            "feb_mar_winpct": (
                float(late["win"].mean()) if len(late) else float("nan")
            ),
        })

    out = stacked.groupby(["season", "team"], group_keys=False).apply(_weighted).reset_index()
    return out


def late_season_sos_weighted(
    regular_season_games: pd.DataFrame,
    team_season: pd.DataFrame,
    cutoff: date,
    tau_days: int = 30,
) -> pd.DataFrame:
    """Opponent-strength-weighted late-season record.

    Uses each opponent's season point_diff_per_game as a coarse strength
    proxy (SRS would be better; this works with what the team-season
    aggregator already produces).
    """
    if regular_season_games.empty or team_season.empty:
        return pd.DataFrame(columns=["season", "team", "late_sos"])

    df = regular_season_games.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] < cutoff]
    if df.empty:
        return pd.DataFrame(columns=["season", "team", "late_sos"])

    strength = team_season.set_index(["season", "team"])["point_diff_per_game"].to_dict()

    left = df[["season", "date", "team_a", "team_b"]].rename(
        columns={"team_a": "team", "team_b": "opp"}
    )
    right = df[["season", "date", "team_b", "team_a"]].rename(
        columns={"team_b": "team", "team_a": "opp"}
    )
    stacked = pd.concat([left, right], ignore_index=True)
    stacked["opp_strength"] = stacked.apply(
        lambda r: strength.get((r["season"], r["opp"]), 0.0), axis=1
    )
    days_back = np.array(
        [(cutoff - d).days for d in stacked["date"]], dtype=float
    )
    stacked["weight"] = np.exp(-days_back / float(tau_days))

    def _weighted(group: pd.DataFrame) -> float:
        w = group["weight"].to_numpy()
        if w.sum() == 0:
            return 0.0
        return float((group["opp_strength"].to_numpy() * w).sum() / w.sum())

    out = stacked.groupby(["season", "team"]).apply(_weighted).reset_index(
        name="late_sos"
    )
    return out
