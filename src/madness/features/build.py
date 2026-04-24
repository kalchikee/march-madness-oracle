"""Orchestrates the full feature pipeline for training and inference.

Walk-forward-safe: every aggregate for season S uses only games played
BEFORE that season's tournament cutoff date. Enforced per season.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from madness.config import PROCESSED_DIR
from madness.features.late_season import (
    late_season_sos_weighted,
    time_decay_weighted_features,
)
from madness.features.matchup import build_matchup_features
from madness.features.momentum import last_n_form, longest_win_streak
from madness.features.rest import add_rest_matchup, days_rest_before
from madness.features.site_splits import build_site_splits
from madness.features.team_season import build_team_season_table
from madness.features.tournament import add_round_index, add_seed_features
from madness.logging_setup import get_logger
from madness.storage import write_parquet

log = get_logger(__name__)


def build_feature_table(
    tournament_games: pd.DataFrame,
    regular_season_games: pd.DataFrame,
    season_cutoff_dates: dict[int, date],
    tau_days: int = 30,
) -> pd.DataFrame:
    """Build a training feature table with walk-forward-safe aggregates."""
    if tournament_games.empty:
        log.warning("no_tournament_games")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for season, cutoff in season_cutoff_dates.items():
        season_games = tournament_games[tournament_games["season"] == season]
        if season_games.empty:
            continue

        rs = regular_season_games[regular_season_games["season"] == season] \
            if not regular_season_games.empty else regular_season_games

        ts = build_team_season_table(rs, cutoff)
        if ts.empty:
            # No per-game data for this season; skip (older seasons)
            log.info("skipping_season_no_rs", season=season)
            continue

        form = last_n_form(rs, cutoff)
        streak = longest_win_streak(rs, cutoff)
        late = time_decay_weighted_features(rs, cutoff, tau_days=tau_days)
        late_sos = late_season_sos_weighted(rs, ts, cutoff, tau_days=tau_days)
        sites = build_site_splits(rs, cutoff)
        rest = days_rest_before(rs, cutoff)

        ts = ts.merge(form, on=["season", "team"], how="left")
        ts = ts.merge(streak, on=["season", "team"], how="left")
        ts = ts.merge(late, on=["season", "team"], how="left")
        ts = ts.merge(late_sos, on=["season", "team"], how="left")
        ts = ts.merge(sites, on=["season", "team"], how="left")
        ts = ts.merge(rest, on=["season", "team"], how="left")

        season_frame = _symmetrize_and_merge(season_games, ts, season)
        if not season_frame.empty:
            # Add rest-matchup diffs (team-level rest already in ts via merge)
            season_frame = add_rest_matchup(season_frame, rest)
            frames.append(season_frame)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = add_round_index(out, "round_name")
    out = add_seed_features(out)
    path = PROCESSED_DIR / "features_train.parquet"
    write_parquet(out, path)
    log.info("features_written", rows=len(out), path=str(path))
    return out


def _symmetrize_and_merge(
    season_games: pd.DataFrame, team_season: pd.DataFrame, season: int
) -> pd.DataFrame:
    """Emit two rows per game: (winner, loser, target=1) and swapped target=0."""
    if season_games.empty:
        return pd.DataFrame()

    pos = season_games.rename(columns={
        "team_winner": "team_a", "team_loser": "team_b",
        "seed_winner": "seed_a", "seed_loser": "seed_b",
        "score_winner": "score_a", "score_loser": "score_b",
    }).copy()
    pos["target"] = 1
    if "score_a" in pos.columns and "score_b" in pos.columns:
        pos["margin"] = pos["score_a"] - pos["score_b"]

    neg = season_games.rename(columns={
        "team_loser": "team_a", "team_winner": "team_b",
        "seed_loser": "seed_a", "seed_winner": "seed_b",
        "score_loser": "score_a", "score_winner": "score_b",
    }).copy()
    neg["target"] = 0
    if "score_a" in neg.columns and "score_b" in neg.columns:
        neg["margin"] = neg["score_a"] - neg["score_b"]

    stacked = pd.concat([pos, neg], ignore_index=True)
    stacked["season"] = season
    return build_matchup_features(stacked, team_season)


def default_season_cutoffs(seasons: list[int]) -> dict[int, date]:
    """Standard cutoff: Selection Sunday — second Sunday of March.

    We use a conservative approximation: March 14 of the tournament year.
    (Selection Sunday typically falls Mar 13-17. Mar 14 avoids accidental
    inclusion of the selection-Sunday day itself.)
    """
    return {s: date(s, 3, 14) for s in seasons}