"""Minimum-viable baseline training pipeline.

Uses only data from:
  - Tournament bracket pages (for target labels: who beat whom, with seeds)
  - School-stats pages (for per-team season aggregates: W, L, SRS, SOS, shooting %s, etc.)

This deliberately skips the expensive per-game conference-schedule scrape.
School-stats alone should beat the 71% chalk baseline by encoding non-seed
signal (SRS, SOS, Four Factors) that the committee partially but not
fully bakes into seeding.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from madness.config import PROCESSED_DIR, current_season
from madness.features.tournament import add_round_index, add_seed_features
from madness.ingest.sports_reference import ingest_school_stats_range
from madness.ingest.tournament_results import build_canonical_tournament_table
from madness.logging_setup import configure, get_logger
from madness.models.logistic import LogisticModel
from madness.models.registry import save_champion
from madness.train.backtest import walk_forward_backtest
from madness.train.evaluate import round_breakdown

configure()
log = get_logger(__name__)


# --- School name normalization: SR tournament pages and school-stats pages
# sometimes disagree. school-stats appends "NCAA" to tourney teams, and
# both use variants ("UConn" vs "Connecticut"). Normalize before joining.

_NAME_FIXES = {
    "UConn": "Connecticut",
    "UAB": "Alabama-Birmingham",
    "UCF": "Central Florida",
    "UMass": "Massachusetts",
    "UNC Asheville": "North Carolina-Asheville",
    "UNC Greensboro": "North Carolina-Greensboro",
    "UNC Wilmington": "North Carolina-Wilmington",
    "UNLV": "Nevada-Las Vegas",
    "UTEP": "Texas-El Paso",
    "UTSA": "Texas-San Antonio",
    "USC": "Southern California",
    "Miami (FL)": "Miami (FL)",
    "BYU": "Brigham Young",
    "SMU": "Southern Methodist",
    "LSU": "Louisiana State",
    "TCU": "Texas Christian",
    "St. John's": "St. John's (NY)",
    "Saint Joseph's": "Saint Joseph's",
    "Saint Mary's (CA)": "Saint Mary's (CA)",
    "NC State": "North Carolina State",
    "Ole Miss": "Mississippi",
    "VCU": "Virginia Commonwealth",
    "Pitt": "Pittsburgh",
}


def normalize_team(name: str) -> str:
    if not isinstance(name, str):
        return name
    n = name.strip()
    # Strip SR's trailing "NCAA" / "NIT" tags on school-stats rows
    n = re.sub(r"\s*(NCAA|NIT|CBI|CIT)$", "", n)
    return _NAME_FIXES.get(n, n)


NUMERIC_FEATURES = [
    "wins", "losses", "win_loss_pct", "srs", "sos",
    "wins_home", "losses_home", "wins_visitor", "losses_visitor",
    "pts", "opp_pts", "fg_pct", "fg3_pct", "ft_pct",
    "trb", "ast", "stl", "blk", "tov", "pf",
]


def prepare_team_season(school_stats: pd.DataFrame) -> pd.DataFrame:
    """Select + rename key columns from school-stats for joining."""
    df = school_stats.copy()
    df["team"] = df["school_name"].apply(normalize_team)
    keep = ["season", "team"] + [c for c in NUMERIC_FEATURES if c in df.columns]
    df = df[keep].copy()
    # Per-game rates
    if "pts" in df.columns and "wins" in df.columns and "losses" in df.columns:
        games = df["wins"] + df["losses"]
        df["pts_per_game"] = df["pts"] / games.replace(0, np.nan)
        df["opp_pts_per_game"] = df["opp_pts"] / games.replace(0, np.nan)
        df["point_diff_per_game"] = df["pts_per_game"] - df["opp_pts_per_game"]
    # Home vs away win pct
    home_g = df.get("wins_home", 0) + df.get("losses_home", 0)
    away_g = df.get("wins_visitor", 0) + df.get("losses_visitor", 0)
    df["home_winpct"] = df.get("wins_home", 0) / home_g.replace(0, np.nan)
    df["away_winpct"] = df.get("wins_visitor", 0) / away_g.replace(0, np.nan)
    df["home_away_gap"] = df["home_winpct"] - df["away_winpct"]
    return df


def build_matchup_frame(
    tournament_games: pd.DataFrame, team_season: pd.DataFrame
) -> pd.DataFrame:
    """Emit a symmetrized matchup frame with diff/ratio features."""
    t = tournament_games.copy()
    t["team_winner"] = t["team_winner"].apply(normalize_team)
    t["team_loser"] = t["team_loser"].apply(normalize_team)

    pos = t.rename(columns={
        "team_winner": "team_a", "team_loser": "team_b",
        "seed_winner": "seed_a", "seed_loser": "seed_b",
    }).copy()
    pos["target"] = 1

    neg = t.rename(columns={
        "team_loser": "team_a", "team_winner": "team_b",
        "seed_loser": "seed_a", "seed_winner": "seed_b",
    }).copy()
    neg["target"] = 0

    stacked = pd.concat([pos, neg], ignore_index=True)

    a = team_season.add_prefix("a_").rename(
        columns={"a_season": "season", "a_team": "team_a"}
    )
    b = team_season.add_prefix("b_").rename(
        columns={"b_season": "season", "b_team": "team_b"}
    )
    merged = stacked.merge(a, on=["season", "team_a"], how="left")
    merged = merged.merge(b, on=["season", "team_b"], how="left")

    numeric_cols = [c for c in team_season.columns if c not in ("season", "team")]
    for c in numeric_cols:
        ac, bc = f"a_{c}", f"b_{c}"
        if ac in merged.columns and bc in merged.columns:
            merged[f"diff_{c}"] = merged[ac] - merged[bc]

    merged = add_round_index(merged, "round_name")
    merged = add_seed_features(merged)
    return merged


def main() -> int:
    start = 2003  # modern era with reliable stats
    end = current_season() - 2  # exclude last tournament for honest holdout

    log.info("loading_data", start=start, end=end)
    tourn = build_canonical_tournament_table(start, end)
    log.info("tournament_loaded", rows=len(tourn))
    if tourn.empty:
        log.error("no_tournament_data")
        return 1

    ss = ingest_school_stats_range(start, end)
    log.info("school_stats_loaded", rows=len(ss))
    if ss.empty:
        log.error("no_school_stats_data")
        return 1

    ts = prepare_team_season(ss)
    matchups = build_matchup_frame(tourn, ts)
    matchups.to_parquet(PROCESSED_DIR / "matchups_baseline.parquet", index=False)

    # Feature columns
    feat_cols = [c for c in matchups.columns if c.startswith("diff_")]
    feat_cols += ["seed_diff", "seed_sum"]
    feat_cols = [c for c in feat_cols if c in matchups.columns]
    log.info("feature_count", n=len(feat_cols), features=feat_cols[:10])

    # Drop rows with all-NaN features (unjoinable team names)
    before = len(matchups)
    matchups = matchups.dropna(subset=feat_cols, how="all")
    after = len(matchups)
    if after < before:
        log.warning("dropped_unjoinable_rows", dropped=before - after, kept=after)

    # Walk-forward backtest
    result = walk_forward_backtest(
        matchups, "target", feat_cols, lambda: LogisticModel(),
        min_train_seasons=8, holdout_last=2,
    )
    summary = result.summary()
    log.info("backtest_summary", **{k: v for k, v in summary.items() if k != "per_season"})
    print(json.dumps(summary, indent=2, default=float))

    # Per-round breakdown on the most recent validation fold
    if result.folds:
        last_fold = result.folds[-1]
        print(f"\nLast validation season: {last_fold.validation_season}")
        print(f"Round breakdown:")
        for rnd, stats in sorted(last_fold.by_round.items()):
            print(f"  round {rnd}: n={stats['n']} acc={stats['accuracy']:.3f} logloss={stats['log_loss']:.3f}")

    # Fit final champion on all train data
    X_full = matchups[feat_cols]
    y_full = matchups["target"].to_numpy().astype(int)
    champion = LogisticModel()
    champion.fit(X_full, y_full)
    save_champion(champion, metrics={
        "mean_accuracy": summary["mean_accuracy"],
        "mean_log_loss": summary["mean_log_loss"],
        "mean_brier": summary["mean_brier"],
    })
    log.info("champion_saved", mean_accuracy=summary["mean_accuracy"])

    # Write backtest report
    report_rows = []
    for fold in result.folds:
        for rnd, stats in fold.by_round.items():
            report_rows.append({
                "season": fold.validation_season,
                "round": rnd,
                "n": stats["n"],
                "accuracy": stats["accuracy"],
                "log_loss": stats["log_loss"],
            })
    if report_rows:
        pd.DataFrame(report_rows).to_csv(PROCESSED_DIR / "backtest_by_round.csv", index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
