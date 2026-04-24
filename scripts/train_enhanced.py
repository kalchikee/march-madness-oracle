"""Enhanced baseline: seeds + SRS + SoS + shooting% + home/away gap.

Uses tournament outcomes (1985-2024) joined with school-stats (2003-2024).
Training restricted to 2003+ because that's when stats coverage is complete.
Reports both logistic and XGBoost results.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from madness.config import INTERIM_DIR, PROCESSED_DIR
from madness.features.tournament import add_round_index
from madness.logging_setup import configure, get_logger
from madness.models.logistic import LogisticModel
from madness.models.registry import save_champion
from madness.train.backtest import walk_forward_backtest

configure()
log = get_logger(__name__)


# Name mapping between SR tournament pages (display names) and SR school-stats
# pages (canonical-ish names). School-stats uses abbreviations and appends
# "NCAA" to tournament teams. This list was built iteratively from mismatch
# logs — extend as needed.
_TOURNAMENT_TO_STATS = {
    "UConn": "Connecticut",
    "Connecticut": "Connecticut",
    "UAB": "Alabama-Birmingham",
    "Alabama-Birmingham": "Alabama-Birmingham",
    "UCF": "Central Florida",
    "Central Florida": "Central Florida",
    "UMass": "Massachusetts",
    "Massachusetts": "Massachusetts",
    "UNLV": "Nevada-Las Vegas",
    "UTEP": "Texas-El Paso",
    "UTSA": "Texas-San Antonio",
    "USC": "Southern California",
    "Southern California": "Southern California",
    "BYU": "Brigham Young",
    "Brigham Young": "Brigham Young",
    "SMU": "Southern Methodist",
    "Southern Methodist": "Southern Methodist",
    "LSU": "Louisiana State",
    "Louisiana State": "Louisiana State",
    "TCU": "Texas Christian",
    "Texas Christian": "Texas Christian",
    "St. John's": "St. John's (NY)",
    "St. John's (NY)": "St. John's (NY)",
    "NC State": "North Carolina State",
    "North Carolina State": "North Carolina State",
    "Ole Miss": "Mississippi",
    "Mississippi": "Mississippi",
    "VCU": "Virginia Commonwealth",
    "Virginia Commonwealth": "Virginia Commonwealth",
    "Pitt": "Pittsburgh",
    "Pittsburgh": "Pittsburgh",
    "UNC Asheville": "North Carolina-Asheville",
    "UNC Greensboro": "North Carolina-Greensboro",
    "UNC Wilmington": "North Carolina-Wilmington",
    "UC Irvine": "California-Irvine",
    "UC Santa Barbara": "California-Santa Barbara",
    "UCSB": "California-Santa Barbara",
    "Florida Gulf Coast": "Florida Gulf Coast",
    "Saint Mary's (CA)": "Saint Mary's (CA)",
    "Saint Mary's": "Saint Mary's (CA)",
    "Loyola Chicago": "Loyola (IL)",
    "Loyola (IL)": "Loyola (IL)",
    "Loyola-Chicago": "Loyola (IL)",
    "Miami (FL)": "Miami (FL)",
    "Miami": "Miami (FL)",
    "Cal State Fullerton": "Cal State Fullerton",
    "USF": "South Florida",
    "South Florida": "South Florida",
    "CSU Bakersfield": "Cal State Bakersfield",
    "Cal State Bakersfield": "Cal State Bakersfield",
    "Middle Tennessee": "Middle Tennessee",
    "MTSU": "Middle Tennessee",
    "Texas A&M-CC": "Texas A&M-Corpus Christi",
    "Texas A&M-Corpus Christi": "Texas A&M-Corpus Christi",
    "SE Louisiana": "Southeastern Louisiana",
    "SE Missouri State": "Southeast Missouri State",
    "Stephen F. Austin": "Stephen F. Austin",
    "UMBC": "Maryland-Baltimore County",
    "ETSU": "East Tennessee State",
    "East Tennessee State": "East Tennessee State",
    "Abilene Christian": "Abilene Christian",
    "Detroit": "Detroit Mercy",
    "Detroit Mercy": "Detroit Mercy",
    "UNC-Asheville": "North Carolina-Asheville",
    "UNC-Greensboro": "North Carolina-Greensboro",
    "UNC-Wilmington": "North Carolina-Wilmington",
    "Cal State Northridge": "Cal State Northridge",
    "Long Island": "Long Island University",
    "LIU": "Long Island University",
    "LIU Brooklyn": "Long Island University",
    "Arkansas-Pine Bluff": "Arkansas-Pine Bluff",
    "Arkansas-Little Rock": "Little Rock",
    "Little Rock": "Little Rock",
    "South Dakota St.": "South Dakota State",
    "South Dakota State": "South Dakota State",
    "North Dakota St.": "North Dakota State",
    "North Dakota State": "North Dakota State",
}


def _strip_ncaa_suffix(name: str) -> str:
    if not isinstance(name, str):
        return name
    return re.sub(r"\s*(NCAA|NIT|CBI|CIT)$", "", name).strip()


def normalize_tournament(name: str) -> str:
    n = (name or "").strip()
    return _TOURNAMENT_TO_STATS.get(n, n)


def load_school_stats() -> pd.DataFrame:
    path = INTERIM_DIR / "school_stats_all.parquet"
    df = pd.read_parquet(path)
    df["team"] = df["school_name"].apply(_strip_ncaa_suffix)
    # Keep the rate features we care about
    keep = [
        "season", "team", "wins", "losses", "win_loss_pct", "srs", "sos",
        "wins_home", "losses_home", "wins_visitor", "losses_visitor",
        "pts", "opp_pts", "fg_pct", "fg3_pct", "ft_pct",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    for c in ("wins", "losses", "wins_home", "losses_home",
              "wins_visitor", "losses_visitor", "pts", "opp_pts"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ("win_loss_pct", "srs", "sos", "fg_pct", "fg3_pct", "ft_pct"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    games = (df["wins"] + df["losses"]).replace(0, np.nan)
    df["point_diff_pg"] = (df["pts"] - df["opp_pts"]) / games
    df["pts_pg"] = df["pts"] / games
    df["opp_pts_pg"] = df["opp_pts"] / games
    home_g = (df["wins_home"] + df["losses_home"]).replace(0, np.nan)
    away_g = (df["wins_visitor"] + df["losses_visitor"]).replace(0, np.nan)
    df["home_winpct"] = df["wins_home"] / home_g
    df["away_winpct"] = df["wins_visitor"] / away_g
    df["home_away_gap"] = df["home_winpct"] - df["away_winpct"]
    return df


def build_dataset() -> tuple[pd.DataFrame, list[str]]:
    tourn = pd.read_parquet(PROCESSED_DIR / "tournament_games.parquet")
    tourn = tourn[tourn["season"] >= 2003].copy()
    tourn["team_winner"] = tourn["team_winner"].apply(normalize_tournament)
    tourn["team_loser"] = tourn["team_loser"].apply(normalize_tournament)

    ss = load_school_stats()

    # Symmetrize
    pos = tourn.rename(columns={
        "team_winner": "team_a", "team_loser": "team_b",
        "seed_winner": "seed_a", "seed_loser": "seed_b",
    }).copy()
    pos["target"] = 1
    neg = tourn.rename(columns={
        "team_loser": "team_a", "team_winner": "team_b",
        "seed_loser": "seed_a", "seed_winner": "seed_b",
    }).copy()
    neg["target"] = 0
    data = pd.concat([pos, neg], ignore_index=True)

    # Join team-season features
    a = ss.add_prefix("a_").rename(columns={"a_season": "season", "a_team": "team_a"})
    b = ss.add_prefix("b_").rename(columns={"b_season": "season", "b_team": "team_b"})
    data = data.merge(a, on=["season", "team_a"], how="left")
    data = data.merge(b, on=["season", "team_b"], how="left")

    # Diffs
    for col in ("srs", "sos", "win_loss_pct", "point_diff_pg",
                "pts_pg", "opp_pts_pg", "home_winpct", "away_winpct",
                "home_away_gap", "fg_pct", "fg3_pct", "ft_pct"):
        ac, bc = f"a_{col}", f"b_{col}"
        if ac in data.columns and bc in data.columns:
            data[f"diff_{col}"] = data[ac] - data[bc]

    # Seed features
    data = add_round_index(data, "round_name")
    data["seed_diff"] = data["seed_b"] - data["seed_a"]
    data["seed_sum"] = data["seed_a"] + data["seed_b"]
    data["seed_a_log"] = np.log(data["seed_a"])
    data["seed_b_log"] = np.log(data["seed_b"])

    feat_cols = [
        c for c in data.columns
        if c.startswith("diff_")
    ] + ["seed_diff", "seed_sum", "seed_a_log", "seed_b_log", "round"]

    # Report join coverage
    joined = data[feat_cols].notna().any(axis=1).mean()
    log.info("join_coverage", pct_with_any_feat=float(joined))

    # Drop rows with no features joined
    any_diff = data[[c for c in feat_cols if c.startswith("diff_")]].notna().any(axis=1)
    data = data[any_diff].copy()
    return data, feat_cols


def main() -> int:
    data, feat_cols = build_dataset()
    log.info("enhanced_dataset_ready", rows=len(data), features=len(feat_cols),
             seasons=int(data["season"].nunique()))

    print(f"Training rows: {len(data)}")
    print(f"Feature count: {len(feat_cols)}")
    print(f"Seasons: {sorted(data['season'].unique())}")

    # Walk-forward logistic
    result = walk_forward_backtest(
        data, "target", feat_cols, lambda: LogisticModel(C=0.5),
        min_train_seasons=6, holdout_last=3,
    )
    summary = result.summary()
    print("\n=== ENHANCED LOGISTIC (seeds + SRS + SoS + shooting%) ===")
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "per_season"},
        indent=2, default=float,
    ))

    # Per-round aggregate
    rows = []
    for fold in result.folds:
        for rnd, s in fold.by_round.items():
            rows.append({"season": fold.validation_season, "round": rnd, **s})
    per_round = pd.DataFrame(rows)
    if not per_round.empty:
        agg = per_round.groupby("round").agg(
            mean_acc=("accuracy", "mean"),
            mean_ll=("log_loss", "mean"),
            total_n=("n", "sum"),
            folds=("season", "count"),
        ).reset_index()
        print("\n=== Per-round walk-forward averages ===")
        print(agg.to_string(index=False))
        per_round.to_csv(PROCESSED_DIR / "enhanced_per_round.csv", index=False)

    # Save as champion
    holdout_seasons = sorted(data["season"].unique())[-3:]
    train_mask = ~data["season"].isin(holdout_seasons)
    champ = LogisticModel(C=0.5)
    champ.fit(data.loc[train_mask, feat_cols], data.loc[train_mask, "target"].to_numpy().astype(int))
    save_champion(champ, metrics={
        "mean_accuracy": summary["mean_accuracy"],
        "mean_log_loss": summary["mean_log_loss"],
        "mean_brier": summary["mean_brier"],
        "n_features": len(feat_cols),
        "model_family": "enhanced_logistic_with_srs_sos",
        "holdout_seasons": [int(s) for s in holdout_seasons],
    })
    log.info("champion_saved_enhanced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
