"""Leakage-corrected training.

School-stats season totals INCLUDE NCAA tournament games, which would
leak the target. We subtract each team's tournament W, L, points for,
and points against to recover pre-tournament aggregates.

SRS and SOS are opponent-quality-weighted and cannot be cleanly
decontaminated by simple subtraction, so we DROP them entirely. Future
work: compute SRS from pre-tournament games only via the conference-
schedule scrape.
"""
from __future__ import annotations

import json
import re
import sys

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


_TOURNAMENT_TO_STATS = {
    "UConn": "Connecticut", "Connecticut": "Connecticut",
    "UAB": "Alabama-Birmingham", "UCF": "Central Florida",
    "UMass": "Massachusetts", "UNLV": "Nevada-Las Vegas",
    "UTEP": "Texas-El Paso", "UTSA": "Texas-San Antonio",
    "USC": "Southern California", "BYU": "Brigham Young",
    "SMU": "Southern Methodist", "LSU": "Louisiana State",
    "TCU": "Texas Christian", "St. John's": "St. John's (NY)",
    "NC State": "North Carolina State", "Ole Miss": "Mississippi",
    "VCU": "Virginia Commonwealth", "Pitt": "Pittsburgh",
    "Miami (FL)": "Miami (FL)", "Miami": "Miami (FL)",
    "USF": "South Florida", "UMBC": "Maryland-Baltimore County",
    "ETSU": "East Tennessee State", "UC Irvine": "California-Irvine",
    "UC Santa Barbara": "California-Santa Barbara",
    "UCSB": "California-Santa Barbara",
    "Loyola Chicago": "Loyola (IL)", "Loyola-Chicago": "Loyola (IL)",
    "Saint Mary's": "Saint Mary's (CA)",
    "Arkansas-Little Rock": "Little Rock",
    "Little Rock": "Little Rock",
    "Long Island": "Long Island University",
    "LIU Brooklyn": "Long Island University",
    "Detroit": "Detroit Mercy",
    "Middle Tennessee": "Middle Tennessee",
    "MTSU": "Middle Tennessee",
    "Texas A&M-CC": "Texas A&M-Corpus Christi",
}


def _strip_ncaa_suffix(name: str) -> str:
    return re.sub(r"\s*(NCAA|NIT|CBI|CIT)$", "", str(name)).strip()


def normalize_tournament(name: str) -> str:
    n = (name or "").strip()
    return _TOURNAMENT_TO_STATS.get(n, n)


def compute_tournament_contributions(tourn: pd.DataFrame) -> pd.DataFrame:
    """Per-team, per-season: wins/losses/points accrued in the NCAA tournament."""
    wins = tourn.rename(columns={"team_winner": "team"})[[
        "season", "team", "score_winner", "score_loser",
    ]].rename(columns={"score_winner": "pts_f", "score_loser": "pts_a"})
    wins["is_win"] = 1
    losses = tourn.rename(columns={"team_loser": "team"})[[
        "season", "team", "score_loser", "score_winner",
    ]].rename(columns={"score_loser": "pts_f", "score_winner": "pts_a"})
    losses["is_win"] = 0
    stacked = pd.concat([wins, losses], ignore_index=True)
    stacked["team"] = stacked["team"].apply(normalize_tournament)
    agg = stacked.groupby(["season", "team"]).agg(
        tourn_wins=("is_win", "sum"),
        tourn_losses=("is_win", lambda s: (s == 0).sum()),
        tourn_pts_for=("pts_f", "sum"),
        tourn_pts_against=("pts_a", "sum"),
    ).reset_index()
    return agg


def load_decontaminated_stats(tourn: pd.DataFrame) -> pd.DataFrame:
    ss = pd.read_parquet(INTERIM_DIR / "school_stats_all.parquet")
    ss["team"] = ss["school_name"].apply(_strip_ncaa_suffix)
    for c in ("wins", "losses", "pts", "opp_pts",
              "wins_home", "losses_home", "wins_visitor", "losses_visitor"):
        if c in ss.columns:
            ss[c] = pd.to_numeric(ss[c], errors="coerce")
    for c in ("win_loss_pct", "fg_pct", "fg3_pct", "ft_pct"):
        if c in ss.columns:
            ss[c] = pd.to_numeric(ss[c], errors="coerce")

    contrib = compute_tournament_contributions(tourn)
    ss = ss.merge(contrib, on=["season", "team"], how="left")
    for c in ("tourn_wins", "tourn_losses", "tourn_pts_for", "tourn_pts_against"):
        ss[c] = ss[c].fillna(0)

    # Subtract tournament contributions
    ss["pre_wins"] = ss["wins"] - ss["tourn_wins"]
    ss["pre_losses"] = ss["losses"] - ss["tourn_losses"]
    ss["pre_pts"] = ss["pts"] - ss["tourn_pts_for"]
    ss["pre_opp_pts"] = ss["opp_pts"] - ss["tourn_pts_against"]

    pre_games = (ss["pre_wins"] + ss["pre_losses"]).replace(0, np.nan)
    ss["pre_winpct"] = ss["pre_wins"] / pre_games
    ss["pre_pts_pg"] = ss["pre_pts"] / pre_games
    ss["pre_opp_pts_pg"] = ss["pre_opp_pts"] / pre_games
    ss["pre_point_diff_pg"] = ss["pre_pts_pg"] - ss["pre_opp_pts_pg"]

    # Home/away splits are NOT contaminated (tournament is neutral)
    home_g = (ss["wins_home"] + ss["losses_home"]).replace(0, np.nan)
    away_g = (ss["wins_visitor"] + ss["losses_visitor"]).replace(0, np.nan)
    ss["home_winpct"] = ss["wins_home"] / home_g
    ss["away_winpct"] = ss["wins_visitor"] / away_g
    ss["home_away_gap"] = ss["home_winpct"] - ss["away_winpct"]

    # FG/3P/FT percentages have tournament contamination but only slightly
    # (per-game rates, 35 games total with 1-6 tournament games). We drop
    # them to be strict — for the first honest baseline keep it clean.

    keep = [
        "season", "team",
        "pre_winpct", "pre_pts_pg", "pre_opp_pts_pg", "pre_point_diff_pg",
        "home_winpct", "away_winpct", "home_away_gap",
        "pre_wins", "pre_losses",
    ]
    return ss[keep].copy()


def build_dataset(tourn: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    tourn = tourn[tourn["season"] >= 2003].copy()
    tourn["team_winner"] = tourn["team_winner"].apply(normalize_tournament)
    tourn["team_loser"] = tourn["team_loser"].apply(normalize_tournament)

    ts = load_decontaminated_stats(tourn)

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

    a = ts.add_prefix("a_").rename(columns={"a_season": "season", "a_team": "team_a"})
    b = ts.add_prefix("b_").rename(columns={"b_season": "season", "b_team": "team_b"})
    data = data.merge(a, on=["season", "team_a"], how="left")
    data = data.merge(b, on=["season", "team_b"], how="left")

    for col in (
        "pre_winpct", "pre_point_diff_pg", "pre_pts_pg", "pre_opp_pts_pg",
        "home_winpct", "away_winpct", "home_away_gap",
        "pre_wins", "pre_losses",
    ):
        ac, bc = f"a_{col}", f"b_{col}"
        if ac in data.columns and bc in data.columns:
            data[f"diff_{col}"] = data[ac] - data[bc]

    data = add_round_index(data, "round_name")
    data["seed_diff"] = data["seed_b"] - data["seed_a"]
    data["seed_sum"] = data["seed_a"] + data["seed_b"]
    data["seed_a_log"] = np.log(data["seed_a"])
    data["seed_b_log"] = np.log(data["seed_b"])

    feat_cols = [c for c in data.columns if c.startswith("diff_")]
    feat_cols += ["seed_diff", "seed_sum", "seed_a_log", "seed_b_log", "round"]

    # Drop unjoinable rows
    any_diff = data[[c for c in feat_cols if c.startswith("diff_")]].notna().any(axis=1)
    before = len(data)
    data = data[any_diff].copy()
    log.info("join_coverage", kept=len(data), dropped=before - len(data))

    return data, feat_cols


def main() -> int:
    tourn = pd.read_parquet(PROCESSED_DIR / "tournament_games.parquet")
    data, feat_cols = build_dataset(tourn)
    print(f"Rows: {len(data)}  Features: {len(feat_cols)}  Seasons: {data['season'].nunique()}")

    result = walk_forward_backtest(
        data, "target", feat_cols, lambda: LogisticModel(C=0.5),
        min_train_seasons=6, holdout_last=3,
    )
    summary = result.summary()
    print("\n=== LEAKAGE-CORRECTED LOGISTIC ===")
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "per_season"},
        indent=2, default=float,
    ))

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
        ).reset_index()
        print("\n=== Per-round ===")
        print(agg.to_string(index=False))
        per_round.to_csv(PROCESSED_DIR / "leakfixed_per_round.csv", index=False)

    holdout_seasons = sorted(data["season"].unique())[-3:]
    train_mask = ~data["season"].isin(holdout_seasons)
    champ = LogisticModel(C=0.5)
    champ.fit(data.loc[train_mask, feat_cols], data.loc[train_mask, "target"].to_numpy().astype(int))
    save_champion(champ, metrics={
        "mean_accuracy": summary["mean_accuracy"],
        "mean_log_loss": summary["mean_log_loss"],
        "mean_brier": summary["mean_brier"],
        "n_features": len(feat_cols),
        "model_family": "leakfixed_logistic",
        "holdout_seasons": [int(s) for s in holdout_seasons],
        "note": "SRS/SOS dropped due to leakage; tournament W/L subtracted from school-stats",
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
