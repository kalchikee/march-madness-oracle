"""Final honest model: tournament-only features + prior-year Torvik rating.

Prior-year-rating trick: to predict season S, use each team's Torvik
rating from season S-1. Season S-1 ratings CANNOT know anything about
season S games, so this is zero-leakage by construction.

This gives the model current-program-strength information without any
chance of target contamination. Caveat: rosters change, so S-1 rating
is a noisy proxy for S strength.
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd

from madness.config import INTERIM_DIR, PROCESSED_DIR
from madness.features.coach_from_tournament import (
    build_school_tourney_experience,
    rolling_seed_upset_rate,
)
from madness.features.tournament import add_round_index
from madness.logging_setup import configure, get_logger
from madness.models.logistic import LogisticModel
from madness.models.registry import save_champion
from madness.train.backtest import walk_forward_backtest

configure()
log = get_logger(__name__)


_TORVIK_NAME_FIXES = {
    "Connecticut": "UConn",
    "Alabama-Birmingham": "UAB",
    "Central Florida": "UCF",
    "Massachusetts": "UMass",
    "Nevada-Las Vegas": "UNLV",
    "Texas-El Paso": "UTEP",
    "Southern California": "USC",
    "Brigham Young": "BYU",
    "Southern Methodist": "SMU",
    "Louisiana State": "LSU",
    "Texas Christian": "TCU",
    "North Carolina State": "NC State",
    "Mississippi": "Ole Miss",
    "Virginia Commonwealth": "VCU",
    "Pittsburgh": "Pitt",
    "Miami FL": "Miami (FL)",
    "Miami Fla.": "Miami (FL)",
    "North Carolina-Asheville": "UNC Asheville",
    "North Carolina-Greensboro": "UNC Greensboro",
    "North Carolina-Wilmington": "UNC Wilmington",
    "Loyola (IL)": "Loyola Chicago",
    "Loyola Chicago": "Loyola Chicago",
    "St. John's (NY)": "St. John's",
    "Saint Mary's (CA)": "Saint Mary's",
    "Cal-Santa Barbara": "UC Santa Barbara",
    "California-Santa Barbara": "UC Santa Barbara",
    "Long Island University": "Long Island",
    "Detroit Mercy": "Detroit",
}


def normalize_torvik_team(name: str) -> str:
    n = (name or "").strip()
    return _TORVIK_NAME_FIXES.get(n, n)


def load_torvik_prior() -> pd.DataFrame:
    """Build a lookup of prior-year ratings."""
    frames = []
    for p in (INTERIM_DIR / "torvik").glob("season_*.parquet"):
        frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["team"] = df["team"].apply(normalize_torvik_team)
    # Key features
    keep = ["season", "team", "adj_oe", "adj_de", "barthag", "adj_tempo", "wab", "rank"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    # Coerce numeric
    for c in ("adj_oe", "adj_de", "barthag", "adj_tempo", "wab", "rank"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Shift season +1 so feature is for predictions OF season+1
    df["for_season"] = df["season"] + 1
    df = df.drop(columns=["season"])
    df = df.rename(columns={"for_season": "season"})
    df = df.rename(columns={c: f"prior_{c}" for c in df.columns if c not in ("season", "team")})
    return df


def build() -> tuple[pd.DataFrame, list[str]]:
    tourn = pd.read_parquet(PROCESSED_DIR / "tournament_games.parquet")

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
    data = add_round_index(data, "round_name")
    data["seed_diff"] = data["seed_b"] - data["seed_a"]
    data["seed_sum"] = data["seed_a"] + data["seed_b"]
    data["seed_a_log"] = np.log(data["seed_a"])
    data["seed_b_log"] = np.log(data["seed_b"])
    data["seed_diff_sq"] = data["seed_diff"] ** 2

    # School history (walk-forward-safe)
    school_exp = build_school_tourney_experience(tourn)
    a = school_exp.add_prefix("a_").rename(columns={"a_season": "season", "a_team": "team_a"})
    b = school_exp.add_prefix("b_").rename(columns={"b_season": "season", "b_team": "team_b"})
    data = data.merge(a, on=["season", "team_a"], how="left")
    data = data.merge(b, on=["season", "team_b"], how="left")
    for col in ("school_career_tourney_wins", "school_career_f4s",
                "school_career_e8s", "school_last5_tourney_wins"):
        ac, bc = f"a_{col}", f"b_{col}"
        if ac in data.columns and bc in data.columns:
            data[ac] = data[ac].fillna(0)
            data[bc] = data[bc].fillna(0)
            data[f"diff_{col}"] = data[ac] - data[bc]

    # Historical upset rate
    hist = rolling_seed_upset_rate(tourn)
    if not hist.empty:
        hist2 = hist.rename(columns={"round_idx": "round"})
        data["seed_min"] = data[["seed_a", "seed_b"]].min(axis=1)
        data["seed_max"] = data[["seed_a", "seed_b"]].max(axis=1)
        data = data.merge(hist2, on=["season", "round", "seed_min", "seed_max"], how="left")
        data["hist_upset_rate"] = data["hist_upset_rate"].fillna(0.5)
        data["hist_n"] = data["hist_n"].fillna(0)
        data["favorite_is_a"] = (data["seed_a"] < data["seed_b"]).astype(int)
        data["hist_fav_win_rate"] = np.where(
            data["favorite_is_a"] == 1,
            1 - data["hist_upset_rate"],
            data["hist_upset_rate"],
        )

    # Prior-year Torvik
    torvik = load_torvik_prior()
    if not torvik.empty:
        pa = torvik.add_prefix("a_").rename(
            columns={"a_season": "season", "a_team": "team_a"}
        )
        pb = torvik.add_prefix("b_").rename(
            columns={"b_season": "season", "b_team": "team_b"}
        )
        data = data.merge(pa, on=["season", "team_a"], how="left")
        data = data.merge(pb, on=["season", "team_b"], how="left")
        for col in ("prior_adj_oe", "prior_adj_de", "prior_barthag",
                    "prior_adj_tempo", "prior_wab", "prior_rank"):
            ac, bc = f"a_{col}", f"b_{col}"
            if ac in data.columns and bc in data.columns:
                data[f"diff_{col}"] = data[ac] - data[bc]

    feat_cols = (
        ["seed_diff", "seed_sum", "seed_a_log", "seed_b_log", "seed_diff_sq", "round"]
        + [c for c in data.columns if c.startswith("diff_school_")]
        + [c for c in data.columns if c.startswith("diff_prior_")]
        + ["hist_fav_win_rate", "hist_n", "favorite_is_a"]
    )
    feat_cols = [c for c in feat_cols if c in data.columns]
    return data, feat_cols


def main() -> int:
    data, feat_cols = build()
    log.info("final_dataset", rows=len(data), feats=len(feat_cols))
    print(f"Rows: {len(data)}  Features: {len(feat_cols)}")
    print(f"Features: {feat_cols}")

    # Only train on seasons where Torvik data exists (2009+ because we need PRIOR year 2008+)
    torvik_eligible = data[data["season"] >= 2009].copy()
    # Drop rows where prior Torvik didn't join for BOTH teams
    prior_cols = [c for c in feat_cols if c.startswith("diff_prior_")]
    any_prior = torvik_eligible[prior_cols].notna().any(axis=1)
    torvik_eligible = torvik_eligible[any_prior].copy()
    print(f"\nTorvik-eligible rows (2009+, prior ratings available): {len(torvik_eligible)}")

    result = walk_forward_backtest(
        torvik_eligible, "target", feat_cols, lambda: LogisticModel(C=0.5),
        min_train_seasons=4, holdout_last=3,
    )
    summary = result.summary()
    print("\n=== FINAL MODEL (tournament + prior Torvik) — walk-forward ===")
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "per_season"},
        indent=2, default=float,
    ))

    # Per-round
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

    # TRUE holdout evaluation
    train = torvik_eligible[torvik_eligible["season"] < 2022]
    holdout = torvik_eligible[torvik_eligible["season"].isin([2022, 2023, 2024])].copy()
    champ = LogisticModel(C=0.5)
    champ.fit(train[feat_cols], train["target"].to_numpy().astype(int))
    holdout["prob"] = champ.predict_proba(holdout[feat_cols])
    winner_rows = holdout[holdout["target"] == 1].copy()
    winner_rows["correct"] = winner_rows["prob"] >= 0.5
    print(f"\n=== TRUE HOLDOUT (2022-2024, never seen) ===")
    print(f"Game-level accuracy: {winner_rows['correct'].mean()*100:.2f}%")
    from madness.features.registry import ROUND_INDEX
    rev = {v: k for k, v in ROUND_INDEX.items()}
    for rnd in sorted(winner_rows["round"].unique()):
        sub = winner_rows[winner_rows["round"] == rnd]
        print(f"  {rev.get(rnd, rnd)}: {sub['correct'].mean()*100:.1f}% (n={len(sub)})")

    # Save champion
    save_champion(champ, metrics={
        "walk_forward_mean_accuracy": summary["mean_accuracy"],
        "walk_forward_mean_log_loss": summary["mean_log_loss"],
        "walk_forward_mean_brier": summary["mean_brier"],
        "holdout_game_accuracy": float(winner_rows["correct"].mean()),
        "n_features": len(feat_cols),
        "model_family": "tournament_plus_prior_torvik_logistic",
        "holdout_seasons": [2022, 2023, 2024],
        "data_window": "2009-2024 (prior-year Torvik required)",
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
