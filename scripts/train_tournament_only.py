"""Tournament-only feature training — pure walk-forward, zero leakage.

Features come EXCLUSIVELY from historical tournament results:
  - Seed features
  - School career tournament wins / F4s / E8s (shifted: only prior-year)
  - School last-5-year tournament wins
  - Historical upset rate for (seed_a, seed_b, round) using only prior seasons

Because these are all derived from tournament data itself, they are
impossible to leak. Training covers 1985-2024.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from madness.config import PROCESSED_DIR
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


def build() -> tuple[pd.DataFrame, list[str]]:
    tourn = pd.read_parquet(PROCESSED_DIR / "tournament_games.parquet")

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
    data = add_round_index(data, "round_name")

    # Seed features
    data["seed_diff"] = data["seed_b"] - data["seed_a"]
    data["seed_sum"] = data["seed_a"] + data["seed_b"]
    data["seed_a_log"] = np.log(data["seed_a"])
    data["seed_b_log"] = np.log(data["seed_b"])
    data["seed_diff_sq"] = data["seed_diff"] ** 2

    # School experience (walk-forward-safe: uses only prior seasons)
    school_exp = build_school_tourney_experience(tourn)
    a = school_exp.add_prefix("a_").rename(
        columns={"a_season": "season", "a_team": "team_a"}
    )
    b = school_exp.add_prefix("b_").rename(
        columns={"b_season": "season", "b_team": "team_b"}
    )
    data = data.merge(a, on=["season", "team_a"], how="left")
    data = data.merge(b, on=["season", "team_b"], how="left")

    for col in ("school_career_tourney_wins", "school_career_f4s",
                "school_career_e8s", "school_last5_tourney_wins"):
        ac, bc = f"a_{col}", f"b_{col}"
        if ac in data.columns and bc in data.columns:
            data[ac] = data[ac].fillna(0)
            data[bc] = data[bc].fillna(0)
            data[f"diff_{col}"] = data[ac] - data[bc]

    # Historical upset rate (walk-forward via rolling computation)
    hist = rolling_seed_upset_rate(tourn)
    if not hist.empty:
        data["seed_min"] = data[["seed_a", "seed_b"]].min(axis=1)
        data["seed_max"] = data[["seed_a", "seed_b"]].max(axis=1)
        data = data.merge(
            hist, on=["season", "round_idx" if "round_idx" in data.columns else "round",
                      "seed_min", "seed_max"], how="left",
            left_on=["season", "round", "seed_min", "seed_max"],
            right_on=["season", "round_idx", "seed_min", "seed_max"],
        ) if False else data  # need to be careful with merge keys
        # Simpler: merge explicitly on round
        pass

    # Explicit merge on historical upset rates
    if not hist.empty:
        data = data.drop(columns=[c for c in ("hist_upset_rate", "hist_n", "round_idx") if c in data.columns], errors="ignore")
        hist2 = hist.rename(columns={"round_idx": "round"})
        data["seed_min"] = data[["seed_a", "seed_b"]].min(axis=1)
        data["seed_max"] = data[["seed_a", "seed_b"]].max(axis=1)
        data = data.merge(hist2, on=["season", "round", "seed_min", "seed_max"], how="left")
        data["hist_upset_rate"] = data["hist_upset_rate"].fillna(0.5)  # Default 50/50 if no prior
        data["hist_n"] = data["hist_n"].fillna(0)
        # Direction: team_a is "favorite" if seed_a < seed_b, so prob_a_wins ≈ 1 - upset_rate
        # But for symmetrized data (both directions), we encode as abs.
        data["favorite_is_a"] = (data["seed_a"] < data["seed_b"]).astype(int)
        data["hist_fav_win_rate"] = np.where(
            data["favorite_is_a"] == 1,
            1 - data["hist_upset_rate"],
            data["hist_upset_rate"],
        )

    feat_cols = [
        "seed_diff", "seed_sum", "seed_a_log", "seed_b_log",
        "seed_diff_sq", "round",
        "diff_school_career_tourney_wins", "diff_school_career_f4s",
        "diff_school_career_e8s", "diff_school_last5_tourney_wins",
        "hist_fav_win_rate", "hist_n", "favorite_is_a",
    ]
    feat_cols = [c for c in feat_cols if c in data.columns]
    return data, feat_cols


def main() -> int:
    data, feat_cols = build()
    log.info("dataset", rows=len(data), feats=len(feat_cols))
    print(f"Rows: {len(data)}")
    print(f"Feature count: {len(feat_cols)}")
    print(f"Features: {feat_cols}")

    result = walk_forward_backtest(
        data, "target", feat_cols, lambda: LogisticModel(C=0.5),
        min_train_seasons=10, holdout_last=3,
    )
    summary = result.summary()
    print("\n=== TOURNAMENT-ONLY LOGISTIC (no external data, no leakage) ===")
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

    # Save champion
    holdout_seasons = sorted(data["season"].unique())[-3:]
    train_mask = ~data["season"].isin(holdout_seasons)
    champ = LogisticModel(C=0.5)
    champ.fit(data.loc[train_mask, feat_cols], data.loc[train_mask, "target"].to_numpy().astype(int))
    save_champion(champ, metrics={
        "mean_accuracy": summary["mean_accuracy"],
        "mean_log_loss": summary["mean_log_loss"],
        "mean_brier": summary["mean_brier"],
        "n_features": len(feat_cols),
        "model_family": "tournament_only_logistic",
        "holdout_seasons": [int(s) for s in holdout_seasons],
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
