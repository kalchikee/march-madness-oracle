"""Seed-only baseline — establishes the floor for all future work.

This model uses ONLY bracket-seed information (no ratings, no stats).
Its walk-forward accuracy is the "chalk+" baseline. Any real feature
set MUST beat this or it's not adding information.

Reports:
  - overall mean accuracy across walk-forward folds
  - per-round accuracy for the last N folds
  - upset-pick accuracy
  - well-calibrated log loss (log loss < 0.55 is calibrated)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from madness.config import PROCESSED_DIR
from madness.features.tournament import add_round_index
from madness.logging_setup import configure, get_logger
from madness.models.logistic import LogisticModel
from madness.models.registry import save_champion
from madness.train.backtest import walk_forward_backtest

configure()
log = get_logger(__name__)


def build_dataset() -> pd.DataFrame:
    path = PROCESSED_DIR / "tournament_games.parquet"
    df = pd.read_parquet(path)

    pos = df.rename(columns={
        "team_winner": "team_a", "team_loser": "team_b",
        "seed_winner": "seed_a", "seed_loser": "seed_b",
    }).copy()
    pos["target"] = 1
    neg = df.rename(columns={
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
    data["seed_ratio"] = data["seed_a"] / data["seed_b"]
    data["seed_diff_sq"] = data["seed_diff"] ** 2
    return data


def main() -> int:
    data = build_dataset()
    log.info("dataset_loaded", rows=len(data), seasons=int(data["season"].nunique()))

    feat_cols = [
        "seed_diff", "seed_sum", "seed_a_log", "seed_b_log",
        "seed_ratio", "round", "seed_diff_sq",
    ]
    result = walk_forward_backtest(
        data, "target", feat_cols, lambda: LogisticModel(C=0.5),
        min_train_seasons=10, holdout_last=3,
    )
    summary = result.summary()
    log.info("backtest_done", mean_accuracy=summary["mean_accuracy"],
             mean_log_loss=summary["mean_log_loss"])
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "per_season"},
        indent=2, default=float,
    ))

    # Per-round table aggregated across all folds
    rows = []
    for fold in result.folds:
        for rnd, s in fold.by_round.items():
            rows.append({"season": fold.validation_season, "round": rnd, **s})
    per_round = pd.DataFrame(rows)
    if not per_round.empty:
        print("\n=== Per-round walk-forward averages ===")
        agg = per_round.groupby("round").agg(
            mean_acc=("accuracy", "mean"),
            mean_ll=("log_loss", "mean"),
            total_n=("n", "sum"),
            folds=("season", "count"),
        ).reset_index()
        print(agg.to_string(index=False))
        per_round.to_csv(PROCESSED_DIR / "seed_baseline_per_round.csv", index=False)

    # Fit on all pre-holdout data and save as champion
    holdout_seasons = sorted(data["season"].unique())[-3:]
    train_mask = ~data["season"].isin(holdout_seasons)
    model = LogisticModel(C=0.5)
    model.fit(data.loc[train_mask, feat_cols], data.loc[train_mask, "target"].to_numpy().astype(int))
    save_champion(model, metrics={
        "mean_accuracy": summary["mean_accuracy"],
        "mean_log_loss": summary["mean_log_loss"],
        "mean_brier": summary["mean_brier"],
        "model_family": "seed_only_logistic",
        "training_rows": int(train_mask.sum()),
        "holdout_seasons": holdout_seasons,
    })
    log.info("champion_saved_as_seed_only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
