"""Stacking ensemble: logistic + XGBoost + LightGBM + CatBoost.

Uses the same leakage-free features as train_80pct_push.py.
Stacking typically adds 0.5-1.5 pp on walk-forward and reduces variance.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from madness.logging_setup import configure, get_logger
from madness.models.ensemble import StackingEnsemble
from madness.models.gbm import CatBoostModel, LGBMModel, XGBModel
from madness.models.logistic import LogisticModel
from madness.models.registry import save_champion
from madness.train.backtest import walk_forward_backtest

# Reuse build() from the 80% push script
sys.path.insert(0, "scripts")
from train_80pct_push import build  # noqa: E402

configure()
log = get_logger(__name__)


def main() -> int:
    data, feat_cols = build()
    prior_cols = [c for c in feat_cols if c.startswith("diff_prior_adj")]
    any_prior = data[prior_cols].notna().any(axis=1)
    data_t = data[any_prior & (data["season"] >= 2009)].copy()
    print(f"Rows: {len(data_t)}  Features: {len(feat_cols)}")

    base_models = [
        LogisticModel(C=0.5),
        XGBModel(params={
            "n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
            "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0,
            "objective": "binary:logistic", "eval_metric": "logloss",
            "tree_method": "hist", "random_state": 42,
        }),
        LGBMModel(params={
            "n_estimators": 300, "num_leaves": 16, "max_depth": 4,
            "learning_rate": 0.05, "min_child_samples": 10,
            "subsample": 0.85, "colsample_bytree": 0.85,
            "reg_lambda": 2.0, "objective": "binary",
            "metric": "binary_logloss", "random_state": 42, "verbosity": -1,
        }),
    ]

    def make_stack():
        return StackingEnsemble(base_models=base_models, n_folds=5)

    # Walk-forward
    r = walk_forward_backtest(
        data_t, "target", feat_cols, make_stack,
        min_train_seasons=4, holdout_last=3,
    )
    s = r.summary()
    print(f"\n=== STACKING ENSEMBLE walk-forward ===")
    print(json.dumps(
        {k: v for k, v in s.items() if k != "per_season"},
        indent=2, default=float,
    ))

    # Per-round
    rows = []
    for fold in r.folds:
        for rnd, stats in fold.by_round.items():
            rows.append({"season": fold.validation_season, "round": rnd, **stats})
    per_round = pd.DataFrame(rows)
    if not per_round.empty:
        agg = per_round.groupby("round").agg(
            mean_acc=("accuracy", "mean"),
            mean_ll=("log_loss", "mean"),
            total_n=("n", "sum"),
        ).reset_index()
        print("\nPer-round walk-forward:")
        print(agg.to_string(index=False))

    # TRUE holdout
    train = data_t[data_t["season"] < 2022]
    holdout = data_t[data_t["season"].isin([2022, 2023, 2024])].copy()
    champ = make_stack()
    champ.fit(train[feat_cols], train["target"].to_numpy().astype(int))
    holdout["prob"] = champ.predict_proba(holdout[feat_cols])
    wr = holdout[holdout["target"] == 1].copy()
    wr["correct"] = wr["prob"] >= 0.5
    holdout_acc = wr["correct"].mean()
    print(f"\n=== TRUE HOLDOUT (2022-2024) ===")
    print(f"Game-level accuracy: {holdout_acc*100:.2f}%")
    from madness.features.registry import ROUND_INDEX
    rev = {v: k for k, v in ROUND_INDEX.items()}
    for rnd in sorted(wr["round"].unique()):
        sub = wr[wr["round"] == rnd]
        print(f"  {rev.get(rnd, rnd)}: {sub['correct'].mean()*100:.1f}% (n={len(sub)})")

    save_champion(champ, metrics={
        "walk_forward_accuracy": s["mean_accuracy"],
        "walk_forward_log_loss": s["mean_log_loss"],
        "holdout_accuracy": float(holdout_acc),
        "model_family": "stacking_lr_xgb_lgbm",
        "n_features": len(feat_cols),
        "holdout_seasons": [2022, 2023, 2024],
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
