"""Walk-forward backtest by tournament year.

THIS IS THE MOST IMPORTANT FILE IN THE REPO.

Never use random k-fold. The tournament has clear temporal structure
and any random split leaks future data (e.g., "a team had 3 R64 games
in 2023 and the model saw 2 of them at train time").

Walk-forward:
    train on tournaments [1985..T-1], validate on T
    step T forward one year at a time
Holdout:
    last 3 tournaments are NEVER looked at until champion promotion time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from madness.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class FoldResult:
    validation_season: int
    n_games: int
    accuracy: float
    log_loss: float
    brier: float
    by_round: dict[int, dict[str, float]] = field(default_factory=dict)


@dataclass
class BacktestResult:
    folds: list[FoldResult]

    @property
    def mean_accuracy(self) -> float:
        if not self.folds:
            return 0.0
        return float(np.mean([f.accuracy for f in self.folds]))

    @property
    def mean_log_loss(self) -> float:
        if not self.folds:
            return 0.0
        return float(np.mean([f.log_loss for f in self.folds]))

    @property
    def mean_brier(self) -> float:
        if not self.folds:
            return 0.0
        return float(np.mean([f.brier for f in self.folds]))

    def summary(self) -> dict:
        return {
            "n_folds": len(self.folds),
            "mean_accuracy": self.mean_accuracy,
            "mean_log_loss": self.mean_log_loss,
            "mean_brier": self.mean_brier,
            "per_season": [
                {
                    "season": f.validation_season,
                    "accuracy": f.accuracy,
                    "log_loss": f.log_loss,
                    "n_games": f.n_games,
                }
                for f in self.folds
            ],
        }


def walk_forward_backtest(
    features: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    model_factory: Callable[[], object],
    min_train_seasons: int = 10,
    holdout_last: int = 3,
    round_col: str = "round",
) -> BacktestResult:
    """Run walk-forward validation, excluding holdout seasons.

    `model_factory` returns a fresh model each fold (so state doesn't leak).
    """
    if features.empty:
        return BacktestResult(folds=[])

    seasons = sorted(features["season"].unique())
    if holdout_last > 0:
        seasons = seasons[:-holdout_last]
    if len(seasons) <= min_train_seasons:
        log.warning(
            "not_enough_seasons",
            seasons=len(seasons),
            min_required=min_train_seasons,
        )
        return BacktestResult(folds=[])

    folds: list[FoldResult] = []
    for i, val_season in enumerate(seasons[min_train_seasons:], start=min_train_seasons):
        train_seasons = seasons[:i]
        train = features[features["season"].isin(train_seasons)]
        val = features[features["season"] == val_season]
        if val.empty or train.empty:
            continue

        X_train = train[feature_cols].copy()
        y_train = train[target_col].to_numpy().astype(int)
        X_val = val[feature_cols].copy()
        y_val = val[target_col].to_numpy().astype(int)

        model = model_factory()
        model.fit(X_train, y_train)
        p = model.predict_proba(X_val)
        preds = (p >= 0.5).astype(int)

        fold = FoldResult(
            validation_season=int(val_season),
            n_games=int(len(val)),
            accuracy=float(accuracy_score(y_val, preds)),
            log_loss=float(log_loss(y_val, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
            brier=float(brier_score_loss(y_val, p)),
        )

        if round_col in val.columns:
            for rnd, sub in val.groupby(round_col):
                idx = sub.index
                mask = val.index.isin(idx)
                if mask.sum() == 0:
                    continue
                yv = y_val[mask]
                pv = p[mask]
                fold.by_round[int(rnd)] = {
                    "accuracy": float(accuracy_score(yv, (pv >= 0.5).astype(int))),
                    "log_loss": float(
                        log_loss(yv, np.clip(pv, 1e-6, 1 - 1e-6), labels=[0, 1])
                    ),
                    "n": int(mask.sum()),
                }

        folds.append(fold)
        log.info(
            "fold_complete",
            season=val_season,
            accuracy=fold.accuracy,
            log_loss=fold.log_loss,
        )

    return BacktestResult(folds=folds)
