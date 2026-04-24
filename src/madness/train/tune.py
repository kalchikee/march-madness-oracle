"""Optuna-driven hyperparameter search with walk-forward objective.

Optimize log loss, not accuracy. Log loss is a proper scoring rule so
tuning against it produces calibrated probabilities; accuracy alone can
be maximized by miscalibrated models that happen to land on the right
side of 0.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import optuna
import pandas as pd

from madness.config import OPTUNA_DB
from madness.logging_setup import get_logger
from madness.train.backtest import walk_forward_backtest

log = get_logger(__name__)


@dataclass
class TuneConfig:
    model_name: str
    n_trials: int = 100
    min_train_seasons: int = 10
    holdout_last: int = 3


def xgb_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
    }


def lgbm_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "objective": "binary",
        "metric": "binary_logloss",
        "random_state": 42,
        "verbosity": -1,
    }


def run_study(
    features: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cfg: TuneConfig,
    make_model_from_params: Callable[[dict], object],
    param_space: Callable[[optuna.Trial], dict],
) -> optuna.Study:
    study = optuna.create_study(
        direction="minimize",
        study_name=f"{cfg.model_name}",
        storage=f"sqlite:///{OPTUNA_DB}",
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = param_space(trial)
        result = walk_forward_backtest(
            features=features,
            target_col=target_col,
            feature_cols=feature_cols,
            model_factory=lambda: make_model_from_params(params),
            min_train_seasons=cfg.min_train_seasons,
            holdout_last=cfg.holdout_last,
        )
        if not result.folds:
            return float("inf")
        return result.mean_log_loss

    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)
    log.info("tune_done", best_value=study.best_value, best_params=study.best_params)
    return study
