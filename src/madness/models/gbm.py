"""Gradient-boosting wrappers: XGBoost, LightGBM, CatBoost.

These are the workhorses for tabular problems like tournament prediction.
All three expose the same fit/predict_proba interface.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from madness.models.base import ModelMetadata


class XGBModel:
    name = "xgboost"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {
            "n_estimators": 600,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "min_child_weight": 3,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": 42,
        }
        self.model = None
        self.feature_names: list[str] = []
        self.metadata: ModelMetadata | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        import xgboost as xgb
        self.feature_names = list(X.columns)
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y)
        self.metadata = ModelMetadata(
            name=self.name,
            trained_at=datetime.utcnow().isoformat(),
            feature_names=self.feature_names,
            params=dict(self.params),
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        return self.model.predict_proba(X[self.feature_names])[:, 1]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory / "model.joblib")
        if self.metadata is not None:
            (directory / "metadata.json").write_text(
                json.dumps(asdict(self.metadata), indent=2)
            )

    @classmethod
    def load(cls, directory: Path) -> "XGBModel":
        m = cls()
        m.model = joblib.load(directory / "model.joblib")
        meta = directory / "metadata.json"
        if meta.exists():
            d = json.loads(meta.read_text())
            m.feature_names = d["feature_names"]
            m.metadata = ModelMetadata(**d)
        return m


class LGBMModel:
    name = "lightgbm"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {
            "n_estimators": 800,
            "max_depth": -1,
            "num_leaves": 48,
            "learning_rate": 0.04,
            "min_child_samples": 20,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "objective": "binary",
            "metric": "binary_logloss",
            "random_state": 42,
            "verbosity": -1,
        }
        self.model = None
        self.feature_names: list[str] = []
        self.metadata: ModelMetadata | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        import lightgbm as lgb
        self.feature_names = list(X.columns)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)
        self.metadata = ModelMetadata(
            name=self.name,
            trained_at=datetime.utcnow().isoformat(),
            feature_names=self.feature_names,
            params=dict(self.params),
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        return self.model.predict_proba(X[self.feature_names])[:, 1]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory / "model.joblib")
        if self.metadata is not None:
            (directory / "metadata.json").write_text(
                json.dumps(asdict(self.metadata), indent=2)
            )

    @classmethod
    def load(cls, directory: Path) -> "LGBMModel":
        m = cls()
        m.model = joblib.load(directory / "model.joblib")
        meta = directory / "metadata.json"
        if meta.exists():
            d = json.loads(meta.read_text())
            m.feature_names = d["feature_names"]
            m.metadata = ModelMetadata(**d)
        return m


class CatBoostModel:
    name = "catboost"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {
            "iterations": 800,
            "depth": 6,
            "learning_rate": 0.04,
            "l2_leaf_reg": 3.0,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": 42,
            "verbose": False,
        }
        self.model = None
        self.feature_names: list[str] = []
        self.metadata: ModelMetadata | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        from catboost import CatBoostClassifier
        self.feature_names = list(X.columns)
        self.model = CatBoostClassifier(**self.params)
        self.model.fit(X, y)
        self.metadata = ModelMetadata(
            name=self.name,
            trained_at=datetime.utcnow().isoformat(),
            feature_names=self.feature_names,
            params=dict(self.params),
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        return self.model.predict_proba(X[self.feature_names])[:, 1]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(directory / "model.cbm"))
        if self.metadata is not None:
            (directory / "metadata.json").write_text(
                json.dumps(asdict(self.metadata), indent=2)
            )

    @classmethod
    def load(cls, directory: Path) -> "CatBoostModel":
        from catboost import CatBoostClassifier
        m = cls()
        m.model = CatBoostClassifier()
        m.model.load_model(str(directory / "model.cbm"))
        meta = directory / "metadata.json"
        if meta.exists():
            d = json.loads(meta.read_text())
            m.feature_names = d["feature_names"]
            m.metadata = ModelMetadata(**d)
        return m
