"""Logistic regression baseline — interpretable, fast, strong floor."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from madness.models.base import ModelMetadata


class LogisticModel:
    name = "logistic_l2"

    def __init__(self, C: float = 1.0, max_iter: int = 2000) -> None:
        self.C = C
        self.max_iter = max_iter
        self.pipeline: Pipeline | None = None
        self.feature_names: list[str] = []
        self.metadata: ModelMetadata | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.feature_names = list(X.columns)
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=self.C, max_iter=self.max_iter, solver="lbfgs",
                n_jobs=None,
            )),
        ])
        self.pipeline.fit(X.values, y)
        self.metadata = ModelMetadata(
            name=self.name,
            trained_at=datetime.utcnow().isoformat(),
            feature_names=self.feature_names,
            params={"C": self.C, "max_iter": self.max_iter},
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.pipeline is not None
        X = X[self.feature_names]
        return self.pipeline.predict_proba(X.values)[:, 1]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, directory / "model.joblib")
        if self.metadata is not None:
            (directory / "metadata.json").write_text(
                json.dumps(asdict(self.metadata), indent=2)
            )

    @classmethod
    def load(cls, directory: Path) -> "LogisticModel":
        m = cls()
        m.pipeline = joblib.load(directory / "model.joblib")
        meta_path = directory / "metadata.json"
        if meta_path.exists():
            d = json.loads(meta_path.read_text())
            m.metadata = ModelMetadata(**d)
            m.feature_names = d["feature_names"]
        return m
