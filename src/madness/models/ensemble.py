"""Stacking ensemble: logistic meta-learner over base models.

Out-of-fold predictions from each base model become features for the
meta-learner. This is typically the production model because it tends
to beat any single base model by 0.5–1.5 pp on log loss.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from madness.models.base import ModelMetadata


class StackingEnsemble:
    name = "stacking_ensemble"

    def __init__(self, base_models: list, n_folds: int = 5) -> None:
        self.base_models = base_models
        self.n_folds = n_folds
        self.fitted_bases: list = []
        self.meta: LogisticRegression | None = None
        self.feature_names: list[str] = []
        self.metadata: ModelMetadata | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.feature_names = list(X.columns)
        n = len(X)
        oof = np.zeros((n, len(self.base_models)))
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        for i, model_proto in enumerate(self.base_models):
            for tr, va in kf.split(X):
                import copy
                m = copy.deepcopy(model_proto)
                m.fit(X.iloc[tr], y[tr])
                oof[va, i] = m.predict_proba(X.iloc[va])

        self.meta = LogisticRegression(max_iter=2000, C=1.0)
        self.meta.fit(oof, y)

        # Refit each base on full data for inference
        self.fitted_bases = []
        import copy
        for model_proto in self.base_models:
            m = copy.deepcopy(model_proto)
            m.fit(X, y)
            self.fitted_bases.append(m)

        self.metadata = ModelMetadata(
            name=self.name,
            trained_at=datetime.utcnow().isoformat(),
            feature_names=self.feature_names,
            params={"base_models": [m.name for m in self.base_models]},
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.meta is not None
        base_probs = np.column_stack([
            m.predict_proba(X) for m in self.fitted_bases
        ])
        return self.meta.predict_proba(base_probs)[:, 1]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.fitted_bases):
            m.save(directory / f"base_{i}_{m.name}")
        joblib.dump(self.meta, directory / "meta.joblib")
        if self.metadata is not None:
            (directory / "metadata.json").write_text(
                json.dumps(asdict(self.metadata), indent=2)
            )

    @classmethod
    def load(cls, directory: Path) -> "StackingEnsemble":
        raise NotImplementedError(
            "Stacking ensemble load requires knowing base model types; "
            "use models.registry.load_champion()."
        )
