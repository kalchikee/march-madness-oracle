"""Uniform model interface: fit/predict_proba/save/load."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


class Model(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
    def save(self, directory: Path) -> None: ...

    @classmethod
    def load(cls, directory: Path) -> "Model": ...


@dataclass
class ModelMetadata:
    name: str
    trained_at: str
    feature_names: list[str]
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    feature_hash: str = ""
    version: str = "0.1.0"
