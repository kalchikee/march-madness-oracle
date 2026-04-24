"""Model smoke tests: fit/predict_proba/save/load round-trips."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from madness.models.logistic import LogisticModel


def _synth_classification(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "f3": rng.normal(size=n),
    })
    logits = 1.2 * X["f1"] - 0.8 * X["f2"] + 0.05 * X["f3"]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logits))).astype(int).values
    return X, y


def test_logistic_fit_predict_beats_chance():
    X, y = _synth_classification()
    m = LogisticModel()
    m.fit(X, y)
    p = m.predict_proba(X)
    # On training data, a separable signal → well over 55% accuracy.
    assert ((p >= 0.5).astype(int) == y).mean() > 0.55


def test_logistic_save_load_roundtrip(tmp_path):
    X, y = _synth_classification()
    m = LogisticModel()
    m.fit(X, y)
    m.save(tmp_path)
    m2 = LogisticModel.load(tmp_path)
    assert np.allclose(m.predict_proba(X), m2.predict_proba(X))
