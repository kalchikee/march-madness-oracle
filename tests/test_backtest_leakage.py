"""The most important test in the repo: prove the backtest cannot leak.

If this test passes, a model trained on tournaments up to year T cannot
see any data from year T+1 or later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from madness.train.backtest import walk_forward_backtest


class _LeakDetector:
    """Dummy model that records the max season it sees during fit."""

    def __init__(self) -> None:
        self.name = "leak_detector"
        self.max_train_season = -1
        self.metadata = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.max_train_season = int(X["_season"].max())

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 0.5)


def _make_fake_features(n_seasons: int = 15, games_per: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for s in range(2000, 2000 + n_seasons):
        for _ in range(games_per):
            rows.append({
                "season": s,
                "_season": s,
                "f1": rng.normal(),
                "target": rng.integers(0, 2),
                "round": rng.integers(1, 7),
            })
    return pd.DataFrame(rows)


def test_walk_forward_no_future_leakage():
    df = _make_fake_features()
    seen = []

    def factory():
        m = _LeakDetector()
        seen.append(m)
        return m

    walk_forward_backtest(
        df, "target", ["f1", "_season"], factory,
        min_train_seasons=5, holdout_last=3,
    )

    seasons_sorted = sorted(df["season"].unique())
    val_seasons = seasons_sorted[5:-3]
    for m, val_season in zip(seen, val_seasons):
        assert m.max_train_season < val_season, (
            f"Leak: trained max={m.max_train_season}, validating {val_season}"
        )


def test_walk_forward_respects_holdout():
    df = _make_fake_features()
    result = walk_forward_backtest(
        df, "target", ["f1", "_season"], _LeakDetector,
        min_train_seasons=5, holdout_last=3,
    )
    max_val = max(f.validation_season for f in result.folds)
    assert max_val < 2000 + 15 - 3


def test_walk_forward_fold_count():
    df = _make_fake_features(n_seasons=12)
    result = walk_forward_backtest(
        df, "target", ["f1", "_season"], _LeakDetector,
        min_train_seasons=4, holdout_last=2,
    )
    # seasons: 2000..2011. holdout removes 2010-2011, min_train=4 skips 2000-2003.
    # Validation folds: 2004..2009 → 6 folds.
    assert len(result.folds) == 6
