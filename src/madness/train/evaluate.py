"""Metrics + bracket scoring.

Raw accuracy is NOT the metric the user cares about. What matters:
  1. Expected bracket points (ESPN scoring)
  2. Log loss by round (calibration)
  3. Accuracy on upset picks
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

# ESPN standard scoring: 10 / 20 / 40 / 80 / 160 / 320 per correct pick by round
ESPN_SCORING = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}


def round_breakdown(
    y_true: np.ndarray, y_prob: np.ndarray, rounds: np.ndarray
) -> pd.DataFrame:
    rows = []
    for r in sorted(set(rounds.tolist())):
        mask = rounds == r
        if mask.sum() == 0:
            continue
        y = y_true[mask]
        p = np.clip(y_prob[mask], 1e-6, 1 - 1e-6)
        preds = (p >= 0.5).astype(int)
        rows.append({
            "round": int(r),
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(y, preds)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(set(y)) == 2 else None,
            "mean_confidence": float(p.mean()),
        })
    return pd.DataFrame(rows)


def expected_bracket_score(
    y_true: np.ndarray, y_prob: np.ndarray, rounds: np.ndarray
) -> float:
    score = 0.0
    for yt, p, r in zip(y_true, y_prob, rounds):
        expected_correct = p if yt == 1 else (1 - p)
        score += expected_correct * ESPN_SCORING.get(int(r), 0)
    return float(score)


def upset_accuracy(
    y_true: np.ndarray, y_prob: np.ndarray, higher_seed_is_team_a: np.ndarray
) -> float:
    """Accuracy ONLY on games where a lower-seeded (underdog) team won."""
    upset = ((y_true == 1) & (~higher_seed_is_team_a)) | (
        (y_true == 0) & higher_seed_is_team_a
    )
    if upset.sum() == 0:
        return float("nan")
    preds = (y_prob >= 0.5).astype(int)
    return float(accuracy_score(y_true[upset], preds[upset]))
