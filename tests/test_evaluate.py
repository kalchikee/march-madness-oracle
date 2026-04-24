"""Metric tests."""
from __future__ import annotations

import numpy as np

from madness.train.evaluate import (
    ESPN_SCORING,
    expected_bracket_score,
    round_breakdown,
)


def test_round_breakdown_basic():
    y = np.array([1, 0, 1, 1])
    p = np.array([0.8, 0.2, 0.6, 0.1])
    rounds = np.array([1, 1, 2, 2])
    df = round_breakdown(y, p, rounds)
    assert set(df["round"]) == {1, 2}


def test_expected_bracket_score_scales_with_round():
    y = np.array([1, 1])
    p = np.array([0.9, 0.9])
    s_r1 = expected_bracket_score(y, p, np.array([1, 1]))
    s_r6 = expected_bracket_score(y, p, np.array([6, 6]))
    assert s_r6 > s_r1
    assert ESPN_SCORING[6] > ESPN_SCORING[1]
