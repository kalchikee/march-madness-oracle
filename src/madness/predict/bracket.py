"""Generate full-bracket predictions from the champion model.

Input: normalized bracket JSON + feature table for the current season.
Output: list of Prediction records, one per R64 matchup plus simulated
later-round probabilities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from madness.logging_setup import get_logger
from madness.models.registry import load_champion

log = get_logger(__name__)


@dataclass
class Prediction:
    region: str
    round_name: str
    seed_a: int
    team_a: str
    seed_b: int
    team_b: str
    prob_a_wins: float
    pick: str
    confidence_tier: str


def confidence_tier(prob: float) -> str:
    """Map a predicted probability to a display tier."""
    p = max(prob, 1 - prob)
    if p >= 0.80:
        return "lock"
    if p >= 0.62:
        return "likely"
    if p >= 0.55:
        return "lean"
    return "tossup"


def build_predictions(
    bracket: dict[str, Any],
    features: pd.DataFrame,
    feature_cols: list[str],
) -> list[Prediction]:
    """Predict every R64 matchup from the bracket JSON."""
    model = load_champion()
    r64 = bracket.get("rounds", {}).get("round_of_64", [])
    rows = []
    for m in r64:
        team_a = m["team_a"]
        team_b = m["team_b"]
        feat = features[
            (features["team_a"] == team_a) & (features["team_b"] == team_b)
        ]
        if feat.empty:
            log.warning("missing_matchup_features", team_a=team_a, team_b=team_b)
            continue
        x = feat[feature_cols].iloc[:1]
        p = float(model.predict_proba(x)[0])
        rows.append(Prediction(
            region=m.get("region", ""),
            round_name="Round of 64",
            seed_a=int(m["seed_a"]),
            team_a=team_a,
            seed_b=int(m["seed_b"]),
            team_b=team_b,
            prob_a_wins=p,
            pick=team_a if p >= 0.5 else team_b,
            confidence_tier=confidence_tier(p),
        ))
    return rows


def predictions_to_frame(preds: list[Prediction]) -> pd.DataFrame:
    return pd.DataFrame([asdict(p) for p in preds])


def save_predictions(preds: list[Prediction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions_to_frame(preds).to_csv(path, index=False)
