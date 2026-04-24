"""Writes March Madness predictions to predictions/YYYY-MM-DD.json.

The kalshi-safety service fetches this file via GitHub raw URL to
decide which picks to back on Kalshi. This module only emits the
JSON — it does not place any bets.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from madness.predict.bracket import Prediction

# Repo root: .../March Madness/src/madness/predictions_file.py -> up 3 dirs
REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = REPO_ROOT / "predictions"

MIN_PROB = float(os.environ.get("KALSHI_MIN_PROB", "0.58"))


def write_predictions_file(
    date: str | None,
    predictions: Iterable[Prediction],
) -> str:
    """Write predictions/<date>.json in the kalshi-safety schema.

    `date` may be ``None`` — defaults to today (UTC).
    """
    iso_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{iso_date}.json"

    picks: list[dict] = []
    for p in predictions:
        prob_a = float(p.prob_a_wins)
        prob_b = 1.0 - prob_a
        favored_a = prob_a >= prob_b
        model_prob = max(prob_a, prob_b)
        if model_prob < MIN_PROB:
            continue
        # March Madness matchups are tournament games on a neutral court;
        # we still need a home/away slot for the shape, so we arbitrarily
        # treat team_a as home and team_b as away.
        home = p.team_a
        away = p.team_b
        picks.append({
            "gameId": f"mm-{iso_date}-{away}-{home}".replace(" ", "_"),
            "home": home,
            "away": away,
            "pickedTeam": p.pick,
            "pickedSide": "home" if favored_a else "away",
            "modelProb": round(model_prob, 4),
            "confidenceTier": p.confidence_tier,
            "extra": {
                "region": p.region,
                "round": p.round_name,
                "seedA": p.seed_a,
                "seedB": p.seed_b,
                "probAWins": round(prob_a, 4),
            },
        })

    payload = {
        "sport": "MARCH_MADNESS",
        "date": iso_date,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "picks": picks,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return str(out_path)
