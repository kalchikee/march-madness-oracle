"""Fetch the current NCAA bracket JSON and normalize it to bracket.json.

Run during the tournament window. If the NCAA endpoint returns nothing,
exits non-fatally so the workflow can skip gracefully.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from madness.config import current_season
from madness.ingest.ncaa_api import current_bracket_json
from madness.logging_setup import configure, get_logger

configure()
log = get_logger(__name__)


def normalize(raw: dict, season: int) -> dict:
    """Map NCAA's shape onto our canonical bracket schema."""
    if not raw:
        return {}
    return {
        "season": season,
        "revealed_at": raw.get("revealedAt"),
        "regions": raw.get("regions", ["East", "West", "South", "Midwest"]),
        "rounds": {
            "round_of_64": raw.get("roundOf64", []),
        },
        "first_four": raw.get("firstFour", []),
    }


def main() -> int:
    season = current_season()
    raw = current_bracket_json(season)
    if not raw:
        log.warning("bracket_unavailable", season=season)
        return 1
    normalized = normalize(raw, season)
    Path("bracket.json").write_text(json.dumps(normalized, indent=2))
    log.info("bracket_written", season=season)
    return 0


if __name__ == "__main__":
    sys.exit(main())
