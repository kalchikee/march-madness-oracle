"""Consolidated tournament-results aggregator.

Cross-checks Sports-Reference against other sources (Wikipedia sanity
layer can be added later) and produces the canonical tournament game
table used by feature engineering and evaluation.
"""
from __future__ import annotations

import pandas as pd

from madness.config import INTERIM_DIR, PROCESSED_DIR
from madness.ingest import sports_reference as sr
from madness.logging_setup import get_logger
from madness.storage import write_parquet

log = get_logger(__name__)


def build_canonical_tournament_table(
    start_season: int, end_season: int, force: bool = False
) -> pd.DataFrame:
    df = sr.ingest_tournament_range(start_season, end_season, force=force)
    if df.empty:
        log.warning("tournament_empty")
        return df
    out = PROCESSED_DIR / "tournament_games.parquet"
    write_parquet(df, out)
    log.info("tournament_canonical_written", rows=len(df), path=str(out))
    return df


def load_tournament_table() -> pd.DataFrame:
    candidates = [
        PROCESSED_DIR / "tournament_games.parquet",
        INTERIM_DIR / "tournament_games.parquet",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p)
    return pd.DataFrame()
