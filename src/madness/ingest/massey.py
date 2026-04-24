"""Massey Ratings ingestion — composite of many rating systems, 1995+."""
from __future__ import annotations

import pandas as pd

from madness.ingest.http import fetch
from madness.logging_setup import get_logger

log = get_logger(__name__)

NAMESPACE = "massey"
BASE = "https://masseyratings.com"


def ingest_season(season: int, force: bool = False) -> pd.DataFrame:
    """Stub — Massey's pages require parsing a text table.

    Will be fleshed out once the core pipeline is proven. Returns empty.
    """
    _ = season, force
    log.info("massey_stub_noop")
    return pd.DataFrame()
