"""Bart Torvik ingestion (barttorvik.com).

Free alternative to KenPom; covers 2008+. Has unofficial JSON endpoints
that return per-team efficiency metrics.
"""
from __future__ import annotations

import json

import pandas as pd

from madness.config import INTERIM_DIR
from madness.ingest.http import fetch
from madness.logging_setup import get_logger
from madness.storage import write_parquet

log = get_logger(__name__)

NAMESPACE = "torvik"
BASE = "https://barttorvik.com"


def team_year_url(season: int) -> str:
    """Torvik's `<YYYY>_team_results.json` endpoint returns an array of
    42-element arrays per team, unauthenticated and not bot-gated.
    The trank.php endpoint is browser-JS-gated and does not work for us.
    """
    return f"{BASE}/{season}_team_results.json"


# Schema of Torvik team_results.json rows.
# Reconstructed from barttorvik's public JSON; column order is stable.
_TORVIK_COLS = [
    "rank", "team", "conf", "record",
    "adj_oe", "adj_de", "barthag", "eff_fg_pct", "eff_fg_pct_def",
    "to_pct", "to_pct_def", "orb_pct", "orb_pct_def",
    "ft_rate", "ft_rate_def", "two_pct", "two_pct_def",
    "three_pct", "three_pct_def", "blk_pct", "blk_pct_def",
    "three_rate", "three_rate_def", "adj_tempo", "wab",
    "wins", "losses", "games",
    "off_rank", "def_rank", "tempo_rank",
    "seed", "conf_wins", "conf_losses",
    "col_34", "col_35", "col_36", "col_37", "col_38", "col_39",
    "col_40", "col_41",
]


def ingest_season(season: int, force: bool = False) -> pd.DataFrame:
    url = team_year_url(season)
    body = fetch(url, namespace=NAMESPACE, force=force)
    parsed = json.loads(body)
    rows = []
    for entry in parsed:
        if not isinstance(entry, list):
            continue
        # Pad/truncate to schema length
        padded = list(entry) + [None] * (len(_TORVIK_COLS) - len(entry))
        padded = padded[: len(_TORVIK_COLS)]
        row = dict(zip(_TORVIK_COLS, padded))
        row["season"] = season
        rows.append(row)
    df = pd.DataFrame(rows)
    out = INTERIM_DIR / "torvik" / f"season_{season}.parquet"
    write_parquet(df, out)
    log.info("torvik_written", season=season, rows=len(df), path=str(out))
    return df


def ingest_range(start: int, end: int, force: bool = False) -> pd.DataFrame:
    frames = []
    for season in range(start, end + 1):
        try:
            frames.append(ingest_season(season, force=force))
        except Exception as exc:
            log.error("torvik_failed", season=season, error=str(exc))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    out = INTERIM_DIR / "torvik_all.parquet"
    write_parquet(df, out)
    return df
