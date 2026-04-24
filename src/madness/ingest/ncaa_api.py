"""NCAA public JSON endpoints — modern-era scores and brackets.

The NCAA publishes bracket and score JSON at data.ncaa.com. Endpoints
change periodically, so we keep two strategies:
  1) try the historical scoreboard endpoint
  2) fall back to scraping ncaa.com/march-madness-live/bracket
"""
from __future__ import annotations

import json

import pandas as pd

from madness.ingest.http import fetch
from madness.logging_setup import get_logger

log = get_logger(__name__)

NAMESPACE = "ncaa_api"
BASE = "https://data.ncaa.com/casablanca"


def scoreboard_url(date_str: str) -> str:
    return f"{BASE}/scoreboard/basketball-men/d1/{date_str.replace('-', '/')}/scoreboard.json"


def ingest_scoreboard(date_str: str, force: bool = False) -> pd.DataFrame:
    url = scoreboard_url(date_str)
    try:
        body = fetch(url, namespace=NAMESPACE, force=force)
    except Exception as exc:
        log.warning("ncaa_scoreboard_miss", date=date_str, error=str(exc))
        return pd.DataFrame()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        log.warning("ncaa_bad_json", date=date_str)
        return pd.DataFrame()
    games = data.get("games", [])
    return pd.json_normalize(games)


def current_bracket_json(season: int, force: bool = False) -> dict:
    """Best-effort fetch of the Selection-Sunday bracket.

    Returns the raw NCAA-formatted dict; callers normalize it.
    """
    url = f"https://www.ncaa.com/march-madness-live/bracket"
    try:
        body = fetch(url, namespace=NAMESPACE, force=force)
    except Exception as exc:
        log.warning("bracket_page_fetch_failed", error=str(exc))
        return {}
    start = body.find("{\"bracket\"")
    if start < 0:
        return {}
    end = body.rfind("}") + 1
    try:
        return json.loads(body[start:end])
    except json.JSONDecodeError:
        log.warning("bracket_page_json_parse_failed")
        return {}
