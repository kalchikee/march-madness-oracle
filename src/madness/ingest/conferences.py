"""Scrape season-by-season conference membership from Sports-Reference.

Source: /cbb/seasons/men/<YYYY>-standings.html
Output: data/external/conferences.csv with columns season, team, conference

Teams switch conferences often in the realignment era; this must be
captured per season, not once.
"""
from __future__ import annotations

import pandas as pd
from bs4 import BeautifulSoup

from madness.config import EXTERNAL_DIR
from madness.ingest.http import fetch
from madness.ingest.sports_reference import _parse_comments
from madness.ingest.sports_reference import NAMESPACE as SR_NS
from madness.logging_setup import get_logger

log = get_logger(__name__)


def standings_url(season: int) -> str:
    return f"https://www.sports-reference.com/cbb/seasons/men/{season}-standings.html"


def parse_standings(html: str, season: int) -> pd.DataFrame:
    """Each conference has its own table; iterate and concat."""
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []
    tables = soup.find_all("table") + [
        t for sub in _parse_comments(soup) for t in sub.find_all("table")
    ]
    for table in tables:
        caption = table.find("caption")
        if caption is None:
            continue
        conference = caption.get_text(strip=True)
        if not conference:
            continue
        tbody = table.find("tbody")
        if tbody is None:
            continue
        for tr in tbody.find_all("tr"):
            school = tr.find(["td", "th"], {"data-stat": "school_name"})
            if school is None:
                continue
            rows.append({
                "season": season,
                "team": school.get_text(strip=True),
                "conference": conference,
            })
    return pd.DataFrame(rows)


def ingest_conferences_range(start: int, end: int, force: bool = False) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in range(start, end + 1):
        try:
            html = fetch(standings_url(season), namespace=SR_NS, force=force)
            frames.append(parse_standings(html, season))
        except Exception as exc:
            log.error("conferences_failed", season=season, error=str(exc))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXTERNAL_DIR / "conferences.csv", index=False)
    log.info("conferences_written", rows=len(df))
    return df
