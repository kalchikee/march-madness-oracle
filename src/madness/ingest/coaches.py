"""Scrape head-coach-per-team-per-season from Sports-Reference.

Source: /cbb/seasons/men/<YYYY>-coaches.html
Output: data/external/coaches.csv with columns season, team, coach_name

Coach data changes slowly (one coach per team per season), so this is
cheap to re-run but rarely needs to.
"""
from __future__ import annotations

import pandas as pd
from bs4 import BeautifulSoup

from madness.config import EXTERNAL_DIR
from madness.ingest.http import fetch
from madness.ingest.sports_reference import NAMESPACE as SR_NS
from madness.logging_setup import get_logger

log = get_logger(__name__)


def coaches_url(season: int) -> str:
    return f"https://www.sports-reference.com/cbb/seasons/men/{season}-coaches.html"


def parse_coaches_page(html: str, season: int) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="coaches") or soup.find("table")
    rows: list[dict] = []
    if table is None:
        return pd.DataFrame(rows)
    tbody = table.find("tbody")
    if tbody is None:
        return pd.DataFrame(rows)
    for tr in tbody.find_all("tr"):
        school_cell = (
            tr.find("td", {"data-stat": "school"})
            or tr.find("td", {"data-stat": "school_name"})
        )
        coach_cell = (
            tr.find("td", {"data-stat": "coach"})
            or tr.find("th", {"data-stat": "coach"})
        )
        if not (school_cell and coach_cell):
            continue
        team_name = school_cell.get_text(strip=True)
        coach_name = coach_cell.get_text(strip=True)
        if not team_name or not coach_name:
            continue
        rows.append({
            "season": season,
            "team": team_name,
            "coach_name": coach_name,
        })
    return pd.DataFrame(rows)


def ingest_coaches_range(start: int, end: int, force: bool = False) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in range(start, end + 1):
        try:
            html = fetch(coaches_url(season), namespace=SR_NS, force=force)
            frames.append(parse_coaches_page(html, season))
        except Exception as exc:
            log.error("coaches_failed", season=season, error=str(exc))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXTERNAL_DIR / "coaches.csv", index=False)
    log.info("coaches_written", rows=len(df))
    return df
