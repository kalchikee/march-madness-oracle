"""KenPom ingestion — the single most predictive public signal.

Requires paid subscription. Credentials from env:
    KENPOM_USER, KENPOM_PASS

If credentials are missing, all functions no-op and return empty frames
so the broader pipeline can proceed on Torvik + Sports-Reference alone.
"""
from __future__ import annotations

import pandas as pd
import requests
from bs4 import BeautifulSoup

from madness.config import DEFAULT_USER_AGENT, INTERIM_DIR, Secrets
from madness.logging_setup import get_logger
from madness.storage import write_parquet

log = get_logger(__name__)

NAMESPACE = "kenpom"
LOGIN_URL = "https://kenpom.com/handlers/login_handler.php"
RATINGS_URL = "https://kenpom.com/index.php"


def _login(user: str, password: str) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = DEFAULT_USER_AGENT
    s.get("https://kenpom.com/", timeout=30)
    resp = s.post(
        LOGIN_URL,
        data={"email": user, "password": password},
        timeout=30,
        allow_redirects=True,
    )
    resp.raise_for_status()
    if "logout" not in resp.text.lower():
        raise RuntimeError("KenPom login did not appear to succeed")
    return s


def ingest_season(season: int) -> pd.DataFrame:
    secrets = Secrets.from_env()
    if not secrets.has_kenpom:
        log.warning("kenpom_no_credentials_skip")
        return pd.DataFrame()
    assert secrets.kenpom_user and secrets.kenpom_pass
    session = _login(secrets.kenpom_user, secrets.kenpom_pass)
    r = session.get(RATINGS_URL, params={"y": str(season)}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table", id="ratings-table")
    if table is None:
        log.error("kenpom_table_not_found", season=season)
        return pd.DataFrame()
    rows = []
    for tr in table.select("tbody tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 5:
            continue
        rows.append(cells)
    df = pd.DataFrame(rows)
    df["season"] = season
    out = INTERIM_DIR / "kenpom" / f"season_{season}.parquet"
    write_parquet(df, out)
    log.info("kenpom_written", season=season, rows=len(df))
    return df


def ingest_range(start: int, end: int) -> pd.DataFrame:
    frames = [ingest_season(s) for s in range(start, end + 1)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
