"""Sports-Reference College Basketball scraper.

Covers: tournament results, per-season team stats, per-game results.
Range: 1975+ for results, ~1997+ for full advanced stats.

Respectful scraping:
- 3 seconds min between requests (enforced in http.fetch)
- Every page cached to disk under data/raw/sports_reference/
- Re-runs never re-hit the site unless --force is passed
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Comment

from madness.config import INTERIM_DIR
from madness.ingest.http import fetch
from madness.logging_setup import get_logger
from madness.storage import write_parquet

log = get_logger(__name__)

NAMESPACE = "sports_reference"
BASE = "https://www.sports-reference.com/cbb"


@dataclass
class TournamentGame:
    season: int
    round_name: str
    region: str | None
    seed_winner: int | None
    seed_loser: int | None
    team_winner: str
    team_loser: str
    score_winner: int | None
    score_loser: int | None
    overtime: bool
    date: str | None


def tournament_url(season: int) -> str:
    return f"{BASE}/postseason/men/{season}-ncaa.html"


def school_stats_url(season: int) -> str:
    """Per-team season aggregates (W, L, SRS, SOS, etc.) — one page per season."""
    return f"{BASE}/seasons/men/{season}-school-stats.html"


def conference_schedule_url(conf_slug: str, season: int) -> str:
    """Per-conference full schedule — lists every game in that conference."""
    return f"{BASE}/conferences/{conf_slug}/men/{season}-schedule.html"


def _parse_comments(soup: BeautifulSoup) -> list[BeautifulSoup]:
    """Sports-Reference hides many tables inside HTML comments."""
    out: list[BeautifulSoup] = []
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            out.append(BeautifulSoup(c, "lxml"))
    return out


_REGION_ROUND_NAMES = [
    "Round of 64", "Round of 32", "Sweet Sixteen", "Elite Eight", "Regional Final"
]
_NATIONAL_ROUND_NAMES = [
    "Final Four", "Championship"
]


def _parse_bracket_region(region_div, region_name: str, season: int) -> list[TournamentGame]:
    """Parse a single region's bracket (#east, #midwest, etc., or #national)."""
    games: list[TournamentGame] = []
    bracket = region_div.find("div", id="bracket")
    if bracket is None:
        return games

    round_names = _NATIONAL_ROUND_NAMES if region_name == "national" else _REGION_ROUND_NAMES

    rounds = bracket.find_all("div", class_="round", recursive=False)
    for idx, round_div in enumerate(rounds):
        round_name = round_names[idx] if idx < len(round_names) else f"Round {idx+1}"
        game_divs = [c for c in round_div.find_all("div", recursive=False)
                     if c.find("a") is not None]
        for game_div in game_divs:
            game = _parse_game_div(game_div, season, round_name, region_name)
            if game is not None:
                games.append(game)
    return games


def _parse_game_div(game_div, season: int, round_name: str, region: str) -> TournamentGame | None:
    team_divs = [c for c in game_div.find_all("div", recursive=False)
                 if c.find("a") is not None]
    if len(team_divs) < 2:
        return None
    parsed: list[tuple[int | None, str, int | None]] = []
    for td in team_divs[:2]:
        seed_span = td.find("span")
        team_link = td.find("a", href=lambda h: h and "/schools/" in h)
        score_link = td.find("a", href=lambda h: h and "/boxscores/" in h)
        seed = _safe_int(seed_span.get_text(strip=True)) if seed_span else None
        team = team_link.get_text(strip=True) if team_link else ""
        score = _safe_int(score_link.get_text(strip=True)) if score_link else None
        if not team:
            return None
        parsed.append((seed, team, score))

    (sa, ta, pa), (sb, tb, pb) = parsed[0], parsed[1]
    # Determine winner via class="winner"
    winner_idx = None
    for i, td in enumerate(team_divs[:2]):
        if td.get("class") and "winner" in td.get("class"):
            winner_idx = i
            break
    if winner_idx is None and pa is not None and pb is not None:
        winner_idx = 0 if pa > pb else 1
    if winner_idx is None:
        return None

    if winner_idx == 0:
        seed_w, team_w, score_w = sa, ta, pa
        seed_l, team_l, score_l = sb, tb, pb
    else:
        seed_w, team_w, score_w = sb, tb, pb
        seed_l, team_l, score_l = sa, ta, pa

    # Venue / date from the trailing span link
    date_str = None
    venue_span = game_div.find_all("span")[-1] if game_div.find_all("span") else None
    if venue_span is not None:
        a = venue_span.find("a", href=lambda h: h and "/boxscores/" in h)
        if a:
            href = a.get("href", "")
            m = re.search(r"(\d{4}-\d{2}-\d{2})", href)
            if m:
                date_str = m.group(1)

    return TournamentGame(
        season=season,
        round_name=round_name,
        region=region if region != "national" else None,
        seed_winner=seed_w,
        seed_loser=seed_l,
        team_winner=team_w,
        team_loser=team_l,
        score_winner=score_w,
        score_loser=score_l,
        overtime=False,  # SR does not expose OT flag in the bracket markup
        date=date_str,
    )


def _safe_int(s: str) -> int | None:
    try:
        return int(s.strip())
    except (ValueError, AttributeError):
        return None


def parse_tournament_page(html: str, season: int) -> list[TournamentGame]:
    """Parse a season's postseason page into game rows."""
    soup = BeautifulSoup(html, "lxml")
    games: list[TournamentGame] = []

    brackets = soup.find("div", id="brackets")
    if brackets is None:
        return games

    for region_div in brackets.find_all("div", recursive=False):
        region_id = region_div.get("id", "")
        if not region_id:
            continue
        games.extend(_parse_bracket_region(region_div, region_id, season))
    return games


def ingest_tournament_season(season: int, force: bool = False) -> list[TournamentGame]:
    url = tournament_url(season)
    html = fetch(url, namespace=NAMESPACE, force=force)
    games = parse_tournament_page(html, season)
    log.info("sr_tournament_parsed", season=season, games=len(games))
    return games


def ingest_tournament_range(start: int, end: int, force: bool = False) -> pd.DataFrame:
    all_rows: list[dict] = []
    for season in range(start, end + 1):
        try:
            games = ingest_tournament_season(season, force=force)
        except Exception as exc:
            log.error("sr_tournament_failed", season=season, error=str(exc))
            continue
        all_rows.extend(asdict(g) for g in games)
    df = pd.DataFrame(all_rows)
    out = INTERIM_DIR / "tournament_games.parquet"
    write_parquet(df, out)
    log.info("sr_tournament_written", rows=len(df), path=str(out))
    return df


# ---- Regular-season per-game results ----

def school_schedule_url(slug: str, season: int) -> str:
    return f"{BASE}/schools/{slug}/men/{season}-schedule.html"


@dataclass
class RegularSeasonGame:
    season: int
    date: str
    team_winner: str
    team_loser: str
    score_winner: int
    score_loser: int
    site_neutral: bool
    site_home: str  # team name that was home, or "" for neutral
    overtime: bool


_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _cell_text(cell) -> str:
    if cell is None:
        return ""
    return cell.get_text(" ", strip=True)


def _parse_schedule_table(table, season: int) -> list[RegularSeasonGame]:
    games: list[RegularSeasonGame] = []
    tbody = table.find("tbody")
    if tbody is None:
        return games
    for tr in tbody.find_all("tr"):
        if tr.get("class") and "thead" in tr.get("class"):
            continue
        date_cell = tr.find(["th", "td"], {"data-stat": "date_game"})
        if date_cell is None:
            # Older eras may not use data-stat — fall back to first cell with a date
            first = tr.find(["th", "td"])
            date_str = _cell_text(first)
            if not _DATE_RE.search(date_str):
                continue
        else:
            date_str = _cell_text(date_cell)

        m = _DATE_RE.search(date_str)
        if not m:
            continue
        date = m.group(0)

        winner_cell = tr.find("td", {"data-stat": "winner_school_name"}) or \
            tr.find("td", {"data-stat": "winner"})
        loser_cell = tr.find("td", {"data-stat": "loser_school_name"}) or \
            tr.find("td", {"data-stat": "loser"})
        win_pts = tr.find("td", {"data-stat": "winner_pts"})
        lose_pts = tr.find("td", {"data-stat": "loser_pts"})
        at_cell = tr.find("td", {"data-stat": "game_location"})
        ot_cell = tr.find("td", {"data-stat": "overtimes"})

        if not (winner_cell and loser_cell and win_pts and lose_pts):
            continue

        try:
            wp = int(_cell_text(win_pts))
            lp = int(_cell_text(lose_pts))
        except ValueError:
            continue

        location_marker = _cell_text(at_cell)
        is_neutral = location_marker == "N"
        # game_location "" means winner was home; "@" means loser was home
        home_team = ""
        if not is_neutral:
            if location_marker == "@":
                home_team = _cell_text(loser_cell)
            else:
                home_team = _cell_text(winner_cell)

        ot_text = _cell_text(ot_cell)
        overtime = bool(ot_text) and ot_text.lower() != ""

        games.append(RegularSeasonGame(
            season=season,
            date=date,
            team_winner=_cell_text(winner_cell),
            team_loser=_cell_text(loser_cell),
            score_winner=wp,
            score_loser=lp,
            site_neutral=is_neutral,
            site_home=home_team,
            overtime=overtime,
        ))
    return games


def parse_schedule_page(html: str, season: int) -> list[RegularSeasonGame]:
    """Sports-Reference season schedule. The main #schedule table may be
    hidden in HTML comments; we check both locations."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="schedule")
    if table is not None:
        games = _parse_schedule_table(table, season)
        if games:
            return games
    # Fallback: comment-hidden tables
    for sub in _parse_comments(soup):
        t = sub.find("table", id="schedule")
        if t is not None:
            games = _parse_schedule_table(t, season)
            if games:
                return games
    return []


def ingest_regular_season(season: int, force: bool = False) -> pd.DataFrame:
    """Parse a season's full schedule into per-game results.

    Note: Sports-Reference does not expose a single "all games" schedule
    page. We therefore aggregate conference-schedule pages (one request
    per conference per season). See `ingest_regular_season_via_conferences`.
    """
    return ingest_regular_season_via_conferences(season, force=force)


def ingest_regular_season_via_conferences(
    season: int, force: bool = False
) -> pd.DataFrame:
    """Pull every conference's schedule page for a season and concat."""
    # Discover conferences from the standings page (cheapest way)
    from madness.ingest.conferences import parse_standings, standings_url
    try:
        stand_html = fetch(standings_url(season), namespace=NAMESPACE, force=force)
    except Exception as exc:
        log.error("standings_fetch_failed", season=season, error=str(exc))
        return pd.DataFrame()
    standings = parse_standings(stand_html, season)
    if standings.empty:
        log.warning("no_standings_no_conferences", season=season)
        return pd.DataFrame()

    conf_slugs = _conference_slugs(standings["conference"].dropna().unique().tolist())
    games: list[RegularSeasonGame] = []
    for slug in conf_slugs:
        url = conference_schedule_url(slug, season)
        try:
            html = fetch(url, namespace=NAMESPACE, force=force)
        except Exception as exc:
            log.warning("conf_schedule_skip", slug=slug, season=season, error=str(exc))
            continue
        games.extend(parse_schedule_page(html, season))

    if not games:
        log.warning("no_games_extracted", season=season)
        return pd.DataFrame()

    df = pd.DataFrame([asdict(g) for g in games])
    # Dedupe (non-conf games appear in both teams' conference schedule pages)
    df = df.drop_duplicates(subset=["season", "date", "team_winner", "team_loser"])
    out = INTERIM_DIR / "regular_season" / f"season_{season}.parquet"
    write_parquet(df, out)
    log.info("sr_regular_season_written", season=season, rows=len(df), path=str(out))
    return df


def _conference_slugs(conf_names: list[str]) -> list[str]:
    """Turn human-readable conference names into URL slugs.

    SR's slugs are lowercase, dashed, e.g. "Big 12 Conference" → "big-12".
    This is best-effort; unknown slugs fail gracefully at fetch time.
    """
    out = []
    for name in conf_names:
        s = name.lower().strip()
        for suffix in (" conference", " conf.", " conf"):
            if s.endswith(suffix):
                s = s[: -len(suffix)]
        s = s.replace("'", "").replace(".", "")
        s = "-".join(s.split())
        out.append(s)
    return out


def ingest_school_stats(season: int, force: bool = False) -> pd.DataFrame:
    """Per-team season aggregates — ONE request per season.

    Returns W, L, SRS, SOS, points-for/against, and (for 1997+) the
    Four Factors columns. This is the fastest way to bootstrap the
    baseline model without needing per-game scrapes.
    """
    url = school_stats_url(season)
    html = fetch(url, namespace=NAMESPACE, force=force)
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="basic_school_stats")
    if table is None:
        for sub in _parse_comments(soup):
            table = sub.find("table", id="basic_school_stats")
            if table is not None:
                break
    if table is None:
        log.warning("school_stats_table_missing", season=season)
        return pd.DataFrame()

    rows: list[dict] = []
    tbody = table.find("tbody")
    if tbody is None:
        return pd.DataFrame()
    for tr in tbody.find_all("tr"):
        if tr.get("class") and "thead" in tr.get("class"):
            continue
        row: dict = {"season": season}
        for cell in tr.find_all(["th", "td"]):
            stat = cell.get("data-stat")
            if stat:
                row[stat] = cell.get_text(strip=True)
        if row.get("school_name"):
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Coerce numeric columns
    for col in df.columns:
        if col in ("season", "school_name", "conf"):
            continue
        df[col] = pd.to_numeric(df[col], errors="ignore")
    out = INTERIM_DIR / "school_stats" / f"season_{season}.parquet"
    write_parquet(df, out)
    log.info("school_stats_written", season=season, rows=len(df))
    return df


def ingest_school_stats_range(
    start: int, end: int, force: bool = False
) -> pd.DataFrame:
    frames = []
    for season in range(start, end + 1):
        try:
            frames.append(ingest_school_stats(season, force=force))
        except Exception as exc:
            log.error("school_stats_failed", season=season, error=str(exc))
    if not frames:
        return pd.DataFrame()
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    out = INTERIM_DIR / "school_stats_all.parquet"
    write_parquet(df, out)
    return df


def ingest_regular_season_range(
    start: int, end: int, force: bool = False
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in range(start, end + 1):
        try:
            frames.append(ingest_regular_season(season, force=force))
        except Exception as exc:
            log.error("sr_regular_season_failed", season=season, error=str(exc))
    if not frames:
        return pd.DataFrame()
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    out = INTERIM_DIR / "regular_season_all.parquet"
    write_parquet(df, out)
    log.info("sr_regular_season_all_written", rows=len(df))
    return df


def to_canonical_game_frame(regular_season_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a schedule frame into the team_a/team_b/site schema used
    by the feature builder."""
    if regular_season_df.empty:
        return pd.DataFrame()
    out = regular_season_df.copy()
    out = out.rename(columns={
        "team_winner": "team_a", "team_loser": "team_b",
        "score_winner": "score_a", "score_loser": "score_b",
    })

    def _site(row):
        if row["site_neutral"]:
            return "neutral"
        if row["site_home"] == row["team_a"]:
            return "home"
        return "away"

    out["site"] = out.apply(_site, axis=1)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out[["season", "date", "team_a", "team_b", "score_a", "score_b", "site", "overtime"]]
