"""Central configuration: paths, constants, season math."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"
CHAMPION_DIR = MODELS_DIR / "champion"
CHALLENGERS_DIR = MODELS_DIR / "challengers"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LEADERBOARD_CSV = EXPERIMENTS_DIR / "leaderboard.csv"
OPTUNA_DB = EXPERIMENTS_DIR / "optuna.db"

CONFIGS_DIR = PROJECT_ROOT / "configs"
RUNS_DIR = PROJECT_ROOT / "runs"
STATE_DIR = PROJECT_ROOT / "state"

DUCKDB_PATH = DATA_DIR / "madness.duckdb"

# NCAA tournament era boundaries
TOURNAMENT_EXPANSION_64 = 1985
TOURNAMENT_EXPANSION_68 = 2011
KENPOM_AVAILABLE_FROM = 2002
TORVIK_AVAILABLE_FROM = 2008
MASSEY_AVAILABLE_FROM = 1995
ADVANCED_STATS_FROM = 1997

# Scrape politeness
DEFAULT_RATE_LIMIT_SECONDS = 7.0
DEFAULT_USER_AGENT = (
    "madness-predictor/0.1 (research project; contact via repo issues)"
)

# GitHub release asset name for DuckDB snapshot
DUCKDB_RELEASE_TAG = "data-latest"
DUCKDB_ASSET_NAME = "madness.duckdb"


def current_season() -> int:
    """NCAA basketball season naming convention: 2025-26 season = 2026.

    The season a given date belongs to is the calendar year of the
    March Madness tournament. Transition point is July 1.
    """
    today = date.today()
    return today.year + 1 if today.month >= 7 else today.year


def season_start_date(season: int) -> date:
    """First possible regular-season game date for a season (~Nov 1)."""
    return date(season - 1, 11, 1)


def season_end_date(season: int) -> date:
    """Last day of the tournament — conservatively late April."""
    return date(season, 4, 15)


def tournament_window(season: int) -> tuple[date, date]:
    """Approximate window for the NCAA tournament for a given season."""
    return date(season, 3, 14), date(season, 4, 10)


def is_tournament_window(today: date | None = None) -> bool:
    today = today or date.today()
    start, end = tournament_window(today.year)
    return start <= today <= end


@dataclass(frozen=True)
class Secrets:
    """Read secrets from env; never log these."""
    discord_webhook_url: str | None = None
    kenpom_user: str | None = None
    kenpom_pass: str | None = None
    github_token: str | None = None

    @classmethod
    def from_env(cls) -> "Secrets":
        return cls(
            discord_webhook_url=os.environ.get("DISCORD_WEBHOOK_URL"),
            kenpom_user=os.environ.get("KENPOM_USER"),
            kenpom_pass=os.environ.get("KENPOM_PASS"),
            github_token=os.environ.get("GITHUB_TOKEN"),
        )

    @property
    def has_kenpom(self) -> bool:
        return bool(self.kenpom_user and self.kenpom_pass)

    @property
    def has_discord(self) -> bool:
        return bool(self.discord_webhook_url)


def ensure_dirs() -> None:
    """Create all runtime directories if missing."""
    for d in (
        RAW_DIR, INTERIM_DIR, PROCESSED_DIR, EXTERNAL_DIR,
        CHAMPION_DIR, CHALLENGERS_DIR,
        EXPERIMENTS_DIR, RUNS_DIR, STATE_DIR, CONFIGS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def run_id() -> str:
    """Stable ID for a single pipeline execution."""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
