"""One-time 50-year backfill across all ingesters.

Run locally once before the first GitHub Actions cycle. Takes ~2-4 hours
because of scrape rate limits — be polite.

Usage:
    python scripts/bootstrap_historical.py --start-season 1985
"""
from __future__ import annotations

import argparse

from madness.config import Secrets, current_season
from madness.ingest import torvik
from madness.ingest.tournament_results import build_canonical_tournament_table
from madness.logging_setup import configure, get_logger

configure()
log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=1985)
    parser.add_argument("--end-season", type=int, default=current_season() - 1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    log.info("bootstrap_start", start=args.start_season, end=args.end_season)
    build_canonical_tournament_table(args.start_season, args.end_season, force=args.force)
    torvik.ingest_range(max(2008, args.start_season), args.end_season, force=args.force)

    secrets = Secrets.from_env()
    if secrets.has_kenpom:
        from madness.ingest import kenpom
        kenpom.ingest_range(max(2002, args.start_season), args.end_season)
    else:
        log.info("kenpom_skipped_no_credentials")

    log.info("bootstrap_complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
