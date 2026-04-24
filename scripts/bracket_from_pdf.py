"""Fallback bracket parser — extract matchups from the official PDF.

Selection Sunday is chaos. When the web JSON endpoint lags or breaks,
the bracket PDF is the most reliable source. This script turns the PDF
into the same normalized bracket.json shape the predict job expects.

Usage:
    python scripts/bracket_from_pdf.py path/to/bracket.pdf

Dependency: pdfplumber (optional; install on demand).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from madness.logging_setup import configure, get_logger

configure()
log = get_logger(__name__)


def parse(pdf_path: Path) -> dict:
    try:
        import pdfplumber
    except ImportError:
        log.error("pdfplumber_missing_install", hint="pip install pdfplumber")
        raise SystemExit(1)

    matchups: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            # TODO (Milestone 13): implement brute-force parser
            # Typical structure: "1 Team Name" lines grouped by region.
            # A line-based parser is fragile; prefer the web JSON when available.
            _ = text
    return {
        "season": None,
        "regions": ["East", "West", "South", "Midwest"],
        "rounds": {"round_of_64": matchups},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, default=Path("bracket.json"))
    args = ap.parse_args()

    bracket = parse(args.pdf)
    args.out.write_text(json.dumps(bracket, indent=2))
    log.info("bracket_pdf_parsed", out=str(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
