"""Pytest fixtures shared across the suite."""
from __future__ import annotations

import sys
from pathlib import Path

# Make `madness` importable without installing the package
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
