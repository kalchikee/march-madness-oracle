"""DuckDB file snapshot management via GitHub release assets.

Philosophy: keep the bulky DuckDB out of git, but keep it versioned and
reproducible by storing as a release asset. Workflow jobs download at
start, update, and upload at end.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from madness.config import DUCKDB_ASSET_NAME, DUCKDB_PATH, DUCKDB_RELEASE_TAG
from madness.logging_setup import get_logger

log = get_logger(__name__)


def connect(read_only: bool = False):
    """Lazy-import duckdb so modules that only need write_parquet don't need it."""
    import duckdb
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DUCKDB_PATH), read_only=read_only)


def download_latest(repo: str) -> bool:
    """Download the current DuckDB release asset via `gh`.

    Returns True if downloaded, False if no asset exists yet (first-run).
    """
    if DUCKDB_PATH.exists():
        log.info("duckdb_present_skip_download", path=str(DUCKDB_PATH))
        return True
    tmp = DUCKDB_PATH.with_suffix(".duckdb.incoming")
    cmd = [
        "gh", "release", "download", DUCKDB_RELEASE_TAG,
        "--repo", repo,
        "--pattern", DUCKDB_ASSET_NAME,
        "--output", str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning(
            "duckdb_download_failed",
            stderr=result.stderr[:400],
            returncode=result.returncode,
        )
        return False
    shutil.move(tmp, DUCKDB_PATH)
    log.info("duckdb_downloaded", size=DUCKDB_PATH.stat().st_size)
    return True


def upload_latest(repo: str, notes: str = "") -> None:
    """Upload current DuckDB as release asset (overwrite or create)."""
    if not DUCKDB_PATH.exists():
        log.warning("duckdb_missing_skip_upload")
        return
    subprocess.run(
        ["gh", "release", "view", DUCKDB_RELEASE_TAG, "--repo", repo],
        capture_output=True,
    )
    exists = subprocess.run(
        ["gh", "release", "view", DUCKDB_RELEASE_TAG, "--repo", repo],
        capture_output=True,
    ).returncode == 0

    if not exists:
        subprocess.run(
            [
                "gh", "release", "create", DUCKDB_RELEASE_TAG,
                "--repo", repo,
                "--title", "Data snapshot (latest)",
                "--notes", notes or "Automated data snapshot.",
            ],
            check=True,
        )

    subprocess.run(
        [
            "gh", "release", "upload", DUCKDB_RELEASE_TAG,
            str(DUCKDB_PATH),
            "--repo", repo,
            "--clobber",
        ],
        check=True,
    )
    log.info("duckdb_uploaded", size=DUCKDB_PATH.stat().st_size)


def table_exists(conn: duckdb.DuckDBPyConnection, name: str) -> bool:
    q = "SELECT 1 FROM information_schema.tables WHERE table_name = ?"
    return conn.execute(q, [name]).fetchone() is not None


def write_parquet(df, path: Path) -> None:
    """Write a DataFrame (pandas or polars) to parquet atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if hasattr(df, "write_parquet"):  # polars
        df.write_parquet(tmp)
    else:  # pandas
        df.to_parquet(tmp, index=False)
    tmp.replace(path)
