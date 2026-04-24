"""Polite HTTP client with on-disk cache, rate limiting, and retries.

Every scraper in this project goes through `fetch()`. That gives us one
place to enforce robots.txt etiquette, rate limits, and caching so that
re-running a pipeline does not re-hit the source site.
"""
from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from madness.config import DEFAULT_RATE_LIMIT_SECONDS, DEFAULT_USER_AGENT, RAW_DIR
from madness.logging_setup import get_logger

log = get_logger(__name__)


class RateLimiter:
    """Per-host minimum interval between requests. Thread-safe."""

    def __init__(self, min_interval: float = DEFAULT_RATE_LIMIT_SECONDS) -> None:
        self.min_interval = min_interval
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        with self._lock:
            last = self._last.get(host, 0.0)
            elapsed = time.monotonic() - last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last[host] = time.monotonic()


_rate_limiter = RateLimiter()
_robots_cache: dict[str, RobotFileParser] = {}


def _robots_allowed(url: str, user_agent: str) -> bool:
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin not in _robots_cache:
        rp = RobotFileParser()
        rp.set_url(f"{origin}/robots.txt")
        try:
            rp.read()
        except Exception as exc:
            log.warning("robots_fetch_failed", origin=origin, error=str(exc))
        _robots_cache[origin] = rp
    return _robots_cache[origin].can_fetch(user_agent, url)


def cache_path(url: str, namespace: str) -> Path:
    """Deterministic cache path for a URL."""
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return RAW_DIR / namespace / digest[:2] / f"{digest}.html"


class PermanentHTTPError(Exception):
    """4xx errors (except 429) — don't retry, caller should skip."""


MAX_RETRY_AFTER_SECONDS = 60.0  # Honor Retry-After up to this; beyond, abort.


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def _get_with_retry(url: str, headers: dict[str, str], timeout: float) -> requests.Response:
    resp = requests.get(url, headers=headers, timeout=timeout)
    if resp.status_code == 429:
        retry_after = float(resp.headers.get("Retry-After", "10"))
        if retry_after > MAX_RETRY_AFTER_SECONDS:
            # Upstream asked for a very long wait — abort rather than block
            log.error("rate_limited_abort", url=url, retry_after=retry_after)
            raise PermanentHTTPError(f"429 with Retry-After {retry_after}s — aborting")
        log.warning("rate_limited", url=url, retry_after=retry_after)
        time.sleep(retry_after)
        raise requests.RequestException(f"429 from {url}")
    if 400 <= resp.status_code < 500:
        # 404 / 403 etc. — permanent, do not retry
        raise PermanentHTTPError(f"{resp.status_code} from {url}")
    if resp.status_code >= 500:
        raise requests.RequestException(f"{resp.status_code} from {url}")
    resp.raise_for_status()
    return resp


def fetch(
    url: str,
    *,
    namespace: str,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = 30.0,
    force: bool = False,
    check_robots: bool = True,
    rate_limit_seconds: float | None = None,
) -> str:
    """Fetch a URL with rate limit + disk cache. Returns response body.

    `namespace` segments the cache (one per data source).
    """
    cp = cache_path(url, namespace)
    if cp.exists() and not force:
        return cp.read_text(encoding="utf-8")

    if check_robots and not _robots_allowed(url, user_agent):
        raise PermissionError(f"robots.txt disallows fetching {url}")

    limiter = (
        RateLimiter(rate_limit_seconds) if rate_limit_seconds is not None else _rate_limiter
    )
    host = urlparse(url).netloc
    limiter.wait(host)

    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    log.info("http_fetch", url=url, namespace=namespace)
    resp = _get_with_retry(url, headers, timeout)

    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(resp.text, encoding="utf-8")
    return resp.text
