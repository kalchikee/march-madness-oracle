"""Discord webhook sender with embed formatting.

Constraints:
- Never log the webhook URL.
- Retry with exponential backoff on 429 / 5xx.
- Respect Discord's 6000-char total-embed and 4096-char-per-field limits
  by splitting into multiple messages when needed.
- Include a `message_key` so re-posts are idempotent (the caller hashes
  the prediction set and stores the last-posted hash in state/).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from madness.logging_setup import get_logger
from madness.predict.bracket import Prediction

log = get_logger(__name__)

COLOR_FAVORITE = 0x2ECC71  # green
COLOR_UPSET_PICK = 0xE74C3C  # red
COLOR_TOSSUP = 0xF1C40F  # yellow

MAX_EMBED_CHARS = 5500  # headroom under the 6000 hard limit
TIER_EMOJI = {
    "lock": "🔒",
    "likely": "✅",
    "lean": "🧭",
    "tossup": "⚠️",
}


@dataclass
class DiscordMessage:
    content: str | None
    embeds: list[dict]


def _predictions_hash(preds: list[Prediction]) -> str:
    serialized = json.dumps(
        [(p.team_a, p.team_b, round(p.prob_a_wins, 3)) for p in preds],
        sort_keys=True,
    )
    return hashlib.sha1(serialized.encode()).hexdigest()[:12]


def format_r64_message(preds: list[Prediction], season: int) -> list[DiscordMessage]:
    by_region: dict[str, list[Prediction]] = {}
    for p in preds:
        by_region.setdefault(p.region or "Unknown", []).append(p)

    messages: list[DiscordMessage] = []
    for region, region_preds in sorted(by_region.items()):
        lines = []
        for p in sorted(region_preds, key=lambda x: min(x.seed_a, x.seed_b)):
            emoji = TIER_EMOJI.get(p.confidence_tier, "")
            pct = round(max(p.prob_a_wins, 1 - p.prob_a_wins) * 100)
            is_upset = (p.pick == p.team_a and p.seed_a > p.seed_b) or (
                p.pick == p.team_b and p.seed_b > p.seed_a
            )
            upset_marker = " 💥" if is_upset else ""
            lines.append(
                f"{emoji} **({p.seed_a})** {p.team_a} vs **({p.seed_b})** {p.team_b}"
                f" → **{p.pick}** ({pct}%){upset_marker}"
            )
        embed = {
            "title": f"{region} Region — Round of 64",
            "description": "\n".join(lines)[:4090],
            "color": COLOR_FAVORITE,
            "footer": {"text": f"Season {season} predictions • hash {_predictions_hash(preds)}"},
        }
        messages.append(DiscordMessage(content=None, embeds=[embed]))
    return messages


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)
def _post_one(webhook_url: str, message: DiscordMessage) -> None:
    payload: dict = {"embeds": message.embeds}
    if message.content:
        payload["content"] = message.content
    resp = requests.post(webhook_url, json=payload, timeout=30)
    if resp.status_code == 429:
        ra = float(resp.headers.get("Retry-After", "5"))
        log.warning("discord_rate_limited", retry_after=ra)
        time.sleep(ra)
        raise requests.RequestException("discord 429")
    if resp.status_code >= 500:
        raise requests.RequestException(f"discord {resp.status_code}")
    resp.raise_for_status()


def send_messages(webhook_url: str, messages: list[DiscordMessage]) -> None:
    if not webhook_url:
        log.warning("discord_no_webhook_skip")
        return
    for m in messages:
        _post_one(webhook_url, m)
        time.sleep(1)  # courtesy spacing
    log.info("discord_sent", n=len(messages))
