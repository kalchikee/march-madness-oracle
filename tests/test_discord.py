"""Discord webhook tests using the `responses` library to mock HTTP."""
from __future__ import annotations

import responses

from madness.notify.discord import (
    DiscordMessage,
    _predictions_hash,
    send_messages,
)
from madness.predict.bracket import Prediction, confidence_tier


WEBHOOK = "https://discord.com/api/webhooks/TEST/TOKEN"


def _sample_preds() -> list[Prediction]:
    return [
        Prediction(
            region="East", round_name="Round of 64",
            seed_a=1, team_a="Duke", seed_b=16, team_b="Sacred Heart",
            prob_a_wins=0.97, pick="Duke",
            confidence_tier=confidence_tier(0.97),
        ),
        Prediction(
            region="East", round_name="Round of 64",
            seed_a=8, team_a="Iowa", seed_b=9, team_b="Utah",
            prob_a_wins=0.52, pick="Iowa",
            confidence_tier=confidence_tier(0.52),
        ),
    ]


def test_predictions_hash_stable():
    preds = _sample_preds()
    assert _predictions_hash(preds) == _predictions_hash(preds)


def test_predictions_hash_changes_with_probability():
    a = _sample_preds()
    b = _sample_preds()
    b[0].prob_a_wins = 0.80
    assert _predictions_hash(a) != _predictions_hash(b)


@responses.activate
def test_send_messages_posts_to_webhook():
    responses.add(responses.POST, WEBHOOK, body="", status=200)
    send_messages(WEBHOOK, [DiscordMessage(content=None, embeds=[{"title": "t"}])])
    assert len(responses.calls) == 1


@responses.activate
def test_send_messages_retries_on_429():
    responses.add(responses.POST, WEBHOOK, body="", status=429,
                  headers={"Retry-After": "0"})
    responses.add(responses.POST, WEBHOOK, body="", status=200)
    send_messages(WEBHOOK, [DiscordMessage(content=None, embeds=[{"title": "t"}])])
    assert len(responses.calls) >= 2


def test_confidence_tiers():
    assert confidence_tier(0.95) == "lock"
    assert confidence_tier(0.70) == "likely"
    assert confidence_tier(0.57) == "lean"
    assert confidence_tier(0.50) == "tossup"
    assert confidence_tier(0.05) == "lock"  # symmetric
