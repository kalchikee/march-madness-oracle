"""Monte Carlo bracket simulation.

For a 64-team bracket, the exact champion probabilities are easy
(multiply through the tree), but the joint distribution of outcomes
(e.g., P(Final Four = {A, B, C, D})) has no closed form. We sample.

10,000 simulations is plenty for round-level probabilities. For
championship prob of deep dark horses, bump to 50,000.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SimulationResult:
    champions: Counter
    final_four_counts: Counter
    round_probs: dict[str, dict[str, float]]  # team -> {round: prob}


def simulate_bracket(
    r64_matchups: list[dict],
    predict_fn,
    n_sims: int = 10_000,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Monte Carlo-advance teams through 6 rounds.

    r64_matchups: [{region, seed_a, team_a, seed_b, team_b}, ...] in
                  bracket order (seeding determines later-round pairings).
    predict_fn(team_a, team_b, round_idx) -> P(team_a wins)
    """
    rng = rng or np.random.default_rng(42)
    rounds_reached: dict[str, Counter] = {}

    champions: Counter = Counter()
    final_fours: Counter = Counter()

    for _ in range(n_sims):
        alive = list(r64_matchups)
        round_idx = 1

        current_bracket = []
        for m in r64_matchups:
            current_bracket.append([
                {"seed": m["seed_a"], "team": m["team_a"], "region": m["region"]},
                {"seed": m["seed_b"], "team": m["team_b"], "region": m["region"]},
            ])

        while len(current_bracket) > 0 and round_idx <= 6:
            winners = []
            for pair in current_bracket:
                a, b = pair[0], pair[1]
                p = predict_fn(a["team"], b["team"], round_idx)
                win_a = rng.random() < p
                winner = a if win_a else b
                winners.append(winner)
                rounds_reached.setdefault(winner["team"], Counter())[round_idx] += 1
            if round_idx == 4:
                for w in winners:
                    final_fours[w["team"]] += 1
            if len(winners) == 1:
                champions[winners[0]["team"]] += 1
                break
            current_bracket = [
                [winners[i], winners[i + 1]] for i in range(0, len(winners), 2)
            ]
            round_idx += 1

    round_probs = {
        team: {
            f"round_{r}": count / n_sims
            for r, count in counts.items()
        }
        for team, counts in rounds_reached.items()
    }
    return SimulationResult(
        champions=champions,
        final_four_counts=final_fours,
        round_probs=round_probs,
    )


def top_champions(result: SimulationResult, top_k: int = 10) -> list[tuple[str, float]]:
    total = sum(result.champions.values())
    if total == 0:
        return []
    return [
        (team, count / total)
        for team, count in result.champions.most_common(top_k)
    ]
