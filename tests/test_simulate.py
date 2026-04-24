"""Bracket simulation smoke test."""
from __future__ import annotations

from madness.predict.simulate import simulate_bracket, top_champions


def _fake_r64(n_regions: int = 4) -> list[dict]:
    teams = []
    for r in range(n_regions):
        for s in range(1, 9):
            teams.append({
                "region": f"R{r}", "seed_a": s, "team_a": f"R{r}S{s}",
                "seed_b": 17 - s, "team_b": f"R{r}S{17-s}",
            })
    return teams


def test_simulate_runs_and_picks_champion():
    matchups = _fake_r64()

    def predict_fn(a: str, b: str, round_idx: int) -> float:
        # Higher-seed number means lower rank → favor A by seed encoded in name
        return 0.65 if int(a.split("S")[-1]) <= int(b.split("S")[-1]) else 0.35

    result = simulate_bracket(matchups, predict_fn, n_sims=200)
    champs = top_champions(result, top_k=3)
    assert len(champs) > 0
    assert sum(p for _, p in champs) <= 1.01


def test_simulate_exact_champion_total():
    matchups = _fake_r64(n_regions=1)

    def predict_fn(a: str, b: str, round_idx: int) -> float:
        return 0.5

    result = simulate_bracket(matchups, predict_fn, n_sims=100)
    assert sum(result.champions.values()) == 100
