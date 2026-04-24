# NCAA March Madness Predictor

ML system that predicts NCAA Division I Men's Basketball Tournament winners, with automated Discord delivery via GitHub Actions.

**Target:** ≥75% walk-forward accuracy, ≥80% as stretch.
**Reality check:** 20–25% of tournament games are coin-flips. A backtest reporting >80% is almost always leaking. See [BACKTEST_REPORT.md](./BACKTEST_REPORT.md) for honest per-round numbers.

## Quick start

```bash
# Install
pip install -e ".[dev]"

# Bootstrap historical data (one-time, 2–4 hours with rate limits)
madness ingest bootstrap --start-season 1985

# Train the baseline
madness train baseline

# Full walk-forward backtest
madness backtest walk-forward --start 2003 --end 2024

# Predict the current bracket and post to Discord
madness predict bracket --season 2026 --post-discord
```

## Architecture

```
GitHub Actions (cron)
  ├── daily_data_refresh.yml    (pulls new results Nov–Apr)
  ├── weekly_retrain.yml        (Sunday: retune + champion/challenger promotion)
  └── tournament_predict.yml    (every 4h Mar 15 – Apr 10: predict + Discord)

Data flow:  Ingest → Interim → Features → Train → Champion → Predict → Discord
Storage:    DuckDB file in GitHub release assets (reproducible, no git bloat)
```

## Secrets (GitHub repo settings → Secrets)

| Name                  | Required | Purpose                                        |
|-----------------------|----------|------------------------------------------------|
| `DISCORD_WEBHOOK_URL` | Yes      | Where predictions are posted                   |
| `KENPOM_USER`         | No       | If supplied, enables KenPom features           |
| `KENPOM_PASS`         | No       | Paired with `KENPOM_USER`                      |

If KenPom credentials are absent the pipeline uses Bart Torvik (free) and other sources and skips KenPom-only features gracefully.

## Repo layout

See [MARCH_MADNESS_PREDICTOR_PLAN.md](./MARCH_MADNESS_PREDICTOR_PLAN.md) for the full design; [RUNBOOK.md](./RUNBOOK.md) for operational procedures.

## Honest accuracy notes

| Benchmark                  | Accuracy |
|----------------------------|----------|
| Pure chalk (higher seed)   | ~71%     |
| Vegas closing line implied | ~72%     |
| FiveThirtyEight (historic) | ~73%     |
| This project (target)      | 75%+     |
| This project (stretch)     | 80%+     |

Rounds are NOT equal difficulty. R64 baselines near 78–82%; Sweet 16 onward drops to 55–65%. Always read backtest reports by round.
