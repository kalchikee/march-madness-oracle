# Operational Runbook

How to operate the predictor once it's live on GitHub Actions.

## Secrets setup (one-time)

Go to: repo → Settings → Secrets and variables → Actions → New repository secret.

Required:
- `DISCORD_WEBHOOK_URL` — create via Discord → Server Settings → Integrations → Webhooks.

Optional:
- `KENPOM_USER`, `KENPOM_PASS` — KenPom login. Without these, pipeline falls back to Torvik + Sports-Reference.

## Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | PR / push main | Lint, type-check, pytest |
| `daily_data_refresh.yml` | Cron 6 AM UTC, Nov–Apr | Pull new results, refresh DuckDB snapshot |
| `weekly_retrain.yml` | Cron Sunday 4 AM UTC | Optuna tune + champion/challenger promotion |
| `tournament_predict.yml` | Cron every 4h, Mar 15 – Apr 10 | Predict bracket, post Discord |

All are `workflow_dispatch`-triggerable from the Actions tab for on-demand runs.

## First-time bootstrap

This must run locally first because the 50-year scrape is long:

```bash
pip install -e ".[dev]"
python scripts/bootstrap_historical.py --start-season 1985
madness train baseline
```

Then push the resulting `data/processed/*.parquet` and `models/champion/` and create the release asset:

```bash
gh release create data-latest data/madness.duckdb \
  --title "Initial data snapshot" \
  --notes "Bootstrap run"
```

## Manual overrides

Drop a JSON file at `overrides/manual_injuries.json`:

```json
{
  "season": 2026,
  "team": "Kansas",
  "note": "Key player out first weekend",
  "win_prob_adjustment": -0.05
}
```

The predictor reads this at inference time (not during training) and adjusts the predicted probability.

## Manually fill a bracket

If the NCAA JSON endpoint fails on Selection Sunday:

```bash
# Option 1: parse the PDF
python scripts/bracket_from_pdf.py path/to/bracket.pdf

# Option 2: hand-write bracket.json in the canonical shape
# (see Phase 5A in MARCH_MADNESS_PREDICTOR_PLAN.md)
```

Then push `bracket.json` and trigger `tournament_predict.yml` from the Actions tab.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| 429 from Sports-Reference | scrape rate too aggressive | increase `DEFAULT_RATE_LIMIT_SECONDS` in `src/madness/config.py` |
| No Discord posts | missing / bad webhook URL | verify secret; check workflow log |
| Backtest reports 85%+ | almost certainly a leakage bug | check feature `cutoff` logic; run `tests/test_backtest_leakage.py` |
| KenPom login failing | session expiry or 2FA | clear `data/raw/kenpom/` and re-run; verify credentials |
| DuckDB asset missing | first run | pipeline auto-creates it on next successful cycle |

## Rotation

- Webhooks: rotate by creating a new Discord webhook and replacing the secret; old secret invalidates automatically.
- KenPom: change password, update secret.

## Failure recovery

Every workflow writes a run manifest to `runs/<timestamp>.json` with inputs, outputs, and metrics. On failure:

1. Check the Actions log → see the failing step.
2. Look at `runs/` for the last-known-good state.
3. If model output is wrong but data is fine: re-trigger `weekly_retrain.yml` with `trials=10` for a quick sanity pass.
4. If data is corrupted: `gh release view data-latest` and roll back by downloading the previous asset version.
