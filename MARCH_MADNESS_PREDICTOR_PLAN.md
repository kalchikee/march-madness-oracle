# NCAA March Madness Predictor — Project Plan for Claude Code

## Project Overview

Build a production-grade machine learning system that predicts the winners of NCAA Division I Men's Basketball Tournament ("March Madness") games. The system ingests ~50 years of historical tournament and regular-season data, engineers a rich feature set, trains and tunes an ensemble of models, iteratively improves accuracy through automated experimentation, and — once tournament time arrives — automatically fetches the bracket, generates predictions, and posts them to a Discord channel via webhook.

**The entire system runs on GitHub Actions on a schedule.** The user's local machine never needs to be on.

### Primary Goals
1. **Historical accuracy floor:** Achieve ≥75% accuracy on held-out tournament games (the Vegas/expert benchmark sits around 70–72%; 538's models historically land near 73%).
2. **Stretch goal:** ≥80% overall accuracy and correctly predict ≥50% of Sweet 16 teams in out-of-sample tournaments.
3. **Automation:** Zero manual steps during tournament week. GitHub Actions pulls fresh data, runs predictions, posts to Discord.
4. **Self-improvement loop:** A scheduled "research" workflow re-runs feature engineering and hyperparameter search, compares candidate models against the current champion, and promotes a new champion only when it beats the current one on a rigorous backtest.

### Non-Goals
- Betting/gambling integration (odds are used only as a calibration feature, not for bet placement).
- Women's tournament (can be added later — architecture should not preclude it).
- Real-time in-game prediction (we predict pre-game only).

---

## Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────┐
│                    GitHub Actions (cron)                       │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Daily data │  │ Weekly model │  │ Tournament-time      │  │
│  │ refresh    │  │ retrain &    │  │ prediction + Discord │  │
│  │            │  │ benchmark    │  │ webhook              │  │
│  └─────┬──────┘  └──────┬───────┘  └──────────┬───────────┘  │
└────────┼────────────────┼─────────────────────┼──────────────┘
         │                │                     │
         ▼                ▼                     ▼
   ┌──────────────────────────────────────────────────┐
   │     Data Lake (DuckDB file committed to repo     │
   │     OR external storage — see Data Storage)      │
   └──────────────────────────────────────────────────┘
         │                │                     │
         ▼                ▼                     ▼
   Ingestion ──► Feature Engineering ──► Model Training ──► Serving
```

### Tech Stack (recommended)
- **Language:** Python 3.11+
- **Package manager:** `uv` (fast, reproducible) or `poetry`
- **Data:** `pandas`, `polars` (for speed on big joins), `duckdb` (embedded analytical DB)
- **ML:** `scikit-learn`, `xgboost`, `lightgbm`, `catboost`. Optional deep learning with `pytorch` for a tabular transformer experiment.
- **Hyperparameter search:** `optuna` with SQLite storage
- **Experiment tracking:** `mlflow` (local file backend, committed selectively) or a single CSV `experiments/leaderboard.csv`
- **Scraping/APIs:** `requests`, `httpx`, `beautifulsoup4`, `lxml`
- **Orchestration:** GitHub Actions (cron + workflow_dispatch)
- **Notifications:** Discord webhook via `requests`
- **Testing:** `pytest`, `pytest-cov`
- **Linting/formatting:** `ruff`, `mypy`

---

## Repository Layout

```
ncaa-madness-predictor/
├── .github/
│   └── workflows/
│       ├── daily_data_refresh.yml
│       ├── weekly_retrain.yml
│       ├── tournament_predict.yml
│       └── ci.yml
├── data/
│   ├── raw/                    # Immutable source data (CSV, JSON, HTML dumps)
│   ├── interim/                # Cleaned, not yet feature-engineered
│   ├── processed/              # Final feature tables (Parquet)
│   └── external/               # Reference: conference lists, coach tenures, etc.
├── src/
│   └── madness/
│       ├── __init__.py
│       ├── config.py           # Paths, constants, current season year
│       ├── ingest/
│       │   ├── kenpom.py       # (if user provides credentials)
│       │   ├── sports_reference.py
│       │   ├── ncaa_api.py
│       │   ├── massey.py
│       │   └── tournament_results.py
│       ├── features/
│       │   ├── team_season.py  # Per-team-per-season aggregates
│       │   ├── matchup.py      # Pairwise features (diffs, ratios)
│       │   ├── tournament.py   # Seed, region, round-specific
│       │   ├── momentum.py     # Last-N-games form, conference tourney result
│       │   ├── coach.py        # Coach tournament experience
│       │   └── build.py        # Orchestrates full feature pipeline
│       ├── models/
│       │   ├── base.py         # Model interface
│       │   ├── logistic.py
│       │   ├── gbm.py          # XGBoost / LightGBM / CatBoost wrappers
│       │   ├── ensemble.py     # Stacking, blending
│       │   └── registry.py     # Load/save champions
│       ├── train/
│       │   ├── backtest.py     # Walk-forward by tournament year
│       │   ├── tune.py         # Optuna driver
│       │   └── evaluate.py     # Metrics + bracket scoring
│       ├── predict/
│       │   ├── bracket.py      # Generate full 64/68-team bracket predictions
│       │   └── simulate.py     # Monte-Carlo round-by-round simulation
│       ├── notify/
│       │   └── discord.py      # Webhook formatter + sender
│       └── cli.py              # Typer-based CLI entry point
├── notebooks/
│   └── exploration/            # Ad-hoc analysis; not required for pipeline
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_backtest.py
│   └── test_discord.py
├── models/
│   ├── champion/               # Current production model artifacts
│   └── challengers/            # Recently-trained candidates
├── experiments/
│   └── leaderboard.csv         # Append-only record of every training run
├── configs/
│   ├── default.yaml
│   ├── tune_xgboost.yaml
│   └── tune_ensemble.yaml
├── scripts/
│   ├── bootstrap_historical.py # One-time 50-year backfill
│   └── bracket_from_pdf.py     # Parse official bracket PDF when released
├── pyproject.toml
├── uv.lock (or poetry.lock)
├── README.md
└── MARCH_MADNESS_PREDICTOR_PLAN.md   # This file
```

---

## Data Storage: Pick a Strategy

Fifty years of tournament data plus per-game regular-season stats is on the order of 100–300 MB compressed as Parquet — small enough to commit to the repo, but not ideal. **Recommended:**

- Keep **raw scraped HTML/JSON** out of git (add to `.gitignore`). It's reproducible.
- Keep **`data/processed/*.parquet`** in the repo *only if* total size stays under ~200 MB; otherwise use **Git LFS** or an **external bucket**.
- **Preferred pattern:** Use a single **DuckDB file** (`data/madness.duckdb`) stored in a **GitHub release asset** (unlimited 2GB-per-file storage, free, versionable). The workflow downloads it at the start of a job, updates it, and uploads the new version as a release asset. This avoids git bloat while keeping everything self-contained in the repo.
- Feature tables and model artifacts (`models/champion/*.pkl`) are small (<10 MB) and *should* be committed so every run is fully reproducible from a fresh clone.

Claude Code: implement the release-asset pattern with a helper in `src/madness/storage.py` that wraps `gh release download` / `gh release upload`.

---

## Phase 1 — Data Ingestion (50 Years of History)

The tournament expanded to 64 teams in 1985 and 68 in 2011. Earlier tournaments (1975–1984) had 32–53 teams with different structures. Data quality and granularity drops significantly before ~1985, and play-by-play / advanced stats are essentially unavailable before ~2002.

**Strategy:** Ingest the full 50 years, but build features that gracefully degrade — a model should be able to train on 1975+ using only basic features, and on 2002+ using the full rich feature set. Track feature availability per season.

### Data sources to implement (in priority order)

1. **Sports-Reference / College Basketball Reference** (`sports-reference.com/cbb/`)
   - Per-season team stats, per-game results, tournament brackets with seeds and results.
   - No official API; use polite scraping with `requests` + `beautifulsoup4`, a real User-Agent, and a rate limit of **1 request every 3 seconds** minimum. Cache every page to `data/raw/sports_reference/<season>/<url_hash>.html` so re-runs don't re-hit the site.
   - Covers 1975+ for results, ~1997+ for advanced stats.

2. **NCAA's public JSON endpoints** (`data.ncaa.com`)
   - Modern-era scores and schedules. Good supplement for recent seasons.

3. **KenPom** (`kenpom.com`) — adjusted efficiency ratings, tempo, luck
   - Covers 2002+. Requires a paid subscription (~$20/yr). The user should provide credentials as GitHub Secrets (`KENPOM_USER`, `KENPOM_PASS`). Implement a login + scrape with session cookies. **If credentials are not provided, skip gracefully** and proceed without KenPom features.
   - KenPom data is widely regarded as one of the single most predictive signals — include it if at all possible.

4. **Bart Torvik** (`barttorvik.com`) — free alternative to KenPom, covers 2008+
   - Has an unofficial JSON endpoint. Always ingest this whether or not KenPom is available.

5. **Massey Ratings** (`masseyratings.com`) — composite of many rating systems, 1995+
   - Lightweight daily scrape.

6. **Historical tournament results**
   - Include seeds, regions, rounds, scores, upsets. Cross-check Sports-Reference against Wikipedia as a sanity layer.

7. **External reference data** (commit to `data/external/`)
   - `conferences.csv` — season-by-season conference membership (teams move around)
   - `coaches.csv` — head coach per team per season
   - `arena_locations.csv` — for travel-distance feature (round-by-round neutral-site geography)

### Ingestion requirements
- Every ingester is **idempotent**: re-running does nothing if data is fresh.
- Every ingester writes to `data/raw/` first, then a normalizer writes to `data/interim/`.
- Each ingester has a `--season` flag and a `--force` flag.
- Every scrape is rate-limited and caches to disk.
- All ingesters log structured JSON to stdout so GitHub Actions logs are parseable.

---

## Phase 2 — Feature Engineering

This is where accuracy is won or lost. Build features at three levels.

### 2A. Team-season features (one row per team per season)
- Record (W, L, win %), home/away/neutral splits
- Strength of schedule (SOS) — use Sports-Reference's SRS and NET-style composite
- Offensive and defensive efficiency (points per 100 possessions), adjusted for opponent
- Tempo (possessions per 40 min)
- eFG%, TS%, TOV%, OReb%, FT rate — the "Four Factors" on offense and defense
- 3-point attempt rate and 3-point defense
- Experience: weighted average player minutes × class year (proxy for veteran vs. young teams)
- Height / roster size (if available from KenPom)
- Injuries flag — hard to automate; leave hook for manual override JSON
- Conference strength (aggregate of conference's teams' SRS)
- Conference tournament result (won/lost in finals/semis)
- Last-10 games record and point differential
- Longest win streak heading into the tournament

### 2B. Matchup features (computed at game time)
- Seed difference and seed ratio
- All team-season features as **differences** and **ratios** (team A minus team B)
- Style clash: tempo compatibility, 3-point-rate vs. 3-point-defense matchup
- Head-to-head history (rare — often zero)
- Common opponents (performance vs. shared opponents this season)
- Distance traveled to game site (miles from campus; first two rounds favor local teams historically)
- Rest days between games
- Historical conference-vs-conference tournament record (e.g., Big Ten vs. ACC) — rolling 5-year window

### 2C. Tournament-context features
- Round (1st four, R64, R32, S16, E8, F4, Championship) — **this matters a lot**; upset rates differ sharply by round
- Historical upset rate by seed matchup (e.g., 12 vs 5 → ~35%)
- Region (some regions stack favorites unevenly)
- Whether the team played in a conference championship game (fatigue vs. momentum — research is mixed; let the model decide)
- Coach tournament experience: career tournament wins, Final Four appearances, Elite Eight appearances
- Coach tenure at current school

### Feature availability flag
Every feature carries an `available_since_year` attribute. The feature builder materializes a sparse table and models that require missing features either impute or get filtered to the eligible time window.

### Implementation notes
- Use `polars` for the per-team-per-season aggregation — it's 5–20× faster than pandas on this workload.
- Write features out as a single Parquet file per season for efficient walk-forward backtesting.
- **Critical: No target leakage.** A team-season feature computed for the 2019 tournament must only use games played **before** the 2019 tournament tipped off. Build a `season_cutoff_date` param into every aggregator and unit-test it.

---

## Phase 3 — Modeling

### 3A. Target definition
- Binary classification: does Team A beat Team B?
- For every historical tournament game, generate **two rows** (A vs. B with target=1 for the winner, and B vs. A with target=0). Symmetrize inputs so the model can't learn "always pick the team in column A."
- Secondary target: point margin regression (useful for tiebreakers and bracket scoring pools).

### 3B. Model roster
Train and benchmark all of these:
1. **Logistic regression** with L2 regularization — strong baseline, interpretable.
2. **XGBoost** — historically the best single tabular model for this problem.
3. **LightGBM** — faster to tune, often comparable.
4. **CatBoost** — handles categorical features (conference, coach) natively.
5. **Random Forest** — baseline for comparison.
6. **Stacking ensemble** — logistic meta-learner over the above. This is usually the production model.
7. **(Experimental) Tabular transformer** — only pursue if time allows; rarely beats GBDT here.

### 3C. Hyperparameter tuning
- Use `optuna` with 100–500 trials per model.
- Optimize log loss (proper scoring rule) not accuracy.
- Store the Optuna study in `experiments/optuna.db` (SQLite) so runs resume across GitHub Actions jobs (hand off via release asset).

### 3D. Validation strategy — **this is the most important section**
**Never** use random k-fold. Use **walk-forward by tournament year**:

- Train on tournaments 1985–2000, validate on 2001, then train 1985–2001 validate on 2002, etc.
- Report the **mean** and **distribution** of accuracy / log loss across the walk-forward windows.
- Keep the **last three tournaments as a pure holdout** — no hyperparameter is tuned on them; they are only looked at when promoting a champion.
- Report accuracy broken down by round (R64, R32, S16, E8, F4, Championship). Good models can get 90%+ on R64 and struggle on S16+.
- Report accuracy by seed matchup class (e.g., upsets of 12 over 5).

### 3E. Bracket scoring
Implement ESPN's standard scoring (10/20/40/80/160/320 per correct pick by round, or the configurable variant). Report **expected bracket points** — this is the metric the user actually cares about, not raw game-level accuracy.

---

## Phase 4 — The Self-Improvement Loop

The user wants the system to "keep analyzing until it gets great accuracy." Build an automated improvement loop:

### 4A. Weekly retrain workflow
Every Sunday (`weekly_retrain.yml`):
1. Pull latest regular-season data.
2. Rebuild features.
3. Run 50 Optuna trials per model (budget ~2 hours total on GitHub's free runner).
4. Fit the best candidates on all data through the most recent completed season.
5. Evaluate each candidate on the walk-forward backtest.
6. Append results to `experiments/leaderboard.csv`.
7. If a candidate **beats the current champion** on both (a) mean walk-forward log loss and (b) holdout accuracy, and the improvement is **statistically meaningful** (bootstrapped CI, p < 0.05 on paired comparison), promote it to `models/champion/`.
8. Commit `experiments/leaderboard.csv` and `models/champion/` back to the repo via a PR or direct commit.

### 4B. Feature experimentation
Maintain `configs/feature_experiments.yaml` — a list of feature groups the retrain job toggles on/off. Each run records which feature groups were included. Over time, the leaderboard reveals which feature ideas help.

### 4C. Guardrails
- Any candidate that is more than 2× slower to train than the champion is rejected.
- Any candidate whose feature list exceeds the champion's by more than 20% is flagged for review.
- Log every promotion with a diff explaining what changed.

---

## Phase 5 — Tournament-Time Prediction & Discord Delivery

### 5A. Bracket ingestion
When Selection Sunday hits (second Sunday of March), the NCAA publishes the bracket. Two options:
- **Preferred:** Scrape the NCAA's official bracket JSON from `ncaa.com/march-madness-live/bracket` — it's been stable for years.
- **Fallback:** User uploads a `bracket.json` to the repo manually.

The bracket JSON should be normalized to:
```json
{
  "season": 2026,
  "revealed_at": "2026-03-15T18:00:00Z",
  "regions": ["East", "West", "South", "Midwest"],
  "first_four": [...],
  "rounds": {
    "round_of_64": [{"region": "East", "seed_a": 1, "team_a": "...", "seed_b": 16, "team_b": "..."}, ...]
  }
}
```

### 5B. Prediction generation
- For each R64 matchup, predict win probability.
- Then **simulate** the bracket 10,000 times (Monte Carlo), advancing teams stochastically by their predicted probabilities, to compute:
  - Expected win probability for each team to reach each round
  - Most likely Final Four, Championship matchup, Champion
  - Flagged potential upsets (games where probability is 45–55% — toss-ups)
- Also output a **"chalk" bracket** — pick highest probability in every game deterministically — for pool-filling.

### 5C. Discord webhook
The user provides a Discord webhook URL as `DISCORD_WEBHOOK_URL` GitHub Secret. On trigger, post a sequence of well-formatted Discord embed messages:
1. **Pre-tournament:** Full Round-of-64 predictions grouped by region, with win probabilities and confidence tiers (🔒 lock, ✅ likely, ⚠️ toss-up, 💥 upset alert).
2. **Before each round:** Predictions for upcoming games only.
3. **Post-tournament:** Recap showing prediction accuracy.

Use Discord's embed format with color coding (green = favorite win, red = upset pick), and respect the 6000-character total embed limit by splitting into multiple messages if needed.

Webhook sender must:
- Retry with exponential backoff on 429/5xx.
- Never log the webhook URL.
- Include a unique message ID in each post so duplicates are detectable.

---

## Phase 6 — GitHub Actions Workflows

### `daily_data_refresh.yml`
- Schedule: `0 6 * * *` (6 AM UTC daily — after games finish in the US)
- Runs November through April only (check current month in a step, exit early otherwise).
- Pulls new game results, updates team-season stats, commits updated data artifacts to a release.

### `weekly_retrain.yml`
- Schedule: `0 4 * * 0` (Sunday 4 AM UTC)
- Runs the self-improvement loop from Phase 4.
- Uses a matrix strategy to train different model families in parallel jobs.
- Total budget: ~4–5 hours of runner time (well within free-tier limits).

### `tournament_predict.yml`
- Schedule: `0 */4 * * *` during March 15 – April 10 (every 4 hours)
- Detects new bracket state (new round published, or previous round's results in).
- Generates predictions and posts to Discord **only if** the prediction set has changed since the last post (dedupe with a hash committed to `state/last_post_hash.txt`).
- Also triggerable via `workflow_dispatch` for on-demand manual runs.

### `ci.yml`
- On every PR and push: lint, type-check, run full pytest suite.
- Block merge on test failure.

### Secrets required
- `DISCORD_WEBHOOK_URL`
- `KENPOM_USER` and `KENPOM_PASS` (optional)
- `GITHUB_TOKEN` is automatic

---

## Phase 7 — Observability & Debugging

- **Structured logging:** every module uses `structlog` with JSON output.
- **Run manifests:** every GitHub Actions run writes `runs/<timestamp>.json` with inputs, outputs, model version, feature hashes, and metrics.
- **Weekly digest:** the Sunday retrain posts a summary to Discord (a separate webhook ideally) with "this week's top model," leaderboard position changes, and any new features tried.
- **Error alerts:** workflow failures trigger a Discord message to the same webhook so the user sees breakages.

---

## Phase 8 — Testing Requirements

Every module needs tests. At minimum:
- **Feature tests:** assert no target leakage (a feature for the 2019 tournament uses no post-Feb-2019 data).
- **Backtest tests:** asserts walk-forward splits don't overlap.
- **Model tests:** trained model achieves >65% accuracy on a fixed known test slice (sanity floor).
- **Discord tests:** webhook sender formats correctly and handles 429 responses (use `responses` library to mock).
- **Integration test:** full pipeline run end-to-end on a tiny 3-season slice completes in <60 seconds.

Target: 80%+ line coverage.

---

## Implementation Order (Recommended for Claude Code)

Do these in order. Each milestone is a standalone PR that leaves the repo in a working state.

1. **Milestone 1 — Repo skeleton.** `pyproject.toml`, directory structure, CI workflow, empty modules with type stubs, README. Verify `pytest` runs and passes an empty suite.

2. **Milestone 2 — Single data source.** Implement Sports-Reference ingestion for tournament results only, 1985–present. Cache HTML to disk. Write normalized output to `data/interim/tournament_games.parquet`. Tests for the parser.

3. **Milestone 3 — Regular-season results.** Extend Sports-Reference ingester to per-game regular-season results. Same caching discipline.

4. **Milestone 4 — Basic features + baseline model.** Team-season stats, seed difference, simple logistic regression. Walk-forward backtest. **Target: 70% accuracy.** This is the skeleton everything else improves on.

5. **Milestone 5 — Advanced stats ingestion.** Torvik (free), then KenPom if credentials supplied. Four Factors features. Target: 73%.

6. **Milestone 6 — GBM models + tuning.** XGBoost with Optuna. Target: 74%.

7. **Milestone 7 — Matchup and context features.** Style clash, round-specific features, coach experience. Target: 75%.

8. **Milestone 8 — Stacking ensemble.** Target: 75–77%.

9. **Milestone 9 — Discord delivery.** End-to-end test on a past tournament (post 2024 predictions to a test channel).

10. **Milestone 10 — GitHub Actions automation.** All three workflows. Verify on `workflow_dispatch` before scheduling.

11. **Milestone 11 — Self-improvement loop.** Weekly retrain, champion promotion, leaderboard.

12. **Milestone 12 — Pre-1985 historical backfill.** The "vast amount of data" tail. Feature availability gating.

13. **Milestone 13 — Monte Carlo bracket simulation + polished Discord output.**

14. **Milestone 14 — Documentation + runbook.** How to rotate secrets, how to manually override a prediction, how to add a new data source.

---

## A Realistic Note on Accuracy

The user asked for "great accuracy." Here's what is achievable so the expectations are set right:

- **~72%** overall game accuracy is a strong target — this is what FiveThirtyEight's model historically hit, and what sharp Vegas lines imply.
- **~75%** is excellent and beats almost all public models.
- **80%+** is not realistic on the full tournament. The tournament is designed to produce upsets; roughly 20–25% of games are genuine coin-flips where the "better" team loses. A model that claims 85% accuracy on tournament games is almost certainly overfit or leaking.
- Measure success by **log loss** (calibration) and **expected bracket score**, not only by raw accuracy. A well-calibrated 72% model beats a poorly calibrated 76% model in a bracket pool.

Build this as a long-running project. The system will get better every year as more data accumulates and more features are tried.

---

## Deliverables Checklist

- [ ] All 14 milestones implemented and merged
- [ ] README with setup instructions and architecture diagram
- [ ] All three GitHub Actions workflows passing
- [ ] Model champion stored in `models/champion/` with metrics documented
- [ ] Experiment leaderboard with at least 50 recorded runs
- [ ] Discord webhook posts verified on a test channel
- [ ] Test coverage ≥ 80%
- [ ] Runbook documenting secrets, manual overrides, and failure recovery
- [ ] A `BACKTEST_REPORT.md` showing performance broken down by round, seed, and season

---

## Handoff to Claude Code

Claude Code: start with Milestone 1. Open a PR for each milestone. Before implementing any scraper, add the target site's robots.txt check and rate limiter. Before training any model, implement the walk-forward backtest and confirm it cannot leak future data. When in doubt, prefer fewer, more robust features to many brittle ones. Commit `experiments/leaderboard.csv` after every significant training run so the user can see progress over time.
