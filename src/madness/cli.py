"""Typer-based CLI entry point."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from madness.config import (
    CHAMPION_DIR,
    LEADERBOARD_CSV,
    PROCESSED_DIR,
    STATE_DIR,
    Secrets,
    current_season,
    ensure_dirs,
)
from madness.logging_setup import configure, get_logger

app = typer.Typer(help="NCAA March Madness Predictor CLI")
ingest_app = typer.Typer(help="Data ingestion commands")
train_app = typer.Typer(help="Training and tuning")
predict_app = typer.Typer(help="Prediction and bracket generation")
app.add_typer(ingest_app, name="ingest")
app.add_typer(train_app, name="train")
app.add_typer(predict_app, name="predict")

log = get_logger(__name__)


@app.callback()
def _root(verbose: bool = False) -> None:
    configure(level="DEBUG" if verbose else "INFO")
    ensure_dirs()


# ---- ingest ----

@ingest_app.command("tournament")
def ingest_tournament(start: int = 1985, end: int = current_season() - 1, force: bool = False) -> None:
    from madness.ingest.tournament_results import build_canonical_tournament_table
    df = build_canonical_tournament_table(start, end, force=force)
    typer.echo(f"Wrote {len(df)} tournament games for seasons {start}–{end}.")


@ingest_app.command("torvik")
def ingest_torvik(start: int = 2008, end: int = current_season() - 1, force: bool = False) -> None:
    from madness.ingest import torvik
    df = torvik.ingest_range(start, end, force=force)
    typer.echo(f"Wrote {len(df)} Torvik rows for seasons {start}–{end}.")


@ingest_app.command("kenpom")
def ingest_kenpom(start: int = 2002, end: int = current_season() - 1) -> None:
    secrets = Secrets.from_env()
    if not secrets.has_kenpom:
        typer.echo("No KenPom credentials; skipping.")
        raise typer.Exit(code=0)
    from madness.ingest import kenpom
    df = kenpom.ingest_range(start, end)
    typer.echo(f"Wrote {len(df)} KenPom rows for seasons {start}–{end}.")


@ingest_app.command("bootstrap")
def bootstrap(start_season: int = 1985, end_season: int = current_season() - 1) -> None:
    """One-shot historical backfill across all sources."""
    ingest_tournament(start_season, end_season, force=False)
    ingest_torvik(max(2008, start_season), end_season, force=False)
    secrets = Secrets.from_env()
    if secrets.has_kenpom:
        ingest_kenpom(max(2002, start_season), end_season)
    typer.echo("Bootstrap complete.")


# ---- train ----

@train_app.command("baseline")
def train_baseline() -> None:
    """Train the logistic baseline end-to-end from processed features."""
    import pandas as pd
    from madness.models.logistic import LogisticModel
    from madness.models.registry import save_champion
    from madness.train.backtest import walk_forward_backtest

    features_path = PROCESSED_DIR / "features_train.parquet"
    if not features_path.exists():
        typer.echo(f"Missing {features_path}; run feature build first.", err=True)
        raise typer.Exit(code=1)
    df = pd.read_parquet(features_path)
    target = "target"
    feat_cols = [c for c in df.columns if c.startswith(("diff_", "ratio_", "seed_"))]
    feat_cols += [c for c in ("round",) if c in df.columns]

    result = walk_forward_backtest(
        df, target, feat_cols, lambda: LogisticModel(),
    )
    typer.echo(json.dumps(result.summary(), indent=2))

    model = LogisticModel()
    model.fit(df[feat_cols], df[target].to_numpy().astype(int))
    save_champion(model, metrics={
        "mean_accuracy": result.mean_accuracy,
        "mean_log_loss": result.mean_log_loss,
    })


@train_app.command("tune")
def train_tune(model: str = "xgboost", trials: int = 100) -> None:
    import pandas as pd
    from madness.models.gbm import LGBMModel, XGBModel
    from madness.train.tune import TuneConfig, lgbm_space, run_study, xgb_space

    features_path = PROCESSED_DIR / "features_train.parquet"
    df = pd.read_parquet(features_path)
    feat_cols = [c for c in df.columns if c.startswith(("diff_", "ratio_", "seed_"))]

    if model == "xgboost":
        space_fn, factory = xgb_space, lambda p: XGBModel(params=p)
    elif model == "lightgbm":
        space_fn, factory = lgbm_space, lambda p: LGBMModel(params=p)
    else:
        typer.echo(f"Unknown model: {model}", err=True)
        raise typer.Exit(code=1)

    study = run_study(
        df, feat_cols, "target", TuneConfig(model_name=model, n_trials=trials),
        factory, space_fn,
    )
    typer.echo(f"Best log loss: {study.best_value:.4f}")
    typer.echo(f"Best params: {json.dumps(study.best_params, indent=2)}")


@train_app.command("backtest")
def train_backtest(min_train_seasons: int = 10, holdout_last: int = 3) -> None:
    import pandas as pd
    from madness.models.registry import load_champion
    from madness.train.backtest import walk_forward_backtest

    df = pd.read_parquet(PROCESSED_DIR / "features_train.parquet")
    feat_cols = [c for c in df.columns if c.startswith(("diff_", "ratio_", "seed_"))]
    champ = load_champion()
    result = walk_forward_backtest(
        df, "target", feat_cols, lambda: champ,
        min_train_seasons=min_train_seasons, holdout_last=holdout_last,
    )
    typer.echo(json.dumps(result.summary(), indent=2))


# ---- predict ----

@predict_app.command("bracket")
def predict_bracket(
    season: int = current_season(),
    bracket_json: Path = Path("bracket.json"),
    post_discord: bool = False,
) -> None:
    import pandas as pd
    from madness.notify.discord import format_r64_message, send_messages
    from madness.predict.bracket import build_predictions, save_predictions

    if not bracket_json.exists():
        typer.echo(f"No bracket at {bracket_json}", err=True)
        raise typer.Exit(code=1)
    bracket = json.loads(bracket_json.read_text())
    features = pd.read_parquet(PROCESSED_DIR / "features_predict.parquet")
    feat_cols = [c for c in features.columns if c.startswith(("diff_", "ratio_", "seed_"))]

    preds = build_predictions(bracket, features, feat_cols)
    out_path = PROCESSED_DIR / f"predictions_{season}.csv"
    save_predictions(preds, out_path)
    typer.echo(f"Wrote {len(preds)} predictions to {out_path}")

    # Emit predictions JSON for kalshi-safety to consume.
    try:
        from madness.predictions_file import write_predictions_file
        json_path = write_predictions_file(date=None, predictions=preds)
        typer.echo(f"Wrote kalshi predictions JSON to {json_path}")
    except Exception as e:
        typer.echo(f"[kalshi] Failed to write predictions file: {e}")

    if post_discord:
        secrets = Secrets.from_env()
        if not secrets.has_discord:
            typer.echo("No DISCORD_WEBHOOK_URL set; skipping.")
            return
        messages = format_r64_message(preds, season)

        # Idempotency: skip if we already posted this exact prediction set
        state_file = STATE_DIR / "last_post_hash.txt"
        from madness.notify.discord import _predictions_hash
        h = _predictions_hash(preds)
        if state_file.exists() and state_file.read_text().strip() == h:
            typer.echo("Predictions unchanged since last post; skipping Discord.")
            return
        send_messages(secrets.discord_webhook_url, messages)
        state_file.write_text(h)


if __name__ == "__main__":
    app()
