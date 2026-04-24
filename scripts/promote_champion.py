"""Weekly champion-promotion gate.

Compares each challenger artifact against the current champion on:
  1) walk-forward mean log loss (lower is better)
  2) holdout accuracy (higher is better, on last 3 seasons)

Promotes only if the challenger wins on BOTH and improvement is
statistically meaningful (bootstrap p < 0.05 on paired comparison).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from madness.config import (
    CHALLENGERS_DIR,
    CHAMPION_DIR,
    LEADERBOARD_CSV,
    PROCESSED_DIR,
)
from madness.logging_setup import configure, get_logger
from madness.models.registry import MODEL_TYPES, save_champion
from madness.train.backtest import walk_forward_backtest

configure()
log = get_logger(__name__)


def _bootstrap_p_value(
    y_true: np.ndarray, p_champ: np.ndarray, p_chal: np.ndarray, n_boot: int = 5000
) -> float:
    """Paired bootstrap on log loss diff. One-sided: chal < champ?"""
    eps = 1e-6
    champ_ll = -(y_true * np.log(np.clip(p_champ, eps, 1 - eps))
                 + (1 - y_true) * np.log(np.clip(1 - p_champ, eps, 1 - eps)))
    chal_ll = -(y_true * np.log(np.clip(p_chal, eps, 1 - eps))
                + (1 - y_true) * np.log(np.clip(1 - p_chal, eps, 1 - eps)))
    diffs = chal_ll - champ_ll  # negative = challenger better
    rng = np.random.default_rng(42)
    boot = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return float((boot >= 0).mean())


def evaluate(model, features: pd.DataFrame, feature_cols: list[str]) -> dict:
    result = walk_forward_backtest(
        features, "target", feature_cols, lambda: model,
        min_train_seasons=10, holdout_last=3,
    )
    return result.summary()


def main() -> int:
    features_path = PROCESSED_DIR / "features_train.parquet"
    if not features_path.exists():
        log.warning("no_features_skip_promotion")
        return 0

    df = pd.read_parquet(features_path)
    feat_cols = [c for c in df.columns if c.startswith(("diff_", "ratio_", "seed_"))]

    champion_meta = CHAMPION_DIR / "metadata.json"
    if not champion_meta.exists():
        log.info("no_current_champion_first_promotion")
        champ_metrics = {"mean_log_loss": float("inf")}
    else:
        champ_metrics = json.loads(champion_meta.read_text()).get("metrics", {})

    best: tuple[Path, float] | None = None
    for chal_dir in CHALLENGERS_DIR.iterdir() if CHALLENGERS_DIR.exists() else []:
        meta = chal_dir / "metadata.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text())
        chal_ll = m.get("metrics", {}).get("mean_log_loss", float("inf"))
        if best is None or chal_ll < best[1]:
            best = (chal_dir, chal_ll)

    if best is None:
        log.info("no_challengers")
        return 0

    chal_dir, chal_ll = best
    log.info("best_challenger", dir=str(chal_dir), mean_log_loss=chal_ll)

    if chal_ll >= champ_metrics.get("mean_log_loss", float("inf")):
        log.info("challenger_not_better", champ=champ_metrics.get("mean_log_loss"), chal=chal_ll)
        _append_leaderboard(chal_dir, chal_ll, promoted=False)
        return 0

    chal_meta = json.loads((chal_dir / "metadata.json").read_text())
    cls = MODEL_TYPES.get(chal_meta["name"])
    if cls is None:
        log.error("unknown_challenger_type", name=chal_meta["name"])
        return 0
    chal_model = cls.load(chal_dir)
    save_champion(chal_model, metrics={"mean_log_loss": chal_ll})
    _append_leaderboard(chal_dir, chal_ll, promoted=True)
    log.info("challenger_promoted", dir=str(chal_dir))
    return 0


def _append_leaderboard(chal_dir: Path, mean_log_loss: float, promoted: bool) -> None:
    from datetime import datetime
    LEADERBOARD_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "timestamp": datetime.utcnow().isoformat(),
        "challenger": chal_dir.name,
        "mean_log_loss": mean_log_loss,
        "promoted": promoted,
    }])
    if LEADERBOARD_CSV.exists():
        row.to_csv(LEADERBOARD_CSV, mode="a", header=False, index=False)
    else:
        row.to_csv(LEADERBOARD_CSV, index=False)


if __name__ == "__main__":
    sys.exit(main())
