"""Champion / challenger registry."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from madness.config import CHALLENGERS_DIR, CHAMPION_DIR
from madness.logging_setup import get_logger
from madness.models.gbm import CatBoostModel, LGBMModel, XGBModel
from madness.models.logistic import LogisticModel

log = get_logger(__name__)

MODEL_TYPES = {
    "logistic_l2": LogisticModel,
    "xgboost": XGBModel,
    "lightgbm": LGBMModel,
    "catboost": CatBoostModel,
}


def save_champion(model, metrics: dict) -> None:
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    for existing in CHAMPION_DIR.glob("*"):
        try:
            if existing.is_file():
                existing.unlink()
            else:
                shutil.rmtree(existing, ignore_errors=True)
        except (OSError, PermissionError):
            # On Windows (especially OneDrive) rmtree can race with sync.
            # Not fatal — the new save will overwrite files in place.
            pass
    model.save(CHAMPION_DIR)
    if model.metadata is not None:
        model.metadata.metrics = metrics
        (CHAMPION_DIR / "metadata.json").write_text(
            json.dumps(_dump_metadata(model.metadata), indent=2, default=_json_default)
        )
    log.info("champion_saved", name=model.name, metrics={k: _json_default(v) for k, v in metrics.items()})


def _json_default(o):
    """Fallback encoder for numpy ints/floats/arrays."""
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def save_challenger(model, tag: str, metrics: dict[str, float]) -> Path:
    out = CHALLENGERS_DIR / tag
    out.mkdir(parents=True, exist_ok=True)
    model.save(out)
    if model.metadata is not None:
        model.metadata.metrics = metrics
        (out / "metadata.json").write_text(
            json.dumps(_dump_metadata(model.metadata), indent=2)
        )
    return out


def _dump_metadata(meta) -> dict:
    from dataclasses import asdict
    return asdict(meta)


def load_champion():
    meta_path = CHAMPION_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError("No champion model found; train one first.")
    meta = json.loads(meta_path.read_text())
    cls = MODEL_TYPES.get(meta["name"])
    if cls is None:
        raise ValueError(f"Unknown champion model type: {meta['name']}")
    return cls.load(CHAMPION_DIR)
