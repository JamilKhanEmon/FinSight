from __future__ import annotations

from typing import Any

import joblib
from fastapi import HTTPException

from ai.config.loader import get_resolved_path

MODEL_CACHE: dict[str, Any] = {}


def get_model(task_model_key: str):
    """Loads and caches serialized Scikit-learn pipelines."""
    if task_model_key not in MODEL_CACHE:
        model_path = get_resolved_path("artifacts/models") / f"{task_model_key}.joblib"
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model artifact '{task_model_key}' not found at {model_path}. Please execute pipeline first.",
            )
        MODEL_CACHE[task_model_key] = joblib.load(model_path)
    return MODEL_CACHE[task_model_key]


def list_cached_models() -> list[str]:
    """Returns currently loaded model identifiers."""
    return list(MODEL_CACHE.keys())
