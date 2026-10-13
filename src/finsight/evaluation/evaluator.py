"""Evaluation and Metrics Assessment Module for Classification and Risk Models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from finsight.utils.logger import setup_logger

logger = setup_logger("finsight.evaluation")


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate comprehensive, balanced evaluation metrics for a binary classifier."""
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Standard metrics
    roc_auc = float(roc_auc_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_pred_proba))

    cm = confusion_matrix(y_true, y_pred).tolist()

    # Threshold curve analysis
    threshold_analysis = []
    for th in np.arange(0.1, 0.95, 0.05):
        th_val = round(float(th), 2)
        th_pred = (y_pred_proba >= th_val).astype(int)
        threshold_analysis.append({
            "threshold": th_val,
            "precision": round(float(precision_score(y_true, th_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, th_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, th_pred, zero_division=0)), 4),
        })

    metrics = {
        "model_name": model_name,
        "decision_threshold": threshold,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm,
        "sample_size": len(y_true),
        "positive_rate": round(float(np.mean(y_true)), 4),
        "threshold_analysis": threshold_analysis,
    }

    return metrics


def save_experiment_metrics(task_name: str, results: dict[str, Any], output_dir: Path) -> Path:
    """Save metrics JSON and human-readable summary table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{task_name}_metrics.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved evaluation metrics to {json_path}")
    return json_path
