"""Model Training and Pipeline Management Module for FinSight."""

from __future__ import annotations

import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

from finsight.config.loader import get_models_config, get_resolved_path
from finsight.evaluation.evaluator import evaluate_binary_classifier
from finsight.utils.logger import setup_logger

logger = setup_logger("finsight.trainer")


def create_model_instance(model_type: str, params: dict[str, Any]):
    """Instantiate a Scikit-Learn or XGBoost classifier."""
    if model_type == "DummyClassifier":
        return DummyClassifier(**params)
    elif model_type == "LogisticRegression":
        return LogisticRegression(**params)
    elif model_type == "DecisionTreeClassifier":
        return DecisionTreeClassifier(**params)
    elif model_type == "RandomForestClassifier":
        return RandomForestClassifier(**params)
    elif model_type == "XGBClassifier":
        if XGBOOST_AVAILABLE:
            return XGBClassifier(**params)
        logger.warning("XGBoost unavailable. Falling back to HistGradientBoostingClassifier.")
        return HistGradientBoostingClassifier(
            max_iter=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.08),
            random_state=params.get("random_state", 42),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


class ModelTrainer:
    """Manages training, cross-validation, and artifact serialization."""

    def __init__(self, task_key: str, model_prefix: str | None = None, random_state: int = 42):
        self.task_key = task_key
        self.model_prefix = model_prefix or task_key
        self.random_state = random_state
        self.models_cfg = get_models_config().get(task_key, {})

    def train_and_evaluate_all(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        preprocessor_fn,
        models_to_train: list[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Pipeline]]:
        """Train candidate models, evaluate on held-out test set, and serialize artifacts."""
        model_specs = self.models_cfg.get("models", {})
        if models_to_train:
            model_specs = {k: v for k, v in model_specs.items() if k in models_to_train}

        metrics_results: dict[str, Any] = {}
        fitted_pipelines: dict[str, Pipeline] = {}
        artifacts_dir = get_resolved_path("artifacts/models")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        for m_name, m_spec in model_specs.items():
            logger.info(f"[{self.task_key}] Training candidate model: {m_name}...")
            start_time = time.time()

            base_estimator = create_model_instance(m_spec["type"], m_spec["params"])
            preprocessor, _, _ = preprocessor_fn()

            full_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", base_estimator),
            ])

            # Fit on training data only
            full_pipeline.fit(X_train, y_train)
            train_duration = round(time.time() - start_time, 2)

            # Predict probabilities on test set
            if hasattr(full_pipeline, "predict_proba"):
                y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = full_pipeline.predict(X_test)

            eval_metrics = evaluate_binary_classifier(
                y_true=y_test,
                y_pred_proba=y_pred_proba,
                model_name=m_name,
            )
            eval_metrics["train_duration_sec"] = train_duration
            eval_metrics["model_type"] = m_spec["type"]

            metrics_results[m_name] = eval_metrics
            fitted_pipelines[m_name] = full_pipeline

            # Save model artifact
            model_file = artifacts_dir / f"{self.model_prefix}_{m_name}.joblib"
            joblib.dump(full_pipeline, model_file)
            logger.info(
                f"[{self.task_key}] {m_name} completed in {train_duration}s | "
                f"PR-AUC: {eval_metrics['pr_auc']:.4f} | ROC-AUC: {eval_metrics['roc_auc']:.4f} | "
                f"F1: {eval_metrics['f1_score']:.4f}"
            )

        return metrics_results, fitted_pipelines
