from __future__ import annotations

import json
from typing import Any

from sklearn.model_selection import train_test_split

from ai.config.loader import get_resolved_path
from ai.data.cleaner import DatasetCleaner
from ai.data.ingestion import load_raw_dataset
from ai.evaluation.evaluator import save_experiment_metrics
from ai.features.fraud import build_fraud_preprocessor
from ai.models.trainer import ModelTrainer
from ai.utils.logger import setup_logger

logger = setup_logger("ai.pipelines.fraud")


def run_fraud_pipeline(force_retrain: bool = False, max_train_samples: int | None = 100000) -> dict[str, Any]:
    """Executes Fraud Detection pipeline with precision-recall and threshold optimization."""
    metrics_dir = get_resolved_path("artifacts/metrics")
    metrics_file = metrics_dir / "fraud_metrics.json"

    if metrics_file.exists() and not force_retrain:
        logger.info(f"Loading cached fraud metrics from {metrics_file}")
        with open(metrics_file, "r") as f:
            return json.load(f)

    logger.info("Executing transaction fraud screening pipeline")
    raw_df = load_raw_dataset("fraud_detection")

    cleaner = DatasetCleaner("fraud_detection")
    cleaned_df, clean_summary = cleaner.clean_fraud_detection(raw_df)

    target_col = "Class"
    y = cleaned_df[target_col].values
    X = cleaned_df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    if max_train_samples and len(X_train) > max_train_samples:
        logger.info(f"Subsampling training set to {max_train_samples:,} rows while preserving full held-out test set.")
        train_idx, _ = train_test_split(
            range(len(X_train)),
            train_size=max_train_samples,
            stratify=y_train,
            random_state=42,
        )
        X_train = X_train.iloc[train_idx].reset_index(drop=True)
        y_train = y_train[train_idx]

    logger.info(
        f"Fraud dataset partition: Train={len(X_train):,} ({y_train.sum()} frauds, {y_train.mean():.3%}), "
        f"Test={len(X_test):,} ({y_test.sum()} frauds, {y_test.mean():.3%})"
    )

    trainer = ModelTrainer(task_key="fraud_detection", random_state=42)
    results, _ = trainer.train_and_evaluate_all(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        preprocessor_fn=build_fraud_preprocessor,
        models_to_train=["dummy", "logistic_regression", "xgboost"],
    )

    save_experiment_metrics("fraud", results, metrics_dir)

    reports_dir = get_resolved_path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "fraud_detection_report.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Credit Card Transaction Fraud Detection Report\n\n")
        f.write("## 1. Class Imbalance Context & Business Objective\n\n")
        f.write(
            "Transaction fraud represents an extreme class imbalance scenario (0.17% positive rate). "
            "ROC-AUC can present an overly optimistic assessment due to the vast majority of legitimate transactions. "
            "Model evaluation prioritizes PR-AUC and false-positive cost analysis.\n\n"
        )
        f.write("## 2. Model Performance Benchmarks\n\n")
        f.write("| Model | PR-AUC | ROC-AUC | F1 Score | Recall | Precision |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        f.writelines(f"| `{m_name}` | {res['pr_auc']:.4f} | {res['roc_auc']:.4f} | {res['f1_score']:.4f} | {res['recall']:.4f} | {res['precision']:.4f} |\n" for m_name, res in results.items())

    logger.info(f"Saved fraud detection report to {report_file}")
    return results
