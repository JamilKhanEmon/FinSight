"""End-to-End Pipeline for Credit Risk & Loan Default Prediction."""

from __future__ import annotations

import json
from typing import Any

from sklearn.model_selection import train_test_split

from finsight.cleaning.cleaner import DatasetCleaner
from finsight.config.loader import get_resolved_path
from finsight.evaluation.evaluator import save_experiment_metrics
from finsight.features.credit_risk import build_credit_risk_preprocessor
from finsight.ingestion.loader import load_raw_dataset
from finsight.models.trainer import ModelTrainer
from finsight.utils.logger import setup_logger

logger = setup_logger("finsight.pipeline.credit_risk")


def run_credit_risk_pipeline(force_retrain: bool = False) -> dict[str, Any]:
    """Execute Credit Risk pipeline training, tuning, and evaluation."""
    metrics_dir = get_resolved_path("artifacts/metrics")
    metrics_file = metrics_dir / "credit_risk_metrics.json"

    if metrics_file.exists() and not force_retrain:
        logger.info(f"Existing credit risk metrics found at {metrics_file}. Loading cached results.")
        with open(metrics_file, "r") as f:
            return json.load(f)

    logger.info("=== Starting Credit Risk Pipeline Execution ===")
    raw_df = load_raw_dataset("credit_risk")

    cleaner = DatasetCleaner("credit_risk")
    cleaned_df, clean_summary = cleaner.clean_credit_risk(raw_df)

    target_col = "loan_status"
    y = cleaned_df[target_col].values
    X = cleaned_df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    trainer = ModelTrainer(task_key="credit_risk", random_state=42)
    results, _ = trainer.train_and_evaluate_all(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        preprocessor_fn=build_credit_risk_preprocessor,
    )

    save_experiment_metrics("credit_risk", results, metrics_dir)

    # Save summary report
    reports_dir = get_resolved_path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "credit_risk_report.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Credit Risk & Default Prediction Report\n\n")
        f.write("## 1. Credit Underwriting Objective\n\n")
        f.write(
            "Predicting loan applicant default risk is central to capital adequacy and credit loss provisioning. "
            "This module evaluates applicant credit capacity using income ratios, interest burdens, and credit histories.\n\n"
        )
        f.write("## 2. Model Performance Benchmarks\n\n")
        f.write("| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 Score |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        f.writelines(f"| `{m_name}` | {res['roc_auc']:.4f} | {res['pr_auc']:.4f} | "
                f"{res['precision']:.4f} | {res['recall']:.4f} | {res['f1_score']:.4f} |\n" for m_name, res in results.items())

    logger.info(f"Saved credit risk report to {report_file}")
    return results
