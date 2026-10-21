from __future__ import annotations

import json
from typing import Any

from sklearn.model_selection import train_test_split

from ai.config.loader import get_resolved_path
from ai.data.cleaner import DatasetCleaner
from ai.data.ingestion import load_raw_dataset
from ai.evaluation.evaluator import save_experiment_metrics
from ai.features.credit_risk import build_credit_risk_preprocessor
from ai.models.trainer import ModelTrainer
from ai.utils.logger import setup_logger

logger = setup_logger("ai.pipelines.credit_risk")


def run_credit_risk_pipeline(force_retrain: bool = False) -> dict[str, Any]:
    """Executes Credit Risk underwriting pipeline training and evaluation."""
    metrics_dir = get_resolved_path("artifacts/metrics")
    metrics_file = metrics_dir / "credit_risk_metrics.json"

    if metrics_file.exists() and not force_retrain:
        logger.info(f"Loading cached credit risk metrics from {metrics_file}")
        with open(metrics_file, "r") as f:
            return json.load(f)

    logger.info("Executing credit risk default underwriting pipeline")
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

    reports_dir = get_resolved_path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "credit_risk_report.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Credit Risk & Loan Default Underwriting Report\n\n")
        f.write("## 1. Credit Underwriting Objective\n\n")
        f.write(
            "Predict likelihood of borrower default to balance interest yield against credit loss provisions. "
            "Engineered features incorporate debt-to-income and employment longevity ratios.\n\n"
        )
        f.write("## 2. Model Performance Benchmarks\n\n")
        f.write("| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 Score |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        f.writelines(f"| `{m_name}` | {res['roc_auc']:.4f} | {res['pr_auc']:.4f} | {res['precision']:.4f} | {res['recall']:.4f} | {res['f1_score']:.4f} |\n" for m_name, res in results.items())

    logger.info(f"Saved credit risk report to {report_file}")
    return results
