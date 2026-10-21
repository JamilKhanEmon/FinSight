from __future__ import annotations

import json
from typing import Any

from sklearn.model_selection import train_test_split

from ai.config.loader import get_resolved_path
from ai.data.cleaner import DatasetCleaner
from ai.data.ingestion import load_raw_dataset
from ai.evaluation.evaluator import save_experiment_metrics
from ai.features.marketing import build_marketing_preprocessor
from ai.models.trainer import ModelTrainer
from ai.utils.logger import setup_logger

logger = setup_logger("ai.pipelines.marketing")


def run_marketing_pipeline(force_retrain: bool = False) -> dict[str, Any]:
    """Executes bank marketing data preparation, prospective training, and benchmarking."""
    metrics_dir = get_resolved_path("artifacts/metrics")
    prospective_metrics_file = metrics_dir / "marketing_prospective_metrics.json"

    if prospective_metrics_file.exists() and not force_retrain:
        logger.info(f"Loading cached marketing prospective metrics from {prospective_metrics_file}")
        with open(prospective_metrics_file, "r") as f:
            return json.load(f)

    logger.info("Running bank marketing campaign analytics pipeline")

    raw_df = load_raw_dataset("bank_marketing")
    cleaner = DatasetCleaner("bank_marketing")
    cleaned_df, clean_summary = cleaner.clean_bank_marketing(raw_df)

    target_col = "y_target"
    y = cleaned_df[target_col].values
    X = cleaned_df.drop(columns=[target_col, "y"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(
        f"Marketing dataset partition: Train={len(X_train):,} ({y_train.mean():.2%} positive), "
        f"Test={len(X_test):,} ({y_test.mean():.2%} positive)"
    )

    # Prospective pre-call training (duration excluded to prevent leakage)
    logger.info("Training prospective marketing models without call duration")
    trainer = ModelTrainer(task_key="marketing_response", model_prefix="marketing_prospective", random_state=42)
    prospective_results, _ = trainer.train_and_evaluate_all(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        preprocessor_fn=lambda: build_marketing_preprocessor(include_duration=False),
    )
    save_experiment_metrics("marketing_prospective", prospective_results, metrics_dir)

    # Retrospective benchmark (duration included for post-call diagnostic comparison)
    logger.info("Training retrospective benchmark models with call duration included")
    retrospective_trainer = ModelTrainer(task_key="marketing_response", model_prefix="marketing_retrospective", random_state=42)
    retrospective_results, _ = retrospective_trainer.train_and_evaluate_all(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        preprocessor_fn=lambda: build_marketing_preprocessor(include_duration=True),
        models_to_train=["logistic_regression", "xgboost"],
    )
    save_experiment_metrics("marketing_retrospective", retrospective_results, metrics_dir)

    reports_dir = get_resolved_path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "marketing_campaign_report.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Bank Marketing Campaign Intelligence & Modeling Report\n\n")
        f.write("## 1. Executive Summary & Objective\n\n")
        f.write(
            "Prospective identification of banking clients most likely to subscribe "
            "to term deposits during direct outbound telephone campaigns. Prospective models strictly exclude "
            "call duration to ensure zero target leakage.\n\n"
        )
        f.write("## 2. Dataset and Cleaning Summary\n\n")
        f.write(f"- Total records: {clean_summary['initial_rows']:,}\n")
        f.write(f"- Dropped duplicates: {clean_summary['rows_removed']:,}\n")
        f.write(f"- Final analytical rows: {clean_summary['final_rows']:,}\n\n")
        f.write("## 3. Prospective Pre-Call Performance (Duration Excluded)\n\n")
        f.write("| Model | PR-AUC | ROC-AUC | F1 Score | Recall | Precision |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        f.writelines(f"| `{m_name}` | {res['pr_auc']:.4f} | {res['roc_auc']:.4f} | {res['f1_score']:.4f} | {res['recall']:.4f} | {res['precision']:.4f} |\n" for m_name, res in prospective_results.items())

    logger.info(f"Saved bank marketing campaign report to {report_file}")
    return prospective_results
