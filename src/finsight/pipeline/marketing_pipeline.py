"""End-to-End Pipeline for Bank Marketing Response Prediction."""

from __future__ import annotations

import json
from typing import Any

from sklearn.model_selection import train_test_split

from finsight.cleaning.cleaner import DatasetCleaner
from finsight.config.loader import get_resolved_path
from finsight.evaluation.evaluator import save_experiment_metrics
from finsight.features.marketing import build_marketing_preprocessor
from finsight.ingestion.loader import load_raw_dataset
from finsight.models.trainer import ModelTrainer
from finsight.utils.logger import setup_logger

logger = setup_logger("finsight.pipeline.marketing")


def run_marketing_pipeline(force_retrain: bool = False) -> dict[str, Any]:
    """Execute complete ingestion, cleaning, prospective training, retrospective benchmark, and reporting."""
    metrics_dir = get_resolved_path("artifacts/metrics")
    prospective_metrics_file = metrics_dir / "marketing_prospective_metrics.json"

    if prospective_metrics_file.exists() and not force_retrain:
        logger.info(f"Existing prospective metrics found at {prospective_metrics_file}. Loading cached results.")
        with open(prospective_metrics_file, "r") as f:
            return json.load(f)

    logger.info("=== Starting Bank Marketing Pipeline Execution ===")

    # 1. Ingestion
    raw_df = load_raw_dataset("bank_marketing")

    # 2. Cleaning
    cleaner = DatasetCleaner("bank_marketing")
    cleaned_df, clean_summary = cleaner.clean_bank_marketing(raw_df)

    # 3. Train/Test Stratified Split (80/20)
    target_col = "y_target"
    y = cleaned_df[target_col].values
    X = cleaned_df.drop(columns=[target_col, "y"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(
        f"Train/Test split complete: Train={len(X_train):,} ({y_train.mean():.2%} positive), "
        f"Test={len(X_test):,} ({y_test.mean():.2%} positive)"
    )

    # 4. Prospective Training & Evaluation (duration EXCLUDED)
    logger.info("--- Running PROSPECTIVE Modeling (Pre-call features only, duration excluded) ---")
    trainer = ModelTrainer(task_key="marketing_response", model_prefix="marketing_prospective", random_state=42)

    def prospective_preprocessor_factory():
        return build_marketing_preprocessor(include_duration=False)

    prospective_results, fitted_models = trainer.train_and_evaluate_all(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        preprocessor_fn=prospective_preprocessor_factory,
    )
    save_experiment_metrics("marketing_prospective", prospective_results, metrics_dir)

    # 5. Retrospective Benchmark (duration INCLUDED - explicitly marked as retrospective post-call benchmark)
    logger.info("--- Running RETROSPECTIVE Benchmark (Includes call duration for post-call contrast) ---")
    retrospective_trainer = ModelTrainer(task_key="marketing_response", model_prefix="marketing_retrospective", random_state=42)

    def retrospective_preprocessor_factory():
        return build_marketing_preprocessor(include_duration=True)

    retrospective_results, _ = retrospective_trainer.train_and_evaluate_all(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        preprocessor_fn=retrospective_preprocessor_factory,
        models_to_train=["logistic_regression", "xgboost"],
    )
    save_experiment_metrics("marketing_retrospective", retrospective_results, metrics_dir)

    # 6. Generate Summary Report
    reports_dir = get_resolved_path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "marketing_campaign_report.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Bank Marketing Campaign Intelligence & Modeling Report\n\n")
        f.write("## 1. Executive Summary & Objective\n\n")
        f.write(
            "The primary business goal is prospective identification of banking clients most likely to subscribe "
            "to term deposits during direct outbound telephone campaigns. Prospective models strictly exclude "
            "call duration, as duration cannot be known before a call takes place.\n\n"
        )
        f.write("## 2. Dataset and Cleaning Summary\n\n")
        f.write(f"- Total records: {clean_summary['initial_rows']:,}\n")
        f.write(f"- Duplicate records removed: {clean_summary['rows_removed']}\n")
        f.write(f"- Cleaned records analyzed: {clean_summary['final_rows']:,}\n")
        f.write(f"- Baseline subscription rate: {y.mean():.2%}\n\n")

        f.write("## 3. Prospective Model Comparison (Leakage-Free, Pre-Call Features Only)\n\n")
        f.write("| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | Brier Score |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        f.writelines(f"| `{m_name}` | {res['roc_auc']:.4f} | {res['pr_auc']:.4f} | "
                f"{res['precision']:.4f} | {res['recall']:.4f} | {res['f1_score']:.4f} | "
                f"{res['brier_score']:.4f} |\n" for m_name, res in prospective_results.items())
        f.write("\n")

        f.write("## 4. Retrospective Benchmark (With Post-Call Duration)\n\n")
        f.write("> [!WARNING]\n")
        f.write(
            "> Call duration is a post-call operational attribute and MUST NOT be used for prospective lead ranking. "
            "This retrospective benchmark is presented strictly for diagnostic comparison.\n\n"
        )
        f.write("| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 Score |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        f.writelines(f"| `{m_name} (Retrospective)` | {res['roc_auc']:.4f} | {res['pr_auc']:.4f} | "
                f"{res['precision']:.4f} | {res['recall']:.4f} | {res['f1_score']:.4f} |\n" for m_name, res in retrospective_results.items())
        f.write("\n")

        f.write("## 5. Strategic Business Recommendations\n\n")
        f.write(
            "- **Targeting Strategy**: Prioritize clients with prior campaign success (`poutcome == 'success'`) "
            "and individuals within favorable macroeconomic periods (lower Euribor 3M rate environments).\n"
            "- **Contact Frequency Cap**: Diminishing returns occur sharply after 4 contacts; campaign managers should "
            "cap outbound attempts per client at 4 to reduce customer fatigue and call center labor expenses.\n"
        )

    logger.info(f"Saved marketing campaign report to {report_file}")
    return prospective_results
