"""Unified CLI Pipeline Runner for FinSight Platform.

Executes end-to-end data pipelines, model training, evaluation, and report generation.
Supports selective execution and --force-retrain flags.
"""

from __future__ import annotations

import argparse
import sys
import time

from finsight.analytics.retail_analytics import run_retail_pipeline
from finsight.experiments.research_experiment import run_baseline_vs_engineered_experiment
from finsight.pipeline.churn_pipeline import run_churn_pipeline
from finsight.pipeline.credit_card_default_pipeline import run_credit_card_default_pipeline
from finsight.pipeline.credit_risk_pipeline import run_credit_risk_pipeline
from finsight.pipeline.fraud_pipeline import run_fraud_pipeline
from finsight.pipeline.marketing_pipeline import run_marketing_pipeline
from finsight.utils.logger import setup_logger

logger = setup_logger("finsight.pipeline_runner")

TASK_RUNNERS = {
    "marketing": ("Bank Marketing Campaign Analytics", run_marketing_pipeline),
    "churn": ("Customer Churn Prediction", run_churn_pipeline),
    "credit_risk": ("Credit Risk Default Underwriting", run_credit_risk_pipeline),
    "fraud": ("Transaction Fraud Detection", run_fraud_pipeline),
    "benchmark": ("Credit Card Default Benchmark (UCI 350)", run_credit_card_default_pipeline),
    "retail": ("Online Retail Analytics & Time-Series Forecasting", run_retail_pipeline),
    "research": ("Research Feature Engineering Experiment", run_baseline_vs_engineered_experiment),
}


def main():
    parser = argparse.ArgumentParser(description="FinSight Unified Pipeline Runner")
    parser.add_argument(
        "--all", action="store_true", help="Execute all analytical and model pipelines"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=list(TASK_RUNNERS.keys()),
        help=f"Select individual task to execute: {list(TASK_RUNNERS.keys())}",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force re-execution and retraining even if artifacts exist",
    )
    args = parser.parse_args()

    if not args.all and not args.task:
        parser.print_help()
        sys.exit(1)

    tasks_to_run = list(TASK_RUNNERS.keys()) if args.all else [args.task]

    logger.info("=================================================================")
    logger.info("   FinSight — Banking Intelligence & Customer Analytics Platform")
    logger.info(f"   Tasks to execute: {', '.join(tasks_to_run)}")
    logger.info(f"   Force retrain: {args.force_retrain}")
    logger.info("=================================================================\n")

    summary_records = {}
    overall_start = time.time()

    for task_key in tasks_to_run:
        task_name, runner_func = TASK_RUNNERS[task_key]
        logger.info(f"\n>>> RUNNING TASK: [{task_key.upper()}] {task_name} <<<")
        task_start = time.time()

        try:
            if task_key == "research":
                result = runner_func()
            else:
                result = runner_func(force_retrain=args.force_retrain)
            elapsed = round(time.time() - task_start, 2)
            summary_records[task_key] = {
                "name": task_name,
                "status": "COMPLETED",
                "duration_sec": elapsed,
                "error": None,
            }
            logger.info(f">>> TASK [{task_key.upper()}] COMPLETED in {elapsed}s <<<\n")
        except Exception as exc:
            elapsed = round(time.time() - task_start, 2)
            logger.error(f">>> TASK [{task_key.upper()}] FAILED after {elapsed}s: {exc}", exc_info=True)
            summary_records[task_key] = {
                "name": task_name,
                "status": "FAILED",
                "duration_sec": elapsed,
                "error": str(exc),
            }

    overall_elapsed = round(time.time() - overall_start, 2)

    logger.info("\n" + "=" * 70)
    logger.info("                      FINSIGHT PIPELINE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Task Key':<15} | {'Status':<10} | {'Duration (s)':<12} | {'Task Name'}")
    logger.info("-" * 70)
    for k, rec in summary_records.items():
        logger.info(f"{k:<15} | {rec['status']:<10} | {rec['duration_sec']:<12} | {rec['name']}")
    logger.info("=" * 70)
    logger.info(f"Total Execution Time: {overall_elapsed}s\n")


if __name__ == "__main__":
    main()
