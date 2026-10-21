from __future__ import annotations

import argparse
import sys
import time

from ai.analytics.retail_analytics import run_retail_pipeline
from ai.pipelines.churn_pipeline import run_churn_pipeline
from ai.pipelines.credit_card_default_pipeline import run_credit_card_default_pipeline
from ai.pipelines.credit_risk_pipeline import run_credit_risk_pipeline
from ai.pipelines.fraud_pipeline import run_fraud_pipeline
from ai.pipelines.marketing_pipeline import run_marketing_pipeline
from ai.utils.logger import setup_logger
from research.experiments.statistical_banking_study import run_full_banking_statistical_study

logger = setup_logger("scripts.run_pipeline")

TASK_RUNNERS = {
    "marketing": ("Bank Marketing Campaign Analytics", run_marketing_pipeline),
    "churn": ("Customer Churn Prediction", run_churn_pipeline),
    "credit_risk": ("Credit Risk Default Underwriting", run_credit_risk_pipeline),
    "fraud": ("Transaction Fraud Detection", run_fraud_pipeline),
    "benchmark": ("Credit Card Default Benchmark (UCI 350)", run_credit_card_default_pipeline),
    "retail": ("Online Retail Analytics & Time-Series Forecasting", run_retail_pipeline),
    "research": ("Statistical Intelligence & Adaptive Ensemble Study", run_full_banking_statistical_study),
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

    logger.info("FinSight Banking Intelligence Platform Pipeline Runner")
    logger.info(f"Tasks scheduled: {', '.join(tasks_to_run)} | Force retrain: {args.force_retrain}")

    summary_records = {}
    overall_start = time.time()

    for task_key in tasks_to_run:
        task_name, runner_func = TASK_RUNNERS[task_key]
        logger.info(f"Running task [{task_key}]: {task_name}")
        task_start = time.time()

        try:
            if task_key == "research":
                runner_func()
            else:
                runner_func(force_retrain=args.force_retrain)
            elapsed = round(time.time() - task_start, 2)
            summary_records[task_key] = {
                "name": task_name,
                "status": "COMPLETED",
                "duration_sec": elapsed,
                "error": None,
            }
            logger.info(f"Task [{task_key}] completed in {elapsed}s")
        except Exception as exc:
            elapsed = round(time.time() - task_start, 2)
            logger.error(f"Task [{task_key}] failed after {elapsed}s: {exc}", exc_info=True)
            summary_records[task_key] = {
                "name": task_name,
                "status": "FAILED",
                "duration_sec": elapsed,
                "error": str(exc),
            }

    overall_elapsed = round(time.time() - overall_start, 2)

    logger.info("FinSight Pipeline Execution Summary:")
    for k, rec in summary_records.items():
        logger.info(f"  {k:15s} | {rec['status']:10s} | {rec['duration_sec']:>6.2f}s | {rec['name']}")
    logger.info(f"Total execution time: {overall_elapsed}s")


if __name__ == "__main__":
    main()
