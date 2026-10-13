"""Dataset Audit CLI Runner for FinSight Platform."""

from __future__ import annotations

import argparse

from finsight.config.loader import get_datasets_config
from finsight.ingestion.loader import load_raw_dataset
from finsight.utils.logger import setup_logger
from finsight.validation.schema import DataQualityAudit, validate_dataset_schema

logger = setup_logger("finsight.audit_runner")


def audit_single_dataset(dataset_key: str, sample_rows: int | None = None) -> bool:
    """Run data quality audit for a specific dataset."""
    logger.info(f"Auditing dataset: {dataset_key}...")
    try:
        df = load_raw_dataset(dataset_key, nrows=sample_rows)
        is_valid = validate_dataset_schema(dataset_key, df)
        if not is_valid:
            logger.error(f"Schema validation failed for {dataset_key}")
            return False

        auditor = DataQualityAudit(dataset_key, df)
        auditor.save_audit()
        logger.info(f"Successfully audited {dataset_key}")
        return True
    except Exception as exc:
        logger.error(f"Error auditing {dataset_key}: {exc}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="FinSight Data Quality Audit")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset key or 'all'")
    parser.add_argument("--sample", type=int, default=None, help="Sample rows (optional for fast run)")
    args = parser.parse_args()

    cfg = get_datasets_config()["datasets"]
    targets = list(cfg.keys()) if args.dataset == "all" else [args.dataset]

    results = {}
    for key in targets:
        logger.info("\n==========================================")
        logger.info(f"AUDITING: {key}")
        logger.info("==========================================")
        ok = audit_single_dataset(key, sample_rows=args.sample)
        results[key] = "PASSED" if ok else "FAILED"

    logger.info("\n=== DATA AUDIT SUMMARY ===")
    for k, status in results.items():
        logger.info(f"  {k:32s}: {status}")


if __name__ == "__main__":
    main()
