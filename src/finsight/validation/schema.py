"""Data Validation and Quality Audit Module for FinSight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from finsight.config.loader import get_datasets_config, get_resolved_path
from finsight.utils.logger import setup_logger

logger = setup_logger("finsight.validation")


class DataQualityAudit:
    """Performs comprehensive data quality audit on raw or cleaned tabular data."""

    def __init__(self, dataset_key: str, df: pd.DataFrame):
        self.dataset_key = dataset_key
        self.df = df
        self.cfg = get_datasets_config()["datasets"].get(dataset_key, {})
        self.target_col = self.cfg.get("target")

    def run_audit(self) -> dict[str, Any]:
        """Execute full statistical and schema audit."""
        row_count, col_count = self.df.shape
        duplicate_rows = int(self.df.duplicated().sum())

        # Column-level audit
        col_audit = {}
        for col in self.df.columns:
            series = self.df[col]
            missing_count = int(series.isna().sum())
            missing_pct = round((missing_count / row_count) * 100, 2)
            n_unique = int(series.nunique(dropna=True))

            info: dict[str, Any] = {
                "dtype": str(series.dtype),
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "n_unique": n_unique,
            }

            if pd.api.types.is_numeric_dtype(series):
                valid_num = series.dropna()
                if len(valid_num) > 0:
                    info.update({
                        "min": float(valid_num.min()),
                        "max": float(valid_num.max()),
                        "mean": round(float(valid_num.mean()), 4),
                        "std": round(float(valid_num.std()), 4) if len(valid_num) > 1 else 0.0,
                        "median": float(valid_num.median()),
                        "zeros_count": int((valid_num == 0).sum()),
                        "negative_count": int((valid_num < 0).sum()),
                    })
            else:
                top_vals = series.value_counts(dropna=False).head(5).to_dict()
                info["top_categories"] = {str(k): int(v) for k, v in top_vals.items()}

            col_audit[col] = info

        # Target column inspection
        target_summary = {}
        if self.target_col and self.target_col in self.df.columns:
            target_series = self.df[self.target_col]
            target_counts = target_series.value_counts(dropna=False).to_dict()
            target_summary = {
                "target_name": self.target_col,
                "distribution": {str(k): int(v) for k, v in target_counts.items()},
                "class_ratio": {
                    str(k): round(int(v) / row_count, 4) for k, v in target_counts.items()
                },
                "positive_label": str(self.cfg.get("target_positive_label")),
            }

        report = {
            "dataset_key": self.dataset_key,
            "dataset_name": self.cfg.get("name", self.dataset_key),
            "source_url": self.cfg.get("source_url", ""),
            "license": self.cfg.get("license", ""),
            "shape": {"rows": row_count, "columns": col_count},
            "duplicate_rows": duplicate_rows,
            "duplicate_pct": round((duplicate_rows / row_count) * 100, 2) if row_count > 0 else 0,
            "total_missing_cells": int(self.df.isna().sum().sum()),
            "target_summary": target_summary,
            "columns": col_audit,
        }

        return report

    def save_audit(self, output_dir: Path | None = None) -> Path:
        """Run audit and save structured JSON and markdown reports."""
        report = self.run_audit()
        if output_dir is None:
            output_dir = get_resolved_path("artifacts/data_audit")
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / f"{self.dataset_key}_audit.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Markdown summary
        md_path = output_dir / f"{self.dataset_key}_audit.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Data Quality Audit: {report['dataset_name']}\n\n")
            f.write(f"- **Source**: {report['source_url']}\n")
            f.write(f"- **License**: {report['license']}\n")
            f.write(f"- **Dimensions**: {report['shape']['rows']:,} rows × {report['shape']['columns']} columns\n")
            f.write(f"- **Duplicate Rows**: {report['duplicate_rows']:,} ({report['duplicate_pct']}%)\n")
            f.write(f"- **Missing Cells**: {report['total_missing_cells']:,}\n\n")

            if report["target_summary"]:
                f.write("## Target Variable Distribution\n\n")
                f.write(f"**Target Column**: `{report['target_summary']['target_name']}`\n\n")
                f.write("| Class | Count | Proportion |\n")
                f.write("| --- | --- | --- |\n")
                for cls, cnt in report["target_summary"]["distribution"].items():
                    ratio = report["target_summary"]["class_ratio"].get(cls, 0.0)
                    f.write(f"| `{cls}` | {cnt:,} | {ratio:.2%} |\n")
                f.write("\n")

            f.write("## Column-Level Quality Metrics\n\n")
            f.write("| Column Name | Type | Missing (Count) | Missing (%) | Unique Values | Details |\n")
            f.write("| --- | --- | --- | --- | --- | --- |\n")
            for col, meta in report["columns"].items():
                details = ""
                if "mean" in meta:
                    details = f"Mean: {meta['mean']:.2f}, Range: [{meta['min']}, {meta['max']}]"
                elif "top_categories" in meta:
                    top_keys = list(meta["top_categories"].keys())[:3]
                    details = f"Top: {', '.join(top_keys)}"
                f.write(
                    f"| `{col}` | `{meta['dtype']}` | {meta['missing_count']} | {meta['missing_pct']}% | {meta['n_unique']} | {details} |\n"
                )

        logger.info(f"Saved audit reports to {json_path} and {md_path}")
        return json_path


def validate_dataset_schema(dataset_key: str, df: pd.DataFrame) -> bool:
    """Check that expected target and core schema criteria are met."""
    datasets_cfg = get_datasets_config()["datasets"]
    if dataset_key not in datasets_cfg:
        logger.error(f"Dataset {dataset_key} not registered in datasets.yaml")
        return False

    cfg = datasets_cfg[dataset_key]
    target = cfg.get("target")

    if target and target not in df.columns:
        logger.error(f"Validation FAILED: Required target '{target}' missing from {dataset_key} DataFrame.")
        return False

    if len(df) == 0:
        logger.error(f"Validation FAILED: {dataset_key} DataFrame is empty.")
        return False

    logger.info(f"Validation PASSED for {dataset_key} ({len(df):,} rows, {len(df.columns)} columns).")
    return True
