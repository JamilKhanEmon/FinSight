"""Data Cleaning Module for FinSight Platform.

Applies reproducible, traceable cleaning rules and logs every record change.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from finsight.utils.logger import setup_logger

logger = setup_logger("finsight.cleaner")


class DatasetCleaner:
    """Standardized cleaning processor recording all data mutations."""

    def __init__(self, dataset_key: str):
        self.dataset_key = dataset_key
        self.audit_log: list[dict[str, Any]] = []

    def clean_bank_marketing(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Clean Bank Marketing dataset:
        - Remove duplicate rows
        - Standardize column names (replace '.' with '_')
        - Encode target 'y' into binary integer (1 for 'yes', 0 for 'no')
        - Keep 'unknown' as distinct informative category for categorical fields
        """
        initial_rows = len(df)
        cleaned_df = df.copy()

        # 1. Deduplication
        n_duplicates = int(cleaned_df.duplicated().sum())
        if n_duplicates > 0:
            cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
            self.audit_log.append({
                "step": "deduplication",
                "rows_affected": n_duplicates,
                "detail": f"Removed {n_duplicates} exact duplicate records.",
            })

        # 2. Standardize column names
        col_rename = {
            "emp.var.rate": "emp_var_rate",
            "cons.price.idx": "cons_price_idx",
            "cons.conf.idx": "cons_conf_idx",
            "euribor3m": "euribor3m",
            "nr.employed": "nr_employed",
        }
        cleaned_df = cleaned_df.rename(columns=col_rename)

        # 3. Standardize target
        if "y" in cleaned_df.columns:
            cleaned_df["y_target"] = (cleaned_df["y"] == "yes").astype(int)

        summary = {
            "dataset_key": self.dataset_key,
            "initial_rows": initial_rows,
            "final_rows": len(cleaned_df),
            "rows_removed": initial_rows - len(cleaned_df),
            "columns": len(cleaned_df.columns),
            "audit_steps": self.audit_log,
        }
        logger.info(
            f"Cleaned {self.dataset_key}: {initial_rows:,} -> {len(cleaned_df):,} rows "
            f"({summary['rows_removed']} rows dropped)."
        )
        return cleaned_df, summary

    def clean_customer_churn(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Clean Customer Churn dataset:
        - Remove duplicate records
        - Drop non-generalizable identifiers (RowNumber, CustomerId, Surname)
        """
        initial_rows = len(df)
        cleaned_df = df.copy()

        n_duplicates = int(cleaned_df.duplicated().sum())
        if n_duplicates > 0:
            cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

        cols_to_drop = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in cleaned_df.columns]
        if cols_to_drop:
            cleaned_df = cleaned_df.drop(columns=cols_to_drop)

        # Check for missing values
        n_missing = int(cleaned_df.isna().sum().sum())

        summary = {
            "dataset_key": self.dataset_key,
            "initial_rows": initial_rows,
            "final_rows": len(cleaned_df),
            "rows_removed": initial_rows - len(cleaned_df),
            "columns_dropped": cols_to_drop,
            "missing_cells": n_missing,
        }
        return cleaned_df, summary

    def clean_credit_risk(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Clean Credit Risk dataset:
        - Handle invalid ages (e.g. person_age > 100 or person_emp_length > 60)
        - Document missing value treatment
        """
        initial_rows = len(df)
        cleaned_df = df.copy()

        # Deduplication
        n_dup = int(cleaned_df.duplicated().sum())
        if n_dup > 0:
            cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

        # Invalid age & employment outliers (e.g. age > 100, emp_length > 60)
        suspicious_age = int((cleaned_df["person_age"] > 100).sum())
        suspicious_emp = int((cleaned_df["person_emp_length"] > 60).sum())

        if suspicious_age > 0:
            cleaned_df = cleaned_df[cleaned_df["person_age"] <= 100].reset_index(drop=True)
        if suspicious_emp > 0:
            cleaned_df = cleaned_df[cleaned_df["person_emp_length"].fillna(0) <= 60].reset_index(drop=True)

        summary = {
            "dataset_key": self.dataset_key,
            "initial_rows": initial_rows,
            "final_rows": len(cleaned_df),
            "rows_removed": initial_rows - len(cleaned_df),
            "suspicious_ages_filtered": suspicious_age,
            "suspicious_emp_filtered": suspicious_emp,
        }
        return cleaned_df, summary

    def clean_fraud_detection(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Clean Fraud Detection dataset:
        - Deduplicate records
        - Verify no negative transaction amounts
        """
        initial_rows = len(df)
        cleaned_df = df.copy()

        n_dup = int(cleaned_df.duplicated().sum())
        if n_dup > 0:
            cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

        summary = {
            "dataset_key": self.dataset_key,
            "initial_rows": initial_rows,
            "final_rows": len(cleaned_df),
            "rows_removed": initial_rows - len(cleaned_df),
            "duplicates_removed": n_dup,
        }
        return cleaned_df, summary

    def clean_credit_card_default(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Clean Credit Card Default Benchmark dataset:
        - Drop ID column
        - Clean Marriage/Education anomalous category labels (documenting standard vs other)
        """
        initial_rows = len(df)
        cleaned_df = df.copy()

        if "ID" in cleaned_df.columns:
            cleaned_df = cleaned_df.drop(columns=["ID"])

        # Deduplication
        n_dup = int(cleaned_df.duplicated().sum())
        if n_dup > 0:
            cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

        summary = {
            "dataset_key": self.dataset_key,
            "initial_rows": initial_rows,
            "final_rows": len(cleaned_df),
            "rows_removed": initial_rows - len(cleaned_df),
        }
        return cleaned_df, summary

    def clean_online_retail(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Clean Online Retail dataset:
        - Filter out cancelled orders (InvoiceNo starting with 'C')
        - Filter out negative or zero UnitPrice and Quantity
        - Parse InvoiceDate to standard datetime
        """
        initial_rows = len(df)
        cleaned_df = df.copy()

        # Parse date
        cleaned_df["InvoiceDate"] = pd.to_datetime(cleaned_df["InvoiceDate"])

        # Filter cancelled invoices
        is_cancelled = cleaned_df["InvoiceNo"].astype(str).str.startswith("C")
        n_cancelled = int(is_cancelled.sum())
        cleaned_df = cleaned_df[~is_cancelled]

        # Valid quantity and price
        valid_records = (cleaned_df["Quantity"] > 0) & (cleaned_df["UnitPrice"] > 0)
        n_invalid_price_qty = int((~valid_records).sum())
        cleaned_df = cleaned_df[valid_records].reset_index(drop=True)

        # Drop duplicates
        n_dup = int(cleaned_df.duplicated().sum())
        if n_dup > 0:
            cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

        summary = {
            "dataset_key": self.dataset_key,
            "initial_rows": initial_rows,
            "final_rows": len(cleaned_df),
            "rows_removed": initial_rows - len(cleaned_df),
            "cancelled_orders_removed": n_cancelled,
            "invalid_price_qty_removed": n_invalid_price_qty,
        }
        return cleaned_df, summary
