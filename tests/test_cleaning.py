"""Unit tests for dataset cleaning operations."""

import pandas as pd

from finsight.cleaning.cleaner import DatasetCleaner


def test_clean_bank_marketing():
    cleaner = DatasetCleaner("bank_marketing")
    sample_df = pd.DataFrame({
        "age": [30, 30, 45],
        "job": ["admin.", "admin.", "technician"],
        "emp.var.rate": [1.1, 1.1, -0.1],
        "cons.price.idx": [93.9, 93.9, 92.8],
        "cons.conf.idx": [-36.4, -36.4, -46.2],
        "euribor3m": [4.8, 4.8, 1.3],
        "nr.employed": [5191.0, 5191.0, 5099.1],
        "y": ["yes", "yes", "no"],
    })

    cleaned_df, summary = cleaner.clean_bank_marketing(sample_df)
    assert len(cleaned_df) == 2  # duplicate dropped
    assert "emp_var_rate" in cleaned_df.columns
    assert "y_target" in cleaned_df.columns
    assert cleaned_df["y_target"].tolist() == [1, 0]
    assert summary["rows_removed"] == 1
