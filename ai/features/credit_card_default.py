from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class CreditCardDefaultFeatureGenerator(BaseEstimator, TransformerMixin):
    """Calculates revolving credit utilization, payment compliance, and billing trends."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        bill_cols = [f"BILL_AMT{i}" for i in range(1, 7) if f"BILL_AMT{i}" in df.columns]
        pay_cols = [f"PAY_AMT{i}" for i in range(1, 7) if f"PAY_AMT{i}" in df.columns]
        pay_status_cols = [c for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] if c in df.columns]

        if bill_cols:
            df["total_billed_6m"] = df[bill_cols].sum(axis=1)
            df["avg_monthly_bill"] = df[bill_cols].mean(axis=1)

        if pay_cols:
            df["total_paid_6m"] = df[pay_cols].sum(axis=1)
            df["avg_monthly_payment"] = df[pay_cols].mean(axis=1)

        if "total_paid_6m" in df.columns and "total_billed_6m" in df.columns:
            billed_safe = df["total_billed_6m"].clip(lower=100.0)
            df["payment_to_bill_ratio"] = (df["total_paid_6m"] / billed_safe).replace([np.inf, -np.inf], np.nan)

        if "LIMIT_BAL" in df.columns and "BILL_AMT1" in df.columns:
            limit_safe = df["LIMIT_BAL"].clip(lower=1000.0)
            df["limit_utilization_recent"] = (df["BILL_AMT1"] / limit_safe).replace([np.inf, -np.inf], np.nan)

        if pay_status_cols:
            df["max_repayment_delay"] = df[pay_status_cols].max(axis=1)
            df["ever_delayed_payment"] = (df["max_repayment_delay"] > 0).astype(int)

        return df.replace([np.inf, -np.inf], np.nan)


def build_credit_card_default_preprocessor() -> tuple[Pipeline, list[str], list[str]]:
    """Builds Scikit-learn preprocessing pipeline for revolving credit default benchmark."""
    raw_numeric = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]
    engineered_numeric = [
        "total_billed_6m", "avg_monthly_bill", "total_paid_6m", "avg_monthly_payment",
        "payment_to_bill_ratio", "limit_utilization_recent", "max_repayment_delay", "ever_delayed_payment"
    ]
    numeric_features = raw_numeric + engineered_numeric

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    column_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )

    full_pipeline = Pipeline([
        ("domain_fe", CreditCardDefaultFeatureGenerator()),
        ("preprocessor", column_preprocessor),
    ])

    return full_pipeline, numeric_features, []
