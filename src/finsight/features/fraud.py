"""Feature Engineering Pipeline for Transaction Fraud Detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


class FraudDomainFeatureGenerator(BaseEstimator, TransformerMixin):
    """Transforms transaction amounts, time periodicity, and transaction velocity indicators."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Log-transformed Amount (handles heavy right skew)
        if "Amount" in df.columns:
            df["log_amount"] = np.log1p(df["Amount"].clip(lower=0.0))
            df["is_zero_amount"] = (df["Amount"] == 0).astype(int)
            df["is_high_value_tx"] = (df["Amount"] > 1000.0).astype(int)

        # 2. Time cyclicity (Time is in seconds from first observation)
        if "Time" in df.columns:
            hours = (df["Time"] / 3600.0) % 24.0
            df["tx_hour"] = hours
            df["tx_sin_time"] = np.sin(2 * np.pi * hours / 24.0)
            df["tx_cos_time"] = np.cos(2 * np.pi * hours / 24.0)
            # Late night / early morning risk indicator (midnight to 5 AM)
            df["is_night_tx"] = ((hours >= 0.0) & (hours <= 5.0)).astype(int)

        df = df.replace([np.inf, -np.inf], np.nan)
        return df


def build_fraud_preprocessor() -> tuple[Pipeline, list[str], list[str]]:
    """Builds Scikit-learn preprocessing pipeline for fraud detection."""
    v_cols = [f"V{i}" for i in range(1, 29)]
    engineered_numeric = [
        "Amount", "log_amount", "is_zero_amount", "is_high_value_tx",
        "tx_hour", "tx_sin_time", "tx_cos_time", "is_night_tx"
    ]
    numeric_features = v_cols + engineered_numeric

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    column_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )

    full_pipeline = Pipeline([
        ("domain_fe", FraudDomainFeatureGenerator()),
        ("preprocessor", column_preprocessor),
    ])

    return full_pipeline, numeric_features, []
