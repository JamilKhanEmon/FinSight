from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ChurnDomainFeatureGenerator(BaseEstimator, TransformerMixin):
    """Derives customer retention indicators and behavioral interactions."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        if "Balance" in df.columns and "EstimatedSalary" in df.columns:
            salary_safe = df["EstimatedSalary"].clip(lower=100.0)
            df["balance_to_salary"] = (df["Balance"] / salary_safe).replace([np.inf, -np.inf], np.nan)

        if "CreditScore" in df.columns and "Age" in df.columns:
            age_safe = df["Age"].clip(lower=18.0)
            df["credit_to_age_ratio"] = df["CreditScore"] / age_safe

        if "IsActiveMember" in df.columns and "NumOfProducts" in df.columns:
            df["active_multi_product"] = (
                (df["IsActiveMember"] == 1) & (df["NumOfProducts"] > 1)
            ).astype(int)

        if "Balance" in df.columns:
            df["is_zero_balance"] = (df["Balance"] == 0).astype(int)

        if "Tenure" in df.columns:
            df["is_new_customer"] = (df["Tenure"] <= 1).astype(int)
            df["is_long_tenure"] = (df["Tenure"] >= 7).astype(int)

        return df.replace([np.inf, -np.inf], np.nan)


def build_churn_preprocessor() -> tuple[Pipeline, list[str], list[str]]:
    """Builds Scikit-learn preprocessing pipeline for customer churn."""
    numeric_features = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
        "balance_to_salary", "credit_to_age_ratio", "active_multi_product",
        "is_zero_balance", "is_new_customer", "is_long_tenure",
    ]
    categorical_features = ["Geography", "Gender"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    column_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    full_pipeline = Pipeline([
        ("domain_fe", ChurnDomainFeatureGenerator()),
        ("preprocessor", column_preprocessor),
    ])

    return full_pipeline, numeric_features, categorical_features
