from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ai.utils.logger import setup_logger

logger = setup_logger("ai.features.marketing")


class MarketingDomainFeatureGenerator(BaseEstimator, TransformerMixin):
    """Generates prospective domain features for deposit marketing campaigns."""

    def __init__(self, include_duration: bool = False):
        self.include_duration = include_duration

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        if not self.include_duration and "duration" in df.columns:
            df = df.drop(columns=["duration"])

        if "age" in df.columns:
            df["age_group"] = pd.cut(
                df["age"],
                bins=[0, 30, 45, 60, 120],
                labels=["young_adult", "prime_age", "mature", "senior"],
                right=False,
            ).astype(str)

        if "campaign" in df.columns:
            df["campaign_log"] = np.log1p(df["campaign"].clip(lower=0))
            df["high_campaign_contacts"] = (df["campaign"] > 4).astype(int)

        if "pdays" in df.columns:
            df["previously_contacted"] = (df["pdays"] != 999).astype(int)
            df["pdays_binned"] = pd.cut(
                df["pdays"],
                bins=[-1, 7, 14, 30, 998, 1000],
                labels=["contacted_recent", "contacted_medium", "contacted_older", "contacted_long", "never_contacted"],
            ).astype(str)

        if "housing" in df.columns and "loan" in df.columns:
            df["dual_debt_burden"] = ((df["housing"] == "yes") & (df["loan"] == "yes")).astype(int)

        if "euribor3m" in df.columns and "nr_employed" in df.columns:
            denom = (df["nr_employed"] / 1000.0).clip(lower=0.1)
            df["euribor_to_employment"] = (df["euribor3m"] / denom).replace([np.inf, -np.inf], np.nan)

        if "cons_price_idx" in df.columns and "cons_conf_idx" in df.columns:
            df["macro_confidence_index"] = df["cons_price_idx"] + (df["cons_conf_idx"] / 10.0)

        return df.replace([np.inf, -np.inf], np.nan)


def build_marketing_preprocessor(
    include_duration: bool = False,
) -> tuple[Pipeline, list[str], list[str]]:
    """Builds a leakage-free Scikit-Learn preprocessing pipeline for bank marketing."""
    raw_numeric = [
        "age", "campaign", "pdays", "previous",
        "emp_var_rate", "cons_price_idx", "cons_conf_idx", "euribor3m", "nr_employed"
    ]
    if include_duration:
        raw_numeric.append("duration")

    engineered_numeric = [
        "campaign_log", "high_campaign_contacts", "previously_contacted",
        "dual_debt_burden", "euribor_to_employment", "macro_confidence_index"
    ]
    numeric_features = raw_numeric + engineered_numeric

    categorical_features = [
        "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "poutcome", "age_group", "pdays_binned"
    ]

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
        ("domain_fe", MarketingDomainFeatureGenerator(include_duration=include_duration)),
        ("preprocessor", column_preprocessor),
    ])

    return full_pipeline, numeric_features, categorical_features
