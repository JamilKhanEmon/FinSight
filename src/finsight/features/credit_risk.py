"""Feature Engineering Pipeline for Credit Risk & Loan Default Prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CreditRiskDomainFeatureGenerator(BaseEstimator, TransformerMixin):
    """Derives banking solvency, leverage, and repayment feasibility ratios."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Solvency & Debt Burden: Loan amount to annual income
        if "loan_amnt" in df.columns and "person_income" in df.columns:
            income_safe = df["person_income"].clip(lower=1000.0)
            df["loan_to_income_calc"] = (df["loan_amnt"] / income_safe).replace([np.inf, -np.inf], np.nan)

        # 2. Employment Stability Indicator: Loan amount per year of employment
        if "loan_amnt" in df.columns and "person_emp_length" in df.columns:
            emp_safe = (df["person_emp_length"].fillna(0) + 1.0)
            df["loan_per_emp_year"] = (df["loan_amnt"] / emp_safe).replace([np.inf, -np.inf], np.nan)

        # 3. Interest rate cost burden: approximate annual interest expense
        if "loan_amnt" in df.columns and "loan_int_rate" in df.columns:
            df["annual_interest_burden"] = (df["loan_amnt"] * (df["loan_int_rate"] / 100.0)).replace([np.inf, -np.inf], np.nan)

        # 4. Credit history to age ratio (proportion of adult life with credit)
        if "cb_person_cred_hist_length" in df.columns and "person_age" in df.columns:
            adult_age_safe = (df["person_age"] - 18.0).clip(lower=1.0)
            df["cred_hist_to_adult_age"] = (
                df["cb_person_cred_hist_length"] / adult_age_safe
            ).clip(upper=1.0).replace([np.inf, -np.inf], np.nan)

        # 5. High-risk flag interaction: low income + high loan percentage
        if "loan_percent_income" in df.columns and "person_income" in df.columns:
            df["high_dti_vulnerability"] = (
                (df["loan_percent_income"] > 0.35) & (df["person_income"] < 40000.0)
            ).astype(int)

        df = df.replace([np.inf, -np.inf], np.nan)
        return df


def build_credit_risk_preprocessor() -> tuple[Pipeline, list[str], list[str]]:
    """Builds Scikit-learn preprocessing pipeline for credit risk."""
    numeric_features = [
        "person_age", "person_income", "person_emp_length", "loan_amnt",
        "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
        "loan_to_income_calc", "loan_per_emp_year", "annual_interest_burden",
        "cred_hist_to_adult_age", "high_dti_vulnerability"
    ]
    categorical_features = [
        "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"
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
        ("domain_fe", CreditRiskDomainFeatureGenerator()),
        ("preprocessor", column_preprocessor),
    ])

    return full_pipeline, numeric_features, categorical_features
