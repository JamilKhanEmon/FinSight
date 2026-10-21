"""Unit tests for domain feature generators across tasks."""

import pandas as pd

from ai.features.churn import ChurnDomainFeatureGenerator
from ai.features.credit_risk import CreditRiskDomainFeatureGenerator
from ai.features.fraud import FraudDomainFeatureGenerator


def test_churn_feature_generator():
    df = pd.DataFrame({
        "Balance": [0.0, 100000.0],
        "EstimatedSalary": [50000.0, 100000.0],
        "CreditScore": [600, 750],
        "Age": [30, 45],
        "IsActiveMember": [1, 1],
        "NumOfProducts": [1, 2],
        "Tenure": [1, 8],
    })
    gen = ChurnDomainFeatureGenerator()
    out = gen.transform(df)
    assert "balance_to_salary" in out.columns
    assert "is_zero_balance" in out.columns
    assert out["is_zero_balance"].tolist() == [1, 0]
    assert out["active_multi_product"].tolist() == [0, 1]


def test_credit_risk_feature_generator():
    df = pd.DataFrame({
        "loan_amnt": [10000.0, 20000.0],
        "person_income": [50000.0, 80000.0],
        "person_emp_length": [3.0, 5.0],
        "loan_int_rate": [10.0, 15.0],
        "cb_person_cred_hist_length": [5, 10],
        "person_age": [28, 40],
        "loan_percent_income": [0.20, 0.40],
    })
    gen = CreditRiskDomainFeatureGenerator()
    out = gen.transform(df)
    assert "loan_to_income_calc" in out.columns
    assert "annual_interest_burden" in out.columns
    assert "high_dti_vulnerability" in out.columns


def test_fraud_feature_generator():
    df = pd.DataFrame({
        "Amount": [0.0, 1500.0],
        "Time": [3600.0, 72000.0],  # 1 hour and 20 hours
    })
    gen = FraudDomainFeatureGenerator()
    out = gen.transform(df)
    assert "log_amount" in out.columns
    assert "is_high_value_tx" in out.columns
    assert out["is_high_value_tx"].tolist() == [0, 1]
    assert "is_night_tx" in out.columns
