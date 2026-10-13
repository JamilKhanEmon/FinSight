"""Unit tests for the Statistical Feature Engine & Cost-Sensitive Ensemble."""

import numpy as np
import pandas as pd

from finsight.features.statistical_feature_engine import (
    AdaptivePCATransformer,
    DomainFinancialRatioGenerator,
    StatisticalInteractionSelector,
    audit_categorical_chi_squared,
    build_banking_statistical_pipeline,
)
from finsight.models.ensemble_engine import (
    build_cost_sensitive_ensemble,
    calibrate_optimal_threshold,
    compute_banking_diagnostics,
)


def test_domain_financial_ratio_generator():
    df = pd.DataFrame({
        "loan_amnt": [10000.0, 50000.0, 20000.0],
        "person_income": [50000.0, 100000.0, 40000.0],
        "person_emp_length": [2.0, 10.0, 5.0],
        "person_age": [25, 45, 30],
        "Balance": [1000.0, 150000.0, 0.0],
        "EstimatedSalary": [40000.0, 80000.0, 30000.0],
    })
    gen = DomainFinancialRatioGenerator(apply_log_scaling=True)
    gen.fit(df)
    out = gen.transform(df)

    assert "ratio_dti_burden" in out.columns
    assert "ratio_tenure_to_age" in out.columns
    assert "ratio_balance_to_salary" in out.columns
    assert not out.isin([np.inf, -np.inf]).any().any()


def test_statistical_interaction_selector():
    np.random.seed(42)
    df = pd.DataFrame({
        "num_a": np.linspace(1, 100, 50),
        "num_b": np.linspace(2, 200, 50) + np.random.normal(0, 1, 50),
        "num_c": np.random.normal(0, 1, 50),
    })
    y = np.array([0] * 25 + [1] * 25)

    screener = StatisticalInteractionSelector(top_n_interactions=2, min_correlation=0.1)
    screener.fit(df, y)
    out = screener.transform(df)

    assert len(screener.selected_pairs_) > 0
    inter_cols = [c for c in out.columns if c.startswith("inter_")]
    assert len(inter_cols) > 0


def test_audit_categorical_chi_squared():
    df = pd.DataFrame({
        "job": ["admin", "blue-collar", "admin", "services", "admin"] * 20,
        "marital": ["married", "single", "married", "single", "married"] * 20,
        "education": ["university", "basic", "university", "high.school", "university"] * 20,
    })
    results = audit_categorical_chi_squared(df, ["job", "marital", "education"])
    assert len(results) > 0
    assert "chi2_stat" in results[0]
    assert "p_value" in results[0]
    assert "is_statistically_significant" in results[0]


def test_adaptive_pca_transformer():
    np.random.seed(42)
    X = np.random.randn(100, 10)
    pca = AdaptivePCATransformer(variance_threshold=0.95)
    X_trans = pca.fit_transform(X)

    assert pca.n_components_ <= 10
    assert pca.cumulative_variance_ >= 0.90
    assert X_trans.shape[0] == 100
    assert X_trans.shape[1] == pca.n_components_


def test_build_banking_statistical_pipeline():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(60, 6), columns=[f"col_{i}" for i in range(6)])
    pipeline = build_banking_statistical_pipeline(apply_pca=True, variance_threshold=0.95, top_n_interactions=3)
    X_out = pipeline.fit_transform(X)
    assert isinstance(X_out, np.ndarray)
    assert X_out.shape[0] == 60


def test_threshold_calibration_and_diagnostics():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    cal = calibrate_optimal_threshold(y_true, y_proba, cost_fp=1.0, benefit_tp=5.0)
    assert 0.05 <= cal["optimal_f1_threshold"] <= 0.95
    assert cal["max_f1_score"] > 0

    diag = compute_banking_diagnostics(y_true, (y_proba >= 0.5).astype(int), y_proba, "TestModel")
    assert "accuracy" in diag
    assert "sensitivity_recall" in diag
    assert "specificity" in diag
    assert diag["accuracy"] == 1.0


def test_cost_sensitive_ensemble_creation():
    y_train = np.array([0] * 80 + [1] * 20)
    voting, models = build_cost_sensitive_ensemble(y_train)
    assert "RandomForest" in models
    assert "XGBoost" in models
    assert voting.voting == "soft"
