"""Tests ensuring direct imports from the ai and backend packages work seamlessly."""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from ai.config.loader import get_datasets_config, get_models_config, get_project_config
from ai.evaluation.evaluator import evaluate_binary_classifier
from ai.features.credit_risk import CreditRiskDomainFeatureGenerator
from ai.features.marketing import MarketingDomainFeatureGenerator
from backend.main import app

client = TestClient(app)


def test_ai_config_loader():
    p_cfg = get_project_config()
    d_cfg = get_datasets_config()
    m_cfg = get_models_config()
    assert "project" in p_cfg
    assert "datasets" in d_cfg
    assert len(m_cfg) > 0


def test_ai_feature_transformers():
    df_m = pd.DataFrame({"age": [25, 45], "campaign": [2, 5], "housing": ["yes", "no"], "loan": ["yes", "no"]})
    gen_m = MarketingDomainFeatureGenerator(include_duration=False)
    out_m = gen_m.transform(df_m)
    assert "age_group" in out_m.columns

    df_cr = pd.DataFrame({
        "loan_amnt": [10000.0],
        "person_income": [50000.0],
        "person_emp_length": [2.0],
        "loan_int_rate": [10.0],
        "cb_person_cred_hist_length": [4],
        "person_age": [28],
        "loan_percent_income": [0.2],
    })
    gen_cr = CreditRiskDomainFeatureGenerator()
    out_cr = gen_cr.transform(df_cr)
    assert "loan_to_income_calc" in out_cr.columns


def test_ai_evaluation():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = evaluate_binary_classifier(y_true, y_prob, "test_model")
    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_backend_direct_client():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
