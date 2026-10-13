"""Integration tests for FastAPI local inference microservice."""

from fastapi.testclient import TestClient

from finsight.api.service import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "FinSight" in data["platform"]


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_api_models():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert len(data["available_models"]) > 0


def test_predict_marketing_endpoint():
    payload = {
        "age": 42,
        "job": "management",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "no",
        "loan": "no",
        "contact": "cellular",
        "month": "aug",
        "day_of_week": "thu",
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp_var_rate": -0.1,
        "cons_price_idx": 93.2,
        "cons_conf_idx": -42.0,
        "euribor3m": 1.4,
        "nr_employed": 5099.1,
    }
    response = client.post("/predict/marketing", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "lead_conversion_probability" in data
    assert "disclaimer" in data
    assert 0.0 <= data["lead_conversion_probability"] <= 1.0


def test_predict_credit_risk_endpoint():
    payload = {
        "person_age": 25,
        "person_income": 45000.0,
        "person_home_ownership": "RENT",
        "person_emp_length": 2.0,
        "loan_intent": "EDUCATION",
        "loan_grade": "A",
        "loan_amnt": 5000.0,
        "loan_int_rate": 7.5,
        "loan_percent_income": 0.11,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3,
    }
    response = client.post("/predict/credit_risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert "risk_tier" in data
    assert data["risk_tier"] in ["Low Risk", "Medium Risk", "High Risk"]
