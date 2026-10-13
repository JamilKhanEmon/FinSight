"""FastAPI Local Prediction Service for FinSight Banking Intelligence Platform."""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from finsight.config.loader import get_resolved_path

app = FastAPI(
    title="FinSight Banking Intelligence API",
    description="Local prediction service for prospective marketing response, credit risk, and churn probability.",
    version="0.1.0",
)

MODEL_CACHE: dict[str, Any] = {}


def get_model(task_model_key: str):
    """Load cached model pipeline from artifacts/models."""
    if task_model_key not in MODEL_CACHE:
        model_path = get_resolved_path("artifacts/models") / f"{task_model_key}.joblib"
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model artifact '{task_model_key}' not found at {model_path}. Please run pipeline first.",
            )
        MODEL_CACHE[task_model_key] = joblib.load(model_path)
    return MODEL_CACHE[task_model_key]


# Pydantic Input Schemas
class MarketingLeadInput(BaseModel):
    age: int = Field(..., ge=18, le=100, example=35)
    job: str = Field(..., example="admin.")
    marital: str = Field(..., example="married")
    education: str = Field(..., example="university.degree")
    default: str = Field("no", example="no")
    housing: str = Field("yes", example="yes")
    loan: str = Field("no", example="no")
    contact: str = Field("cellular", example="cellular")
    month: str = Field("may", example="may")
    day_of_week: str = Field("mon", example="mon")
    campaign: int = Field(1, ge=1, le=50, example=1)
    pdays: int = Field(999, ge=0, le=999, example=999)
    previous: int = Field(0, ge=0, le=20, example=0)
    poutcome: str = Field("nonexistent", example="nonexistent")
    emp_var_rate: float = Field(1.1, example=1.1)
    cons_price_idx: float = Field(93.994, example=93.994)
    cons_conf_idx: float = Field(-36.4, example=-36.4)
    euribor3m: float = Field(4.857, example=4.857)
    nr_employed: float = Field(5191.0, example=5191.0)


class CreditRiskInput(BaseModel):
    person_age: int = Field(..., ge=18, le=100, example=28)
    person_income: float = Field(..., ge=0, example=65000.0)
    person_home_ownership: str = Field(..., example="RENT")
    person_emp_length: float = Field(2.0, ge=0, le=60, example=3.0)
    loan_intent: str = Field(..., example="PERSONAL")
    loan_grade: str = Field(..., example="B")
    loan_amnt: float = Field(..., ge=500, example=10000.0)
    loan_int_rate: float = Field(11.14, ge=1.0, le=40.0, example=11.14)
    loan_percent_income: float = Field(..., ge=0.0, le=1.0, example=0.15)
    cb_person_default_on_file: str = Field("N", example="N")
    cb_person_cred_hist_length: int = Field(4, ge=0, le=50, example=4)


class ChurnInput(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850, example=650)
    Geography: str = Field("France", example="France")
    Gender: str = Field("Female", example="Female")
    Age: int = Field(..., ge=18, le=100, example=42)
    Tenure: int = Field(..., ge=0, le=10, example=5)
    Balance: float = Field(..., ge=0, example=75000.0)
    NumOfProducts: int = Field(..., ge=1, le=4, example=1)
    HasCrCard: int = Field(1, ge=0, le=1, example=1)
    IsActiveMember: int = Field(1, ge=0, le=1, example=1)
    EstimatedSalary: float = Field(..., ge=0, example=60000.0)


DISCLAIMER = (
    "Model predictions reflect probabilistic estimates based on historical statistical patterns. "
    "Predictions do not establish causality and must be combined with human banking judgment."
)


@app.get("/")
def read_root():
    return {
        "platform": "FinSight Banking Intelligence API",
        "version": "0.1.0",
        "status": "operational",
        "endpoints": ["/health", "/models", "/predict/marketing", "/predict/credit_risk", "/predict/churn"],
    }


@app.get("/health")
def read_health():
    return {"status": "healthy", "service": "finsight-api"}


@app.get("/models")
def list_models():
    models_dir = get_resolved_path("artifacts/models")
    available_files = [f.stem for f in models_dir.glob("*.joblib")]
    return {
        "available_models": available_files,
        "default_active_models": {
            "marketing": "marketing_response_xgboost",
            "credit_risk": "credit_risk_xgboost",
            "churn": "customer_churn_random_forest",
        },
    }


@app.post("/predict/marketing")
def predict_marketing(lead: MarketingLeadInput):
    model = get_model("marketing_prospective_xgboost")
    df = pd.DataFrame([lead.model_dump()])

    proba = float(model.predict_proba(df)[0, 1])
    is_promising = bool(proba >= 0.35)

    return {
        "task": "marketing_response_prediction",
        "lead_conversion_probability": round(proba, 4),
        "recommended_for_outreach": is_promising,
        "decision_threshold_applied": 0.35,
        "model_version": "xgboost-prospective-v1",
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict/credit_risk")
def predict_credit_risk(applicant: CreditRiskInput):
    model = get_model("credit_risk_xgboost")
    df = pd.DataFrame([applicant.model_dump()])

    proba = float(model.predict_proba(df)[0, 1])
    risk_tier = "High Risk" if proba >= 0.50 else "Medium Risk" if proba >= 0.20 else "Low Risk"

    return {
        "task": "credit_default_prediction",
        "default_probability": round(proba, 4),
        "risk_tier": risk_tier,
        "model_version": "xgboost-credit-risk-v1",
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict/churn")
def predict_churn(customer: ChurnInput):
    model = get_model("customer_churn_random_forest")
    df = pd.DataFrame([customer.model_dump()])

    proba = float(model.predict_proba(df)[0, 1])
    retention_risk = "Elevated" if proba >= 0.40 else "Standard"

    return {
        "task": "customer_churn_prediction",
        "churn_probability": round(proba, 4),
        "retention_risk_tier": retention_risk,
        "model_version": "rf-churn-v1",
        "disclaimer": DISCLAIMER,
    }
