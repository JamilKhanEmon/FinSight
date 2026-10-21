from __future__ import annotations

import pandas as pd
from fastapi import APIRouter

from backend.schemas.payloads import ChurnInput, CreditRiskInput, MarketingLeadInput
from backend.services.model_registry import get_model

router = APIRouter(prefix="/predict", tags=["Predictive Inference"])

DISCLAIMER = (
    "Model predictions reflect probabilistic estimates based on historical statistical patterns. "
    "Predictions do not establish causality and must be combined with human banking judgment."
)


@router.post("/marketing")
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


@router.post("/credit_risk")
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


@router.post("/churn")
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
