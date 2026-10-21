from __future__ import annotations

from fastapi import APIRouter

from ai.config.loader import get_resolved_path

router = APIRouter(tags=["Health & Status"])


@router.get("/")
def read_root():
    return {
        "platform": "FinSight Banking Intelligence API",
        "version": "0.1.0",
        "status": "operational",
        "endpoints": ["/health", "/models", "/predict/marketing", "/predict/credit_risk", "/predict/churn"],
    }


@router.get("/health")
def read_health():
    return {"status": "healthy", "service": "finsight-api"}


@router.get("/models")
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
