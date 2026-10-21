from __future__ import annotations

import json
from pathlib import Path

import joblib
import streamlit as st

from frontend.styles import DASHBOARD_CSS
from frontend.views import (
    render_churn,
    render_credit_risk,
    render_data_audit,
    render_documentation,
    render_forecasting,
    render_fraud,
    render_leaderboard,
    render_marketing,
    render_overview,
    render_research,
    render_segmentation,
)

st.set_page_config(
    page_title="FinSight | Banking Intelligence & Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"


@st.cache_resource
def load_joblib_model(model_name: str):
    """Loads a serialized Scikit-learn or XGBoost pipeline."""
    model_path = MODELS_DIR / f"{model_name}.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None


@st.cache_data
def load_json_artifact(file_path: Path):
    """Parses a cached JSON artifact."""
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_text_artifact(file_path: Path):
    """Reads a cached Markdown or text artifact."""
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def main():
    st.sidebar.title("🏛️ FinSight Intelligence")
    st.sidebar.markdown("*Enterprise Banking & ML Platform*")

    nav_options = [
        "Executive Overview",
        "Data Quality & Audit Explorer",
        "Marketing Campaign Analytics",
        "Customer Churn Analytics",
        "Credit Risk Underwriting",
        "Transaction Fraud Detection",
        "Customer Segmentation (RFM)",
        "Transaction Trends & Forecasting",
        "Model Performance Comparison",
        "Research & Feature Engineering",
        "Strategic Documentation",
    ]

    selected_page = st.sidebar.radio("Navigation", nav_options)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Hardware**: macOS Apple Silicon")
    st.sidebar.markdown("**Engine**: Scikit-Learn | XGBoost | FastAPI")

    if selected_page == "Executive Overview":
        render_overview(ARTIFACTS_DIR, load_json_artifact)
    elif selected_page == "Data Quality & Audit Explorer":
        render_data_audit(ARTIFACTS_DIR, load_json_artifact)
    elif selected_page == "Marketing Campaign Analytics":
        render_marketing(ARTIFACTS_DIR, load_json_artifact, load_joblib_model)
    elif selected_page == "Customer Churn Analytics":
        render_churn(ARTIFACTS_DIR, load_json_artifact, load_joblib_model)
    elif selected_page == "Credit Risk Underwriting":
        render_credit_risk(ARTIFACTS_DIR, load_json_artifact, load_joblib_model)
    elif selected_page == "Transaction Fraud Detection":
        render_fraud(ARTIFACTS_DIR, load_json_artifact, load_joblib_model)
    elif selected_page == "Customer Segmentation (RFM)":
        render_segmentation(ARTIFACTS_DIR, load_json_artifact)
    elif selected_page == "Transaction Trends & Forecasting":
        render_forecasting(ARTIFACTS_DIR, load_json_artifact)
    elif selected_page == "Model Performance Comparison":
        render_leaderboard(ARTIFACTS_DIR, load_json_artifact)
    elif selected_page == "Research & Feature Engineering":
        render_research(ARTIFACTS_DIR, load_json_artifact, load_text_artifact)
    elif selected_page == "Strategic Documentation":
        render_documentation(ARTIFACTS_DIR, load_text_artifact)


if __name__ == "__main__":
    main()
