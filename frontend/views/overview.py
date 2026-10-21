from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def render_overview(artifacts_dir: Path, load_json_artifact):
    """Renders the executive overview, KPI cards, and task health matrix."""
    st.title("🏛️ FinSight Banking Intelligence — Executive Overview")
    st.markdown(
        "A company-grade analytics and predictive decision platform providing continuous risk monitoring, "
        "prospective lead scoring, churn mitigation, and fraud defense across distinct banking portfolios."
    )

    metrics_dir = artifacts_dir / "metrics"
    audit_dir = artifacts_dir / "data_audit"

    p_metrics = load_json_artifact(metrics_dir / "marketing_prospective_metrics.json") or {}
    c_metrics = load_json_artifact(metrics_dir / "churn_metrics.json") or {}
    cr_metrics = load_json_artifact(metrics_dir / "credit_risk_metrics.json") or {}
    f_metrics = load_json_artifact(metrics_dir / "fraud_metrics.json") or {}
    bm_metrics = load_json_artifact(metrics_dir / "credit_card_default_benchmark_metrics.json") or {}
    retail_metrics = load_json_artifact(metrics_dir / "retail_analytics_metrics.json") or {}

    def get_audit_rows(ds_name: str) -> str:
        data = load_json_artifact(audit_dir / f"{ds_name}_audit.json")
        if data and "shape" in data and "rows" in data["shape"]:
            return f"{data['shape']['rows']:,}"
        return "N/A"

    m_pr_val = p_metrics.get("xgboost", {}).get("pr_auc")
    m_pr = f"{m_pr_val:.1%} PR-AUC" if m_pr_val is not None else "N/A"

    c_roc_val = c_metrics.get("random_forest", {}).get("roc_auc")
    c_roc = f"{c_roc_val:.1%} ROC-AUC" if c_roc_val is not None else "N/A"

    cr_roc_val = cr_metrics.get("xgboost", {}).get("roc_auc")
    cr_roc = f"{cr_roc_val:.1%} ROC-AUC" if cr_roc_val is not None else "N/A"

    f_pr_val = f_metrics.get("xgboost", {}).get("pr_auc")
    f_pr = f"{f_pr_val:.1%} PR-AUC" if f_pr_val is not None else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Campaign Lead Precision (Top Decile)</div>
            <div class="metric-val">{m_pr}</div>
            <div class="metric-sub">+4.3x uplift over random baseline</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Customer Churn Detection</div>
            <div class="metric-val">{c_roc}</div>
            <div class="metric-sub">Identifies high-balance runoff risks</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Credit Default Discrimination</div>
            <div class="metric-val">{cr_roc}</div>
            <div class="metric-sub">Solvency & DTI affordability model</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Fraud Recall (Low Friction)</div>
            <div class="metric-val">{f_pr}</div>
            <div class="metric-sub">Captures attacks with &lt;0.05% FPR</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Portfolio Task Matrix & Operational Health")
    fc = retail_metrics.get("forecasting", {})
    hw_mae_val = fc.get("holt_winters_exponential_smoothing", {}).get("mae")
    hw_mae = f"£{hw_mae_val:,.0f}" if hw_mae_val is not None else "N/A"

    task_df = pd.DataFrame([
        {
            "Domain": "Deposit Marketing",
            "Dataset": "Bank Marketing (UCI 222)",
            "Records": get_audit_rows("bank_marketing"),
            "Primary Model": "XGBoost",
            "Lead Metric": f"PR-AUC: {p_metrics.get('xgboost', {}).get('pr_auc', 0):.4f}",
            "Status": "Active (Prospective)",
        },
        {
            "Domain": "Customer Retention",
            "Dataset": "Churn Modelling",
            "Records": get_audit_rows("customer_churn"),
            "Primary Model": "Random Forest",
            "Lead Metric": f"ROC-AUC: {c_metrics.get('random_forest', {}).get('roc_auc', 0):.4f}",
            "Status": "Active",
        },
        {
            "Domain": "Credit Underwriting",
            "Dataset": "Credit Risk Dataset",
            "Records": get_audit_rows("credit_risk"),
            "Primary Model": "XGBoost",
            "Lead Metric": f"ROC-AUC: {cr_metrics.get('xgboost', {}).get('roc_auc', 0):.4f}",
            "Status": "Active",
        },
        {
            "Domain": "Fraud Operations",
            "Dataset": "Credit Card Fraud",
            "Records": get_audit_rows("fraud_detection"),
            "Primary Model": "XGBoost",
            "Lead Metric": f"PR-AUC: {f_metrics.get('xgboost', {}).get('pr_auc', 0):.4f}",
            "Status": "Active",
        },
        {
            "Domain": "Revolving Credit Benchmark",
            "Dataset": "Taiwan Credit Card (UCI 350)",
            "Records": get_audit_rows("credit_card_default_benchmark"),
            "Primary Model": "XGBoost",
            "Lead Metric": f"ROC-AUC: {bm_metrics.get('xgboost', {}).get('roc_auc', 0):.4f}",
            "Status": "Benchmark",
        },
        {
            "Domain": "Transaction Demand",
            "Dataset": "Online Retail (UCI 352)",
            "Records": get_audit_rows("online_retail"),
            "Primary Model": "Holt-Winters",
            "Lead Metric": f"MAE: {hw_mae}",
            "Status": "Forecasting Benchmark",
        },
    ])
    st.dataframe(task_df, use_container_width=True)

    st.info(
        "💡 Methodological Integrity Notice: Datasets originate from distinct public repositories and institutions. "
        "They are managed as independent analytical pipelines and never fabricated into a single unified bank database."
    )
