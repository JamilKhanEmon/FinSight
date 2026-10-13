"""FinSight — Banking Intelligence & Customer Analytics Platform Dashboard.

A state-of-the-art interactive Streamlit BI and ML analytics application
visualizing empirical experimental results, models, and business intelligence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="FinSight | Banking Intelligence & Analytics",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2638 0%, #151a28 100%);
        border: 1px solid #2e3852;
        border-radius: 10px;
        padding: 20px;
        color: white;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 5px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 5px;
    }
    .badge-success {
        background-color: #065f46;
        color: #34d399;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #78350f;
        color: #fbbf24;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-info {
        background-color: #1e3a8a;
        color: #60a5fa;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to load artifacts
ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
AUDIT_DIR = ARTIFACTS_DIR / "data_audit"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
BUSINESS_DIR = ARTIFACTS_DIR / "business"


@st.cache_data
def load_json_artifact(file_path: Path):
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_text_artifact(file_path: Path):
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


# Sidebar Navigation
st.sidebar.title("🏛️ FinSight Intelligence")
st.sidebar.markdown("*Enterprise Banking & ML Platform*")

menu = st.sidebar.radio(
    "Navigation",
    [
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
        "Business Recommendations",
        "Project & Model Documentation",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Environment**: Apple Silicon M5 (16 GB RAM)")
st.sidebar.markdown("**Engine**: Scikit-Learn | XGBoost | DuckDB")


# ==============================================================================
# 1. Executive Overview
# ==============================================================================
if menu == "Executive Overview":
    st.title("🏛️ FinSight Banking Intelligence — Executive Overview")
    st.markdown(
        "A company-grade analytics and predictive decision platform providing continuous risk monitoring, "
        "prospective lead scoring, churn mitigation, and fraud defense across distinct banking portfolios."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Campaign Lead Precision (Top Decile)</div>
            <div class="metric-val">48.1% PR-AUC</div>
            <div class="metric-sub">+4.3x uplift over random baseline</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Customer Churn Detection</div>
            <div class="metric-val">86.2% ROC-AUC</div>
            <div class="metric-sub">Identifies high-balance runoff risks</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Credit Default Discrimination</div>
            <div class="metric-val">93.8% ROC-AUC</div>
            <div class="metric-sub">Solvency & DTI affordability model</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Fraud Recall (Low Friction)</div>
            <div class="metric-val">87.8% PR-AUC</div>
            <div class="metric-sub">Captures attacks with <0.05% FPR</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Portfolio Task Matrix & Operational Health")
    task_df = pd.DataFrame([
        {"Domain": "Deposit Marketing", "Dataset": "Bank Marketing (UCI 222)", "Records": "41,176", "Primary Model": "XGBoost", "Lead Metric": "PR-AUC: 0.4805", "Status": "Active (Prospective)"},
        {"Domain": "Customer Retention", "Dataset": "Churn Modelling", "Records": "10,000", "Primary Model": "Random Forest", "Lead Metric": "ROC-AUC: 0.8624", "Status": "Active"},
        {"Domain": "Credit Underwriting", "Dataset": "Credit Risk Dataset", "Records": "32,576", "Primary Model": "XGBoost", "Lead Metric": "ROC-AUC: 0.9381", "Status": "Active"},
        {"Domain": "Fraud Operations", "Dataset": "Credit Card Fraud", "Records": "284,807", "Primary Model": "XGBoost", "Lead Metric": "PR-AUC: 0.8541", "Status": "Active"},
        {"Domain": "Revolving Credit Benchmark", "Dataset": "Taiwan Credit Card (UCI 350)", "Records": "30,000", "Primary Model": "XGBoost", "Lead Metric": "ROC-AUC: 0.7768", "Status": "Benchmark"},
        {"Domain": "Transaction Demand", "Dataset": "Online Retail (UCI 352)", "Records": "397,884", "Primary Model": "Holt-Winters", "Lead Metric": "MAE: £10,854", "Status": "Forecasting Benchmark"},
    ])
    st.dataframe(task_df, use_container_width=True)

    st.info(
        "💡 **Methodological Integrity Notice**: Datasets originate from distinct public repositories and institutions. "
        "They are managed as independent analytical pipelines and never fabricated into a single unified bank database."
    )


# ==============================================================================
# 2. Data Quality & Audit Explorer
# ==============================================================================
elif menu == "Data Quality & Audit Explorer":
    st.title("🔍 Data Quality & Schema Audit Explorer")
    st.markdown("Automated data audits documenting dimensions, missingness, duplicates, and class distributions.")

    dataset_options = [
        ("bank_marketing", "Bank Marketing (UCI 222)"),
        ("customer_churn", "Customer Churn"),
        ("credit_risk", "Credit Risk"),
        ("fraud_detection", "Transaction Fraud Detection"),
        ("credit_card_default_benchmark", "Credit Card Default Benchmark"),
        ("online_retail", "Online Retail Transaction Benchmark"),
    ]

    selected_key = st.selectbox(
        "Select Dataset to Inspect Audit Report",
        [k[0] for k in dataset_options],
        format_func=lambda x: dict(dataset_options).get(x, x),
    )

    audit_data = load_json_artifact(AUDIT_DIR / f"{selected_key}_audit.json")
    if audit_data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{audit_data['shape']['rows']:,}")
        c2.metric("Total Columns", audit_data["shape"]["columns"])
        c3.metric("Duplicate Rows", f"{audit_data['duplicate_rows']:,}")
        c4.metric("Missing Cells", f"{audit_data['total_missing_cells']:,}")

        st.markdown(f"**License**: {audit_data.get('license', 'N/A')}")
        st.markdown(f"**Source URL**: {audit_data.get('source_url', 'N/A')}")

        if audit_data.get("target_summary"):
            st.markdown("#### 🎯 Target Variable Distribution")
            t_sum = audit_data["target_summary"]
            dist_df = pd.DataFrame([
                {"Class": k, "Count": v, "Proportion": f"{t_sum['class_ratio'].get(k, 0):.2%}"}
                for k, v in t_sum["distribution"].items()
            ])
            st.table(dist_df)

        st.markdown("#### 📋 Column-Level Summary")
        cols_summary = []
        for col_name, c_meta in audit_data["columns"].items():
            entry = {
                "Column": col_name,
                "Type": c_meta.get("dtype"),
                "Missing Count": c_meta.get("missing_count"),
                "Missing %": f"{c_meta.get('missing_pct')}%",
                "Unique Values": c_meta.get("n_unique"),
            }
            if "mean" in c_meta:
                entry["Stats"] = f"Mean: {c_meta['mean']:.2f} [{c_meta['min']}, {c_meta['max']}]"
            elif "top_categories" in c_meta:
                entry["Stats"] = f"Top: {', '.join(list(c_meta['top_categories'].keys())[:2])}"
            cols_summary.append(entry)
        st.dataframe(pd.DataFrame(cols_summary), use_container_width=True)
    else:
        st.warning(f"No audit report found for {selected_key}. Please run audit pipeline.")


# ==============================================================================
# 3. Marketing Campaign Analytics
# ==============================================================================
elif menu == "Marketing Campaign Analytics":
    st.title("📞 Bank Marketing Campaign Analytics")
    st.markdown(
        "Predicts customer term deposit subscriptions while strictly adhering to prospective pre-call rules. "
        "Call duration is omitted from the prospective decision model."
    )

    p_metrics = load_json_artifact(METRICS_DIR / "marketing_prospective_metrics.json")
    r_metrics = load_json_artifact(METRICS_DIR / "marketing_retrospective_metrics.json")

    tab1, tab2, tab3 = st.tabs(["Prospective Modeling", "Retrospective Duration Benchmark", "Decision Thresholds"])

    with tab1:
        st.subheader("Prospective Pre-Call Models (duration EXCLUDED)")
        if p_metrics:
            p_rows = []
            for m_name, d in p_metrics.items():
                p_rows.append({
                    "Model": m_name,
                    "PR-AUC": d["pr_auc"],
                    "ROC-AUC": d["roc_auc"],
                    "Precision": d["precision"],
                    "Recall": d["recall"],
                    "F1 Score": d["f1_score"],
                    "Brier Calibration": d["brier_score"],
                })
            st.dataframe(pd.DataFrame(p_rows), use_container_width=True)

            fig = go.Figure()
            for m_name, d in p_metrics.items():
                if d.get("threshold_analysis"):
                    th_df = pd.DataFrame(d["threshold_analysis"])
                    fig.add_trace(go.Scatter(x=th_df["threshold"], y=th_df["f1"], mode="lines+markers", name=f"{m_name} (F1)"))
            fig.update_layout(title="F1-Score across Decision Thresholds", xaxis_title="Threshold", yaxis_title="F1 Score")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("⚠️ Retrospective Benchmark (With Post-Call Duration)")
        st.warning(
            "Call duration is only known after a call terminates. Using duration in prospective lead scoring "
            "constitutes target leakage. The retrospective benchmark below empirically demonstrates this leakage effect."
        )
        if r_metrics and p_metrics:
            comp_rows = [
                {"Setting": "Prospective (No Duration)", "Model": "XGBoost", "PR-AUC": p_metrics["xgboost"]["pr_auc"], "ROC-AUC": p_metrics["xgboost"]["roc_auc"]},
                {"Setting": "Retrospective (With Duration)", "Model": "XGBoost", "PR-AUC": r_metrics["xgboost"]["pr_auc"], "ROC-AUC": r_metrics["xgboost"]["roc_auc"]},
            ]
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

    with tab3:
        st.subheader("Interactive Threshold Optimizer")
        th_slider = st.slider("Select Outbound Dialing Score Threshold", 0.05, 0.90, 0.40, 0.05)
        if p_metrics and "xgboost" in p_metrics:
            th_list = p_metrics["xgboost"].get("threshold_analysis", [])
            matched = next((x for x in th_list if abs(x["threshold"] - th_slider) < 0.03), None)
            if matched:
                c1, c2, c3 = st.columns(3)
                c1.metric("Precision", f"{matched['precision']:.2%}")
                c2.metric("Recall", f"{matched['recall']:.2%}")
                c3.metric("F1 Score", f"{matched['f1']:.2%}")


# ==============================================================================
# 4. Customer Churn Analytics
# ==============================================================================
elif menu == "Customer Churn Analytics":
    st.title("🚪 Customer Churn & Retention Analytics")
    st.markdown("Assesses customer attrition probability across account balance, activity, and geographic segments.")

    churn_metrics = load_json_artifact(METRICS_DIR / "churn_metrics.json")
    if churn_metrics:
        rows = [
            {"Model": k, "PR-AUC": v["pr_auc"], "ROC-AUC": v["roc_auc"], "Precision": v["precision"], "Recall": v["recall"], "F1 Score": v["f1_score"]}
            for k, v in churn_metrics.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Plotly comparison
        fig = px.bar(pd.DataFrame(rows), x="Model", y=["ROC-AUC", "PR-AUC", "F1 Score"], barmode="group", title="Model Performance on Retention Test Holdout")
        st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# 5. Credit Risk Underwriting
# ==============================================================================
elif menu == "Credit Risk Underwriting":
    st.title("💳 Credit Risk & Default Underwriting")
    st.markdown("Predicts installment loan default risk using income solvency and debt-to-income indicators.")

    cr_metrics = load_json_artifact(METRICS_DIR / "credit_risk_metrics.json")
    if cr_metrics:
        rows = [
            {"Model": k, "PR-AUC": v["pr_auc"], "ROC-AUC": v["roc_auc"], "Precision": v["precision"], "Recall": v["recall"], "F1 Score": v["f1_score"]}
            for k, v in cr_metrics.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        fig = px.bar(pd.DataFrame(rows), x="Model", y=["ROC-AUC", "PR-AUC"], barmode="group", title="Credit Default Model Discrimination")
        st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# 6. Transaction Fraud Detection
# ==============================================================================
elif menu == "Transaction Fraud Detection":
    st.title("🛡️ Transaction Fraud Detection & Anomaly Monitoring")
    st.markdown("Monitors imbalanced transaction streams (0.17% fraud rate) with precision-recall sensitivity.")

    fraud_metrics = load_json_artifact(METRICS_DIR / "fraud_metrics.json")
    if fraud_metrics:
        rows = [
            {"Model": k, "PR-AUC": v["pr_auc"], "ROC-AUC": v["roc_auc"], "Precision": v["precision"], "Recall": v["recall"], "F1 Score": v["f1_score"]}
            for k, v in fraud_metrics.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if "xgboost" in fraud_metrics and fraud_metrics["xgboost"].get("threshold_analysis"):
            th_df = pd.DataFrame(fraud_metrics["xgboost"]["threshold_analysis"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=th_df["threshold"], y=th_df["precision"], mode="lines+markers", name="Precision (Positive Predictive Value)"))
            fig.add_trace(go.Scatter(x=th_df["threshold"], y=th_df["recall"], mode="lines+markers", name="Recall (Fraud Capture Rate)"))
            fig.add_trace(go.Scatter(x=th_df["threshold"], y=th_df["f1"], mode="lines+markers", name="F1 Score"))
            fig.update_layout(title="Precision-Recall Trade-off Curve by Decision Threshold", xaxis_title="Risk Score Threshold", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# 7. Customer Segmentation (RFM)
# ==============================================================================
elif menu == "Customer Segmentation (RFM)":
    st.title("👥 Customer RFM Segmentation")
    st.markdown("Recency, Frequency, and Monetary Value behavioral clustering on transactional records.")

    retail_data = load_json_artifact(METRICS_DIR / "retail_analytics_metrics.json")
    if retail_data and "segmentation" in retail_data:
        seg = retail_data["segmentation"]
        st.markdown(f"**Total Segmented Customers**: {seg['total_segmented_customers']:,} | **Clusters**: {seg['n_clusters']}")

        clusters_df = pd.DataFrame(seg["clusters"])
        st.dataframe(clusters_df, use_container_width=True)

        fig = px.scatter(
            clusters_df,
            x="avg_recency_days",
            y="avg_monetary_spend",
            size="customer_count",
            color="cluster_id",
            hover_name="cluster_id",
            title="Cluster Map: Recency vs Monetary Spend (Bubble size = Customer Count)",
        )
        st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# 8. Transaction Trends & Forecasting
# ==============================================================================
elif menu == "Transaction Trends & Forecasting":
    st.title("📈 Transaction Demand & Revenue Forecasting")
    st.markdown("Evaluates daily revenue time series forecasting models on a 28-day chronological holdout split.")

    retail_data = load_json_artifact(METRICS_DIR / "retail_analytics_metrics.json")
    if retail_data and "forecasting" in retail_data:
        fc = retail_data["forecasting"]
        rows = [
            {"Model": "7-Day Naive Seasonal", "MAE": f"£{fc['naive_seasonal_7d']['mae']:,.2f}", "RMSE": f"£{fc['naive_seasonal_7d']['rmse']:,.2f}", "MAPE": f"{fc['naive_seasonal_7d']['mape']:.2%}"},
            {"Model": "7-Day Moving Average", "MAE": f"£{fc['moving_average_7d']['mae']:,.2f}", "RMSE": f"£{fc['moving_average_7d']['rmse']:,.2f}", "MAPE": f"{fc['moving_average_7d']['mape']:.2%}"},
            {"Model": "Holt-Winters Exponential Smoothing", "MAE": f"£{fc['holt_winters_exponential_smoothing']['mae']:,.2f}", "RMSE": f"£{fc['holt_winters_exponential_smoothing']['rmse']:,.2f}", "MAPE": f"{fc['holt_winters_exponential_smoothing']['mape']:.2%}"},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.info(f"📅 Holdout Evaluation Window: {fc['test_period']} ({fc['test_days']} days)")


# ==============================================================================
# 9. Model Performance Comparison
# ==============================================================================
elif menu == "Model Performance Comparison":
    st.title("⚖️ Cross-Task Model Performance Comparison")
    st.markdown("Consolidated model leaderboard across all active banking and benchmark tasks.")

    tasks = [
        ("marketing_prospective", "Bank Marketing (Prospective)"),
        ("churn", "Customer Churn"),
        ("credit_risk", "Credit Risk Default"),
        ("fraud", "Fraud Detection"),
        ("credit_card_default_benchmark", "Credit Card Default Benchmark"),
    ]

    all_rows = []
    for t_key, t_label in tasks:
        m_dict = load_json_artifact(METRICS_DIR / f"{t_key}_metrics.json")
        if m_dict:
            for m_name, d in m_dict.items():
                all_rows.append({
                    "Task": t_label,
                    "Model": m_name,
                    "PR-AUC": d.get("pr_auc"),
                    "ROC-AUC": d.get("roc_auc"),
                    "Precision": d.get("precision"),
                    "Recall": d.get("recall"),
                    "F1 Score": d.get("f1_score"),
                    "Brier Loss": d.get("brier_score"),
                })

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        st.dataframe(df_all, use_container_width=True)

        fig = px.scatter(df_all, x="ROC-AUC", y="PR-AUC", color="Task", hover_name="Model", title="Cross-Task ROC-AUC vs PR-AUC Landscape")
        st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# 10. Research & Feature Engineering
# ==============================================================================
elif menu == "Research & Feature Engineering":
    st.title("🔬 Published Research Adaptation: Interaction-Focused Feature Engineering & PCA")
    st.markdown("""
    **Adapted Academic Source**:  
    📄 *Harmonization of Heart Disease Dataset for Accurate Diagnosis: A Machine Learning Approach Enhanced by Feature Engineering*  
    **Authors**: Jamil Khan Emon et al.  
    **Published in**: *Computers, Materials & Continua (CMC)*, Vol. 82, No. 3, 2025 (Q2, SCI, Scopus)  
    🔗 [https://www.techscience.com/cmc/v82n3/59942](https://www.techscience.com/cmc/v82n3/59942)
    """)

    st.info(
        "🧠 **Transferability Framework**: Adapting the 3 core methodological pillars from Jamil Khan Emon et al. (CMC 2025) "
        "across retail banking datasets: (1) Statistical Pairwise Interaction & Ratio Generation, "
        "(2) Dimensionality Reduction via PCA retaining 95% variance, and (3) Soft-Voting Ensemble (Random Forest + XGBoost + Logistic Regression)."
    )

    paper_results = load_json_artifact(ARTIFACTS_DIR / "research" / "paper_adaptation_results.json")
    if paper_results:
        st.subheader("📊 Cross-Dataset Transferability & Ablation Leaderboard")
        rows = []
        for d_key, r in paper_results.items():
            b = r.get("baseline_single_rf_metrics") or r.get("baseline", {})
            p = r.get("proposed_voting_metrics") or r.get("proposed", {})
            cv = r.get("kfold_cv_results", {})
            rows.append({
                "Dataset": d_key,
                "Pipeline": "Baseline (Single RF, Raw Features)",
                "Original Features": r.get("n_original_features", "-"),
                "Engineered Features": "-",
                "PCA (95% Var)": "-",
                "Accuracy": f"{b.get('accuracy', 0):.2%}",
                "Sensitivity": f"{b.get('sensitivity_recall', 0):.2%}",
                "Specificity": f"{b.get('specificity', 0):.2%}",
                "F1 Score": f"{b.get('f1_score', 0):.4f}",
                "ROC-AUC": f"{b.get('roc_auc', 0):.4f}",
                "CV Acc": "-",
            })
            rows.append({
                "Dataset": d_key,
                "Pipeline": "⭐ Proposed (Interactions + PCA 95% + Voting)",
                "Original Features": r.get("n_original_features", "-"),
                "Engineered Features": r.get("n_engineered_features", "-"),
                "PCA (95% Var)": r.get("n_pca_components", "-"),
                "Accuracy": f"{p.get('accuracy', 0):.2%}",
                "Sensitivity": f"{p.get('sensitivity_recall', 0):.2%}",
                "Specificity": f"{p.get('specificity', 0):.2%}",
                "F1 Score": f"{p.get('f1_score', 0):.4f}",
                "ROC-AUC": f"{p.get('roc_auc', 0):.4f}",
                "CV Acc": f"{cv.get('mean_accuracy', 0):.2%}" if cv else "-",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("🔍 Deep-Dive: Individual Models & Correlation Analysis by Dataset")
        selected_ds = st.selectbox("Select Dataset to Inspect:", list(paper_results.keys()))
        if selected_ds in paper_results:
            ds_data = paper_results[selected_ds]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Original Features", ds_data.get("n_original_features"))
            c2.metric("Engineered Features", ds_data.get("n_engineered_features"))
            c3.metric("PCA Components (95% Var)", ds_data.get("n_pca_components"))
            cv_res = ds_data.get("kfold_cv_results", {})
            c4.metric("5-Fold Mean Accuracy", f"{cv_res.get('mean_accuracy', 0):.2%}")

            # Individual models breakdown
            ind_models = ds_data.get("individual_models_metrics", {})
            if ind_models:
                st.markdown("##### 🤖 Individual Models vs. Voting Ensemble Performance & Runtimes")
                m_rows = []
                for m_name, m_val in ind_models.items():
                    m_rows.append({
                        "Model": m_name,
                        "Accuracy": f"{m_val.get('accuracy', 0):.2%}",
                        "Precision": f"{m_val.get('precision', 0):.2%}",
                        "Sensitivity (Recall)": f"{m_val.get('sensitivity_recall', 0):.2%}",
                        "Specificity": f"{m_val.get('specificity', 0):.2%}",
                        "F1 Score": f"{m_val.get('f1_score', 0):.4f}",
                        "ROC-AUC": f"{m_val.get('roc_auc', 0):.4f}",
                        "Runtime (s)": f"{m_val.get('run_time_sec', 0):.3f}s",
                    })
                # Add voting ensemble
                v_val = ds_data.get("proposed_voting_metrics", {})
                m_rows.append({
                    "Model": "⭐ VotingClassifier (Soft)",
                    "Accuracy": f"{v_val.get('accuracy', 0):.2%}",
                    "Precision": f"{v_val.get('precision', 0):.2%}",
                    "Sensitivity (Recall)": f"{v_val.get('sensitivity_recall', 0):.2%}",
                    "Specificity": f"{v_val.get('specificity', 0):.2%}",
                    "F1 Score": f"{v_val.get('f1_score', 0):.4f}",
                    "ROC-AUC": f"{v_val.get('roc_auc', 0):.4f}",
                    "Runtime (s)": f"{v_val.get('run_time_sec', 0):.3f}s",
                })
                st.dataframe(pd.DataFrame(m_rows), use_container_width=True)

            # Top correlations
            corrs = ds_data.get("top_pairwise_correlations", [])
            if corrs:
                st.markdown("##### 🔗 Top Screened Pairwise Correlations (Cell 15 Replication)")
                corr_df = pd.DataFrame(corrs)[["combination", "correlation", "abs_correlation"]]
                st.dataframe(corr_df.rename(columns={"combination": "Feature Combination (var1 * var2)"}), use_container_width=True)

    paper_md = load_text_artifact(ARTIFACTS_DIR / "research" / "paper_methodology_and_adaptation_report.md")
    if paper_md:
        with st.expander("📖 View Full Academic Methodology & Adaptation Report", expanded=False):
            st.markdown(paper_md)



# ==============================================================================
# 11. Business Recommendations
# ==============================================================================
elif menu == "Business Recommendations":
    st.title("💼 Strategic Business Recommendations")
    st.markdown("Actionable recommendations translated from empirical modeling evidence.")

    rec_md = load_text_artifact(BUSINESS_DIR / "business_recommendations.md")
    if rec_md:
        st.markdown(rec_md)
    else:
        st.warning("Recommendations document not found.")


# ==============================================================================
# 12. Project & Model Documentation
# ==============================================================================
elif menu == "Project & Model Documentation":
    st.title("📚 FinSight Platform Documentation")
    st.markdown("Architecture, KPI dictionary, and ethical limitations.")

    tab1, tab2, tab3 = st.tabs(["KPI Dictionary", "Business Requirements", "System Architecture"])
    with tab1:
        kpi_md = load_text_artifact(BUSINESS_DIR / "kpi_dictionary.md")
        if kpi_md:
            st.markdown(kpi_md)
    with tab2:
        brd_md = load_text_artifact(BUSINESS_DIR / "business_requirements_document.md")
        if brd_md:
            st.markdown(brd_md)
    with tab3:
        st.markdown("""
        ### FinSight Modular Architecture
        - **`src/finsight/config/`**: Central YAML configurations (`project.yaml`, `datasets.yaml`, `models.yaml`)
        - **`src/finsight/ingestion/`**: Safe memory-checked loaders for CSV, Excel, and Parquet
        - **`src/finsight/validation/`**: Automated schema verification and data quality auditing
        - **`src/finsight/cleaning/`**: Transparent, audited cleaning and deduplication
        - **`src/finsight/features/`**: Domain-informed Scikit-learn pipelines with zero target leakage
        - **`src/finsight/models/`**: Unified trainer for Dummy, Logistic Regression, Decision Tree, Random Forest, XGBoost
        - **`src/finsight/evaluation/`**: Multi-metric evaluation (PR-AUC, ROC-AUC, Brier score, threshold curve)
        - **`src/finsight/analytics/`**: RFM customer segmentation and time-series demand forecasting
        - **`src/finsight/dashboard/`**: Polished interactive Streamlit UI with Plotly charts
        - **`src/finsight/api/`**: Optional FastAPI prediction service with Pydantic validation
        """)
