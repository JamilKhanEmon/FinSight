from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_marketing(artifacts_dir: Path, load_json_artifact, load_joblib_model):
    """Renders Bank Marketing Campaign Analytics view."""
    st.title("📞 Bank Marketing Campaign Analytics")
    st.markdown(
        "Predicts customer term deposit subscriptions while strictly adhering to prospective pre-call rules. "
        "Call duration is omitted from the prospective decision model."
    )

    metrics_dir = artifacts_dir / "metrics"
    p_metrics = load_json_artifact(metrics_dir / "marketing_prospective_metrics.json")
    r_metrics = load_json_artifact(metrics_dir / "marketing_retrospective_metrics.json")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Prospective Modeling",
        "Retrospective Duration Benchmark",
        "Decision Thresholds",
        "🔮 Live Outbound Lead Tester",
    ])

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

    with tab4:
        st.subheader("🔮 Live Outbound Lead Scoring Simulator")
        st.markdown(
            "Enter prospective customer attributes to determine in real-time whether the call center dialer "
            "should route this client for prioritized term-deposit outreach."
        )

        col_p1, col_p2, col_p3 = st.columns([1, 1, 1])
        preset = None
        with col_p1:
            if st.button("📋 Load High-Probability Profile (Senior, Prior Success)", use_container_width=True):
                preset = "high"
        with col_p2:
            if st.button("📋 Load Moderate Profile (Prime Age, Neutral)", use_container_width=True):
                preset = "medium"
        with col_p3:
            if st.button("📋 Load Low-Probability Profile (High Debt, Multiple Calls)", use_container_width=True):
                preset = "low"

        if preset == "high":
            default_age = 58
            default_job = "retired"
            default_marital = "married"
            default_edu = "university.degree"
            default_housing = "no"
            default_loan = "no"
            default_campaign = 1
            default_pdays = 3
            default_poutcome = "success"
            default_euribor = 1.286
        elif preset == "low":
            default_age = 24
            default_job = "blue-collar"
            default_marital = "single"
            default_edu = "basic.4y"
            default_housing = "yes"
            default_loan = "yes"
            default_campaign = 7
            default_pdays = 999
            default_poutcome = "nonexistent"
            default_euribor = 4.857
        else:
            default_age = 38
            default_job = "admin."
            default_marital = "married"
            default_edu = "university.degree"
            default_housing = "yes"
            default_loan = "no"
            default_campaign = 2
            default_pdays = 999
            default_poutcome = "nonexistent"
            default_euribor = 4.857

        with st.form("marketing_lead_form"):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                age_val = st.number_input("Client Age", min_value=18, max_value=100, value=default_age)
                job_list = ["admin.", "blue-collar", "technician", "services", "management", "retired", "entrepreneur", "self-employed", "housemaid", "student", "unemployed"]
                job_val = st.selectbox("Job / Occupation", job_list, index=job_list.index(default_job) if default_job in job_list else 0)
                marital_list = ["married", "single", "divorced"]
                marital_val = st.selectbox("Marital Status", marital_list, index=marital_list.index(default_marital) if default_marital in marital_list else 0)
                edu_list = ["university.degree", "high.school", "basic.9y", "professional.course", "basic.4y", "basic.6y", "unknown"]
                edu_val = st.selectbox("Education Level", edu_list, index=edu_list.index(default_edu) if default_edu in edu_list else 0)

            with f_col2:
                housing_val = st.selectbox("Housing Loan?", ["no", "yes"], index=0 if default_housing == "no" else 1)
                loan_val = st.selectbox("Personal Loan?", ["no", "yes"], index=0 if default_loan == "no" else 1)
                campaign_val = st.number_input("Contacts during Current Campaign", min_value=1, max_value=50, value=default_campaign)
                poutcome_list = ["nonexistent", "success", "failure"]
                poutcome_val = st.selectbox("Previous Campaign Outcome", poutcome_list, index=poutcome_list.index(default_poutcome) if default_poutcome in poutcome_list else 0)

            with f_col3:
                pdays_val = st.number_input("Days Since Previous Contact (999 = Never)", min_value=0, max_value=999, value=default_pdays)
                contact_val = st.selectbox("Contact Communication Type", ["cellular", "telephone"])
                euribor_val = st.number_input("Euribor 3-Month Interest Rate (%)", min_value=0.1, max_value=10.0, value=float(default_euribor), step=0.1)
                dialer_threshold = st.slider("Dialer Outreach Threshold", min_value=0.10, max_value=0.80, value=0.35, step=0.05)

            submitted = st.form_submit_button("🚀 Run Live AI Lead Scoring", use_container_width=True)

        if submitted:
            model = load_joblib_model("marketing_prospective_xgboost")
            if model is not None:
                lead_data = {
                    "age": int(age_val),
                    "job": str(job_val),
                    "marital": str(marital_val),
                    "education": str(edu_val),
                    "default": "no",
                    "housing": str(housing_val),
                    "loan": str(loan_val),
                    "contact": str(contact_val),
                    "month": "may",
                    "day_of_week": "mon",
                    "campaign": int(campaign_val),
                    "pdays": int(pdays_val),
                    "previous": 1 if pdays_val != 999 else 0,
                    "poutcome": str(poutcome_val),
                    "emp_var_rate": 1.1 if euribor_val > 3.0 else -1.8,
                    "cons_price_idx": 93.994,
                    "cons_conf_idx": -36.4,
                    "euribor3m": float(euribor_val),
                    "nr_employed": 5191.0 if euribor_val > 3.0 else 5099.1,
                }
                lead_df = pd.DataFrame([lead_data])
                proba = float(model.predict_proba(lead_df)[0, 1])

                st.markdown("### 🎯 Live Model Prediction Result")
                res_c1, res_c2, res_c3 = st.columns([1.5, 1, 1.5])
                with res_c1:
                    if proba >= dialer_threshold:
                        st.success("🟢 RECOMMENDED FOR OUTREACH (Call Client!)")
                        st.markdown("**Dialer Decision**: Route to Outbound Tele-Sales Team immediately.")
                    else:
                        st.error("🔴 SUPPRESS OUTREACH (Do Not Call)")
                        st.markdown("**Dialer Decision**: Do not dial. Low expected conversion; preserve dialer budget.")

                with res_c2:
                    st.metric("Conversion Probability", f"{proba:.2%}", delta=f"{(proba - dialer_threshold):.2%} vs cutoff")

                with res_c3:
                    st.progress(min(max(proba, 0.0), 1.0))
                    if proba >= 0.50:
                        st.caption("🔥 High-conviction prospect (Expected top-decile conversion).")
                    elif proba >= dialer_threshold:
                        st.caption("⚡ Moderate-conviction prospect (Above operational dialing cutoff).")
                    else:
                        st.caption("💤 Below operational cutoff threshold.")
            else:
                st.error("Model 'marketing_prospective_xgboost.joblib' not found in artifacts/models.")
