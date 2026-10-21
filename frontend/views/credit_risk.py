from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def render_credit_risk(artifacts_dir: Path, load_json_artifact, load_joblib_model):
    """Renders Credit Risk Underwriting view."""
    st.title("💳 Credit Risk & Default Underwriting")
    st.markdown("Predicts installment loan default risk using income solvency and debt-to-income indicators.")

    metrics_dir = artifacts_dir / "metrics"
    cr_metrics = load_json_artifact(metrics_dir / "credit_risk_metrics.json")
    tab_cr1, tab_cr2 = st.tabs(["📊 Underwriting Benchmarks", "🔮 Live Credit Underwriting Simulator"])

    with tab_cr1:
        if cr_metrics:
            rows = [
                {"Model": k, "PR-AUC": v["pr_auc"], "ROC-AUC": v["roc_auc"], "Precision": v["precision"], "Recall": v["recall"], "F1 Score": v["f1_score"]}
                for k, v in cr_metrics.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            fig = px.bar(pd.DataFrame(rows), x="Model", y=["ROC-AUC", "PR-AUC"], barmode="group", title="Credit Default Model Discrimination")
            st.plotly_chart(fig, use_container_width=True)

    with tab_cr2:
        st.subheader("🔮 Live Loan Credit Underwriting Simulator")
        st.markdown("Assess loan applicant solvency and receive automated IFRS 9 risk tiering and approval recommendations.")

        col_cr1, col_cr2 = st.columns(2)
        cr_preset = None
        with col_cr1:
            if st.button("📋 Load Prime Applicant (High Income, Stable Job, Clean History)", use_container_width=True):
                cr_preset = "prime"
        with col_cr2:
            if st.button("📋 Load Subprime Applicant (Low Income, High Loan %, Past Default)", use_container_width=True):
                cr_preset = "subprime"

        if cr_preset == "prime":
            def_cr_age, def_income, def_home, def_emp, def_intent, def_grade, def_amnt, def_rate, def_hist, def_def = (
                35, 95000.0, "MORTGAGE", 8.0, "HOMEIMPROVEMENT", "A", 12000.0, 7.5, 9, "N"
            )
        elif cr_preset == "subprime":
            def_cr_age, def_income, def_home, def_emp, def_intent, def_grade, def_amnt, def_rate, def_hist, def_def = (
                22, 22000.0, "RENT", 1.0, "PERSONAL", "D", 14000.0, 16.5, 2, "Y"
            )
        else:
            def_cr_age, def_income, def_home, def_emp, def_intent, def_grade, def_amnt, def_rate, def_hist, def_def = (
                28, 65000.0, "RENT", 3.0, "PERSONAL", "B", 10000.0, 11.14, 4, "N"
            )

        with st.form("credit_risk_sim_form"):
            cr_c1, cr_c2, cr_c3 = st.columns(3)
            with cr_c1:
                p_age = st.number_input("Applicant Age", 18, 100, int(def_cr_age))
                p_inc = st.number_input("Annual Income ($)", 5000.0, 500000.0, float(def_income), step=5000.0)
                home_list = ["RENT", "OWN", "MORTGAGE", "OTHER"]
                p_home = st.selectbox("Home Ownership", home_list, index=home_list.index(def_home))
                p_emp = st.number_input("Employment Length (Years)", 0.0, 50.0, float(def_emp), step=0.5)

            with cr_c2:
                intent_list = ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
                p_intent = st.selectbox("Loan Intent", intent_list, index=intent_list.index(def_intent))
                grade_list = ["A", "B", "C", "D", "E", "F", "G"]
                p_grade = st.selectbox("Loan Grade", grade_list, index=grade_list.index(def_grade))
                p_amnt = st.number_input("Loan Amount ($)", 500.0, 100000.0, float(def_amnt), step=1000.0)
                p_rate = st.number_input("Loan Interest Rate (%)", 1.0, 35.0, float(def_rate), step=0.25)

            with cr_c3:
                p_def = st.selectbox("Historical Default on File?", ["N", "Y"], index=0 if def_def == "N" else 1)
                p_hist = st.number_input("Credit History Length (Years)", 1, 40, int(def_hist))
                cr_threshold = st.slider("Underwriting Risk Cutoff", 0.10, 0.70, 0.35, 0.05)

            submitted_cr = st.form_submit_button("🚀 Run Live Credit Underwriting", use_container_width=True)

        if submitted_cr:
            cr_model = load_joblib_model("credit_risk_xgboost")
            if cr_model:
                dti = p_amnt / max(p_inc, 1000.0)
                cr_payload = {
                    "person_age": int(p_age),
                    "person_income": float(p_inc),
                    "person_home_ownership": str(p_home),
                    "person_emp_length": float(p_emp),
                    "loan_intent": str(p_intent),
                    "loan_grade": str(p_grade),
                    "loan_amnt": float(p_amnt),
                    "loan_int_rate": float(p_rate),
                    "loan_percent_income": float(dti),
                    "cb_person_default_on_file": str(p_def),
                    "cb_person_cred_hist_length": int(p_hist),
                }
                cr_proba = float(cr_model.predict_proba(pd.DataFrame([cr_payload]))[0, 1])

                st.markdown("### 🎯 Live Credit Decision Result")
                cr_res1, cr_res2, cr_res3 = st.columns([1.5, 1, 1.5])
                with cr_res1:
                    if cr_proba < cr_threshold:
                        st.success("🟢 LOAN APPLICATION APPROVED")
                        st.markdown(f"**Underwriting Tier**: Prime / Low Risk. DTI Ratio: `{dti:.1%}`")
                    else:
                        st.error("🔴 LOAN APPLICATION DECLINED / REFERRED")
                        st.markdown(f"**Underwriting Tier**: High Risk Default. Exceeds risk tolerance at DTI `{dti:.1%}`")

                with cr_res2:
                    st.metric("Default Probability", f"{cr_proba:.2%}", delta=f"{(cr_proba - cr_threshold):.2%} vs threshold", delta_color="inverse")

                with cr_res3:
                    st.progress(min(max(cr_proba, 0.0), 1.0))
                    if cr_proba < 0.20:
                        st.caption("✅ Excellent solvency profile; low probability of default.")
                    elif cr_proba < cr_threshold:
                        st.caption("⚡ Acceptable risk; monitor repayment under standard provisioning.")
                    else:
                        st.caption("⚠️ High probability of delinquency under IFRS 9 Stage 2/3 classification.")
            else:
                st.error("Model 'credit_risk_xgboost.joblib' not found in artifacts/models.")
