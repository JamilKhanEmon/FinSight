from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def render_churn(artifacts_dir: Path, load_json_artifact, load_joblib_model):
    """Renders Customer Churn & Retention Analytics view."""
    st.title("🚪 Customer Churn & Retention Analytics")
    st.markdown("Assesses customer attrition probability across account balance, activity, and geographic segments.")

    metrics_dir = artifacts_dir / "metrics"
    churn_metrics = load_json_artifact(metrics_dir / "churn_metrics.json")
    tab_churn1, tab_churn2 = st.tabs(["📊 Performance & Benchmarks", "🔮 Live Customer Churn Risk Simulator"])

    with tab_churn1:
        if churn_metrics:
            rows = [
                {"Model": k, "PR-AUC": v["pr_auc"], "ROC-AUC": v["roc_auc"], "Precision": v["precision"], "Recall": v["recall"], "F1 Score": v["f1_score"]}
                for k, v in churn_metrics.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            fig = px.bar(pd.DataFrame(rows), x="Model", y=["ROC-AUC", "PR-AUC", "F1 Score"], barmode="group", title="Model Performance on Retention Test Holdout")
            st.plotly_chart(fig, use_container_width=True)

    with tab_churn2:
        st.subheader("🔮 Live Customer Churn Risk Simulator")
        st.markdown("Evaluate individual customer retention risk and generate personalized proactive retention actions.")

        col_c1, col_c2 = st.columns(2)
        c_preset = None
        with col_c1:
            if st.button("📋 Load High-Risk Profile (Senior, High Balance, 1 Product, Inactive)", use_container_width=True):
                c_preset = "high"
        with col_c2:
            if st.button("📋 Load Loyal/Stable Profile (Younger, Active, 2 Products)", use_container_width=True):
                c_preset = "loyal"

        if c_preset == "high":
            def_score, def_geo, def_gender, def_age, def_tenure, def_bal, def_prods, def_card, def_act, def_sal = (
                520, "Germany", "Female", 50, 3, 125000.0, 1, 1, 0, 85000.0
            )
        elif c_preset == "loyal":
            def_score, def_geo, def_gender, def_age, def_tenure, def_bal, def_prods, def_card, def_act, def_sal = (
                740, "France", "Male", 32, 6, 45000.0, 2, 1, 1, 62000.0
            )
        else:
            def_score, def_geo, def_gender, def_age, def_tenure, def_bal, def_prods, def_card, def_act, def_sal = (
                650, "France", "Female", 42, 5, 75000.0, 1, 1, 1, 60000.0
            )

        with st.form("churn_sim_form"):
            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                cs_val = st.number_input("Credit Score", 300, 850, int(def_score))
                geo_list = ["France", "Germany", "Spain"]
                geo_val = st.selectbox("Geography", geo_list, index=geo_list.index(def_geo))
                gender_val = st.selectbox("Gender", ["Female", "Male"], index=0 if def_gender == "Female" else 1)
                age_val = st.number_input("Customer Age", 18, 100, int(def_age))

            with c_f2:
                tenure_val = st.number_input("Tenure (Years with Bank)", 0, 10, int(def_tenure))
                balance_val = st.number_input("Account Balance ($)", 0.0, 300000.0, float(def_bal), step=1000.0)
                num_prods_val = st.selectbox("Number of Products", [1, 2, 3, 4], index=[1, 2, 3, 4].index(def_prods))

            with c_f3:
                card_val = st.selectbox("Has Credit Card?", ["Yes", "No"], index=0 if def_card == 1 else 1)
                act_val = st.selectbox("Is Active Member?", ["Yes", "No"], index=0 if def_act == 1 else 1)
                salary_val = st.number_input("Estimated Annual Salary ($)", 0.0, 300000.0, float(def_sal), step=5000.0)
                churn_th = st.slider("Churn Decision Cutoff Threshold", 0.10, 0.90, 0.40, 0.05)

            submitted_churn = st.form_submit_button("🚀 Run Live Churn Evaluation", use_container_width=True)

        if submitted_churn:
            churn_model = load_joblib_model("customer_churn_random_forest")
            if churn_model:
                c_data = {
                    "CreditScore": int(cs_val),
                    "Geography": str(geo_val),
                    "Gender": str(gender_val),
                    "Age": int(age_val),
                    "Tenure": int(tenure_val),
                    "Balance": float(balance_val),
                    "NumOfProducts": int(num_prods_val),
                    "HasCrCard": 1 if card_val == "Yes" else 0,
                    "IsActiveMember": 1 if act_val == "Yes" else 0,
                    "EstimatedSalary": float(salary_val),
                }
                c_proba = float(churn_model.predict_proba(pd.DataFrame([c_data]))[0, 1])

                st.markdown("### 🎯 Live Churn Prediction Result")
                res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1.5])
                with res_col1:
                    if c_proba >= churn_th:
                        st.error("🔴 ELEVATED RETENTION RISK (At-Risk of Churn)")
                        st.markdown("**Action**: Trigger proactive retention offer. Assign dedicated relationship manager.")
                    else:
                        st.success("🟢 STABLE ACCOUNT (Low Attrition Risk)")
                        st.markdown("**Action**: Standard relationship management. Good candidate for cross-selling.")

                with res_col2:
                    st.metric("Churn Probability", f"{c_proba:.2%}", delta=f"{(c_proba - churn_th):.2%} vs threshold")

                with res_col3:
                    st.progress(min(max(c_proba, 0.0), 1.0))
                    if c_proba >= 0.50:
                        st.caption("⚠️ High-attrition danger zone. High balance runoff risk.")
                    else:
                        st.caption("✅ Account relationship within safe retention parameters.")
            else:
                st.error("Model 'customer_churn_random_forest.joblib' not found in artifacts/models.")
