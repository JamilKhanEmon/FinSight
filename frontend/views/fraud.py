from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_fraud(artifacts_dir: Path, load_json_artifact, load_joblib_model):
    """Renders Transaction Fraud Detection view."""
    st.title("🛡️ Transaction Fraud Detection & Anomaly Monitoring")
    st.markdown("Monitors imbalanced transaction streams (0.17% fraud rate) with precision-recall sensitivity.")

    metrics_dir = artifacts_dir / "metrics"
    fraud_metrics = load_json_artifact(metrics_dir / "fraud_metrics.json")
    tab_fr1, tab_fr2 = st.tabs(["📊 Fraud PR Curves & Evaluation", "🔮 Live Transaction Fraud Screener"])

    with tab_fr1:
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

    with tab_fr2:
        st.subheader("🔮 Live Transaction Fraud Screener")
        st.markdown("Screens transactional velocity, amount, and latent PCA anomaly patterns to block unauthorized attacks in milliseconds.")

        col_fr1, col_fr2 = st.columns(2)
        fr_preset = None
        with col_fr1:
            if st.button("📋 Load Normal Grocery POS Transaction ($42.50, Daytime)", use_container_width=True):
                fr_preset = "normal"
        with col_fr2:
            if st.button("📋 Load Suspicious High-Velocity Card Attack ($1,850.00, Night)", use_container_width=True):
                fr_preset = "fraud"

        if fr_preset == "fraud":
            def_amt, def_time, def_v1, def_v2, def_v3, def_v4, def_v14 = (
                1850.0, 7200.0, -4.5, 3.8, -6.2, 4.1, -7.5
            )
        else:
            def_amt, def_time, def_v1, def_v2, def_v3, def_v4, def_v14 = (
                42.50, 48000.0, 0.1, -0.2, 0.3, 0.05, 0.1
            )

        with st.form("fraud_sim_form"):
            fr_c1, fr_c2, fr_c3 = st.columns(3)
            with fr_c1:
                tx_amt = st.number_input("Transaction Amount ($)", 0.01, 10000.0, float(def_amt), step=10.0)
                tx_time = st.number_input("Observation Timestamp (Seconds)", 0.0, 172800.0, float(def_time), step=1000.0)
            with fr_c2:
                v1_val = st.number_input("Latent Feature V1", -20.0, 20.0, float(def_v1), step=0.5)
                v2_val = st.number_input("Latent Feature V2", -20.0, 20.0, float(def_v2), step=0.5)
                v3_val = st.number_input("Latent Feature V3", -20.0, 20.0, float(def_v3), step=0.5)
            with fr_c3:
                v4_val = st.number_input("Latent Feature V4", -20.0, 20.0, float(def_v4), step=0.5)
                v14_val = st.number_input("Latent Feature V14 (Key Anomaly Indicator)", -20.0, 20.0, float(def_v14), step=0.5)
                fr_cutoff = st.slider("Fraud Risk Score Alert Threshold", 0.01, 0.50, 0.05, 0.01)

            submitted_fr = st.form_submit_button("🚀 Screen Transaction", use_container_width=True)

        if submitted_fr:
            fr_model = load_joblib_model("fraud_detection_xgboost")
            if fr_model:
                fr_payload = {"Time": float(tx_time), "Amount": float(tx_amt)}
                for i in range(1, 29):
                    fr_payload[f"V{i}"] = 0.0
                fr_payload["V1"] = float(v1_val)
                fr_payload["V2"] = float(v2_val)
                fr_payload["V3"] = float(v3_val)
                fr_payload["V4"] = float(v4_val)
                fr_payload["V14"] = float(v14_val)

                fr_proba = float(fr_model.predict_proba(pd.DataFrame([fr_payload]))[0, 1])

                st.markdown("### 🎯 Live Fraud Screening Result")
                fr_r1, fr_r2, fr_r3 = st.columns([1.5, 1, 1.5])
                with fr_r1:
                    if fr_proba >= fr_cutoff:
                        st.error("🚨 FRAUD ATTACK SUSPECTED (Transaction Blocked)")
                        st.markdown("**Action**: Transaction declined. Automated SMS sent to cardholder for Step-up 2FA verification.")
                    else:
                        st.success("🟢 TRANSACTION APPROVED (Normal Behavior)")
                        st.markdown("**Action**: Payment cleared within <5ms. Low anomaly risk.")

                with fr_r2:
                    st.metric("Fraud Risk Score", f"{fr_proba:.3%}", delta=f"{(fr_proba - fr_cutoff):.3%} vs alert cutoff", delta_color="inverse")

                with fr_r3:
                    st.progress(min(max(fr_proba * 5, 0.0), 1.0))
                    if fr_proba >= fr_cutoff:
                        st.caption("⚠️ Transaction shows high statistical divergence from cardholder baseline.")
                    else:
                        st.caption("✅ Transaction conforms to legitimate spending pattern.")
            else:
                st.error("Model 'fraud_detection_xgboost.joblib' not found in artifacts/models.")
