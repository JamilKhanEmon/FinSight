from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def render_leaderboard(artifacts_dir: Path, load_json_artifact):
    """Renders Consolidated Cross-Task Model Performance Leaderboard."""
    st.title("⚖️ Cross-Task Model Performance Comparison")
    st.markdown("Consolidated model leaderboard across all active banking and benchmark tasks.")

    metrics_dir = artifacts_dir / "metrics"
    tasks = [
        ("marketing_prospective", "Bank Marketing (Prospective)"),
        ("churn", "Customer Churn"),
        ("credit_risk", "Credit Risk Default"),
        ("fraud", "Fraud Detection"),
        ("credit_card_default_benchmark", "Credit Card Default Benchmark"),
    ]

    all_rows = []
    for t_key, t_label in tasks:
        m_dict = load_json_artifact(metrics_dir / f"{t_key}_metrics.json")
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

        fig = px.scatter(
            df_all,
            x="ROC-AUC",
            y="PR-AUC",
            color="Task",
            hover_name="Model",
            title="Cross-Task ROC-AUC vs PR-AUC Landscape",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No model metrics found. Please execute the pipeline tasks.")
