from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def render_forecasting(artifacts_dir: Path, load_json_artifact):
    """Renders Transaction Trends & Forecasting view."""
    st.title("📈 Transaction Demand & Revenue Forecasting")
    st.markdown("Evaluates daily revenue time series forecasting models on a 28-day chronological holdout split.")

    metrics_dir = artifacts_dir / "metrics"
    retail_data = load_json_artifact(metrics_dir / "retail_analytics_metrics.json")
    if retail_data and "forecasting" in retail_data:
        fc = retail_data["forecasting"]
        rows = [
            {"Model": "7-Day Naive Seasonal", "MAE": f"£{fc['naive_seasonal_7d']['mae']:,.2f}", "RMSE": f"£{fc['naive_seasonal_7d']['rmse']:,.2f}", "MAPE": f"{fc['naive_seasonal_7d']['mape']:.2%}"},
            {"Model": "7-Day Moving Average", "MAE": f"£{fc['moving_average_7d']['mae']:,.2f}", "RMSE": f"£{fc['moving_average_7d']['rmse']:,.2f}", "MAPE": f"{fc['moving_average_7d']['mape']:.2%}"},
            {"Model": "Holt-Winters Exponential Smoothing", "MAE": f"£{fc['holt_winters_exponential_smoothing']['mae']:,.2f}", "RMSE": f"£{fc['holt_winters_exponential_smoothing']['rmse']:,.2f}", "MAPE": f"{fc['holt_winters_exponential_smoothing']['mape']:.2%}"},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.info(f"📅 Holdout Evaluation Window: {fc['test_period']} ({fc['test_days']} days)")
    else:
        st.warning("No forecasting metrics found. Please run retail analytics pipeline.")
