from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def render_segmentation(artifacts_dir: Path, load_json_artifact):
    """Renders Customer RFM Segmentation view."""
    st.title("👥 Customer RFM Segmentation")
    st.markdown("Recency, Frequency, and Monetary Value behavioral clustering on transactional records.")

    metrics_dir = artifacts_dir / "metrics"
    retail_data = load_json_artifact(metrics_dir / "retail_analytics_metrics.json")
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
    else:
        st.warning("No segmentation metrics found. Please run retail analytics pipeline.")
