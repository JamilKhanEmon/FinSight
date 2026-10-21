from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def render_data_audit(artifacts_dir: Path, load_json_artifact):
    """Renders the dataset audit explorer with schema and missingness summaries."""
    st.title("🔍 Data Quality & Schema Audit Explorer")
    st.markdown("Automated data audits documenting dimensions, missingness, duplicates, and class distributions.")

    audit_dir = artifacts_dir / "data_audit"
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

    audit_data = load_json_artifact(audit_dir / f"{selected_key}_audit.json")
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
