from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_documentation(artifacts_dir: Path, load_text_artifact):
    """Renders Business Recommendations, KPI Dictionary, and BRD."""
    st.title("📚 FinSight Platform Documentation & Strategy")

    business_dir = artifacts_dir / "business"
    tab1, tab2, tab3 = st.tabs(["KPI Dictionary", "Business Requirements", "Strategic Recommendations"])

    with tab1:
        kpi_md = load_text_artifact(business_dir / "kpi_dictionary.md")
        if kpi_md:
            st.markdown(kpi_md)
        else:
            st.info("KPI dictionary file not found in artifacts/business.")

    with tab2:
        brd_md = load_text_artifact(business_dir / "business_requirements_document.md")
        if brd_md:
            st.markdown(brd_md)
        else:
            st.info("Business requirements document not found in artifacts/business.")

    with tab3:
        rec_md = load_text_artifact(business_dir / "business_recommendations.md")
        if rec_md:
            st.markdown(rec_md)
        else:
            st.info("Recommendations document not found in artifacts/business.")
