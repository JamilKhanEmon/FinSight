from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def render_research(artifacts_dir: Path, load_json_artifact, load_text_artifact):
    """Renders Statistical Intelligence & Adaptive Ensemble Workbench view."""
    st.title("🔬 Statistical Intelligence & Adaptive Ensemble Workbench")
    st.markdown("""
    **Methodology Provenance**: Derived from empirical research on pairwise feature screening, 
    PCA variance retention, and heterogeneous soft-voting ensembles (*Jamil Khan Emon et al., CMC 2025*), 
    enhanced with domain financial ratios, magnitude stabilization, and asymmetric threshold calibration.
    """)

    research_dir = artifacts_dir / "research"
    paper_results = load_json_artifact(research_dir / "banking_statistical_intelligence_results.json")
    if not paper_results:
        paper_results = load_json_artifact(research_dir / "paper_adaptation_results.json")

    if paper_results:
        st.subheader("📊 Executive Cross-Dataset Benchmark Leaderboard")
        rows = []
        for d_key, r in paper_results.items():
            b = r.get("baseline_single_rf_metrics") or r.get("baseline", {})
            p = r.get("calibrated_voting_metrics") or r.get("proposed_voting_metrics") or r.get("proposed", {})
            rows.append({
                "Dataset": d_key,
                "Pipeline": "Baseline (Single RF, Raw Features)",
                "Orig Features": r.get("n_original_features", "-"),
                "Eng Features": "-",
                "PCA (95% Var)": "-",
                "Accuracy": f"{b.get('accuracy', 0):.2%}",
                "Sensitivity (Recall)": f"{b.get('sensitivity_recall', 0):.2%}",
                "Specificity": f"{b.get('specificity', 0):.2%}",
                "F1 Score": f"{b.get('f1_score', 0):.4f}",
                "ROC-AUC": f"{b.get('roc_auc', 0):.4f}",
                "Optimal Threshold": "0.500",
            })
            rows.append({
                "Dataset": d_key,
                "Pipeline": "⭐ Proposed (Statistical Screening + PCA 95% + Calibrated Ensemble)",
                "Orig Features": r.get("n_original_features", "-"),
                "Eng Features": r.get("n_engineered_features", "-"),
                "PCA (95% Var)": r.get("n_pca_components", "-"),
                "Accuracy": f"{p.get('accuracy', 0):.2%}",
                "Sensitivity (Recall)": f"{p.get('sensitivity_recall', 0):.2%}",
                "Specificity": f"{p.get('specificity', 0):.2%}",
                "F1 Score": f"{p.get('f1_score', 0):.4f}",
                "ROC-AUC": f"{p.get('roc_auc', 0):.4f}",
                "Optimal Threshold": f"{p.get('decision_threshold', 0.5):.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("🔍 Deep-Dive: Dataset Statistical Intelligence & Model Diagnostics")
        selected_ds = st.selectbox("Select Dataset to Inspect:", list(paper_results.keys()))
        if selected_ds in paper_results:
            ds_data = paper_results[selected_ds]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Original Features", ds_data.get("n_original_features"))
            c2.metric("Engineered Features", ds_data.get("n_engineered_features"))
            c3.metric("PCA Components (95% Var)", ds_data.get("n_pca_components"))
            p_val = ds_data.get("calibrated_voting_metrics", {})
            c4.metric("Optimal Cutoff Threshold", f"{p_val.get('decision_threshold', 0.5):.3f}")

            ind_models = ds_data.get("individual_models_metrics", {})
            if ind_models:
                st.markdown("##### Heterogeneous Models vs. Calibrated Soft-Voting Ensemble Performance")
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
                v_val = ds_data.get("calibrated_voting_metrics") or ds_data.get("proposed_voting_metrics", {})
                m_rows.append({
                    "Model": "⭐ Calibrated Soft-Voting Ensemble",
                    "Accuracy": f"{v_val.get('accuracy', 0):.2%}",
                    "Precision": f"{v_val.get('precision', 0):.2%}",
                    "Sensitivity (Recall)": f"{v_val.get('sensitivity_recall', 0):.2%}",
                    "Specificity": f"{v_val.get('specificity', 0):.2%}",
                    "F1 Score": f"{v_val.get('f1_score', 0):.4f}",
                    "ROC-AUC": f"{v_val.get('roc_auc', 0):.4f}",
                    "Runtime (s)": f"{v_val.get('run_time_sec', 0):.3f}s",
                })
                st.dataframe(pd.DataFrame(m_rows), use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("##### Top Statistically Screened Interactions")
                inter_list = ds_data.get("screened_interactions", [])
                if inter_list:
                    st.dataframe(pd.DataFrame(inter_list), use_container_width=True)
                else:
                    st.info("No interaction pairs recorded.")

            with col_b:
                st.markdown("##### Significant Categorical Associations (Chi-Square p < 0.05)")
                chi_list = ds_data.get("significant_categorical_chi2", [])
                if chi_list:
                    st.dataframe(pd.DataFrame(chi_list), use_container_width=True)
                else:
                    st.info("No significant categorical pairs or purely numerical dataset.")

            cv_res = ds_data.get("cv_stability_5fold", {})
            if cv_res:
                st.markdown("##### 5-Fold Stratified Cross-Validation Stability")
                cv_c1, cv_c2, cv_c3, cv_c4 = st.columns(4)
                cv_c1.metric("Mean Accuracy", f"{cv_res.get('mean_accuracy', 0):.2%}", f"±{cv_res.get('std_accuracy', 0):.2%}")
                cv_c2.metric("Mean Sensitivity", f"{cv_res.get('mean_sensitivity', 0):.2%}", f"±{cv_res.get('std_sensitivity', 0):.2%}")
                cv_c3.metric("Mean Specificity", f"{cv_res.get('mean_specificity', 0):.2%}", f"±{cv_res.get('std_specificity', 0):.2%}")
                cv_c4.metric("Mean F1 Score", f"{cv_res.get('mean_f1_score', 0):.4f}")

    stat_report_md = load_text_artifact(research_dir / "banking_statistical_intelligence_report.md")
    if not stat_report_md:
        stat_report_md = load_text_artifact(research_dir / "paper_methodology_and_adaptation_report.md")

    if stat_report_md:
        with st.expander("📖 View Full Statistical Intelligence & Executive Report", expanded=False):
            st.markdown(stat_report_md)
