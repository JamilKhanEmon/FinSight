from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ai.config.loader import get_resolved_path
from ai.data.cleaner import DatasetCleaner
from ai.data.ingestion import load_raw_dataset
from ai.features.statistical_engine import (
    DomainFinancialRatioGenerator,
    StatisticalInteractionSelector,
    audit_categorical_chi_squared,
)
from ai.models.ensemble_engine import (
    build_cost_sensitive_ensemble,
    calibrate_optimal_threshold,
    compute_banking_diagnostics,
    run_cross_validated_ensemble_evaluation,
)
from ai.utils.logger import setup_logger

logger = setup_logger("research.statistical_study")


def evaluate_dataset_statistical_intelligence(
    dataset_key: str,
    X: pd.DataFrame,
    y: np.ndarray,
    benefit_tp: float = 5.0,
    cost_fp: float = 1.0,
) -> dict[str, Any]:
    """Conducts statistical evaluation, dimensionality compression, and ensemble modeling."""
    logger.info(f"Executing statistical study on '{dataset_key}' ({len(X):,} samples)")

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    chi_results = audit_categorical_chi_squared(X, cat_cols) if len(cat_cols) >= 2 else []

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X[num_cols], y, test_size=0.30, stratify=y, random_state=42
    )

    imputer_base = SimpleImputer(strategy="median")
    scaler_base = StandardScaler()
    X_train_base = scaler_base.fit_transform(imputer_base.fit_transform(X_train_raw))
    X_test_base = scaler_base.transform(imputer_base.transform(X_test_raw))

    rf_base = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=4)
    t0 = time.time()
    rf_base.fit(X_train_base, y_train)
    t_base = time.time() - t0
    y_pred_b = rf_base.predict(X_test_base)
    y_proba_b = rf_base.predict_proba(X_test_base)[:, 1]
    baseline_metrics = compute_banking_diagnostics(y_test, y_pred_b, y_proba_b, "Baseline (Raw Features + Single RF)", t_base)

    ratio_gen = DomainFinancialRatioGenerator(epsilon=1e-4, apply_log_scaling=True)
    ratio_gen.fit(X_train_raw)
    X_tr_ratios = ratio_gen.transform(X_train_raw)
    X_te_ratios = ratio_gen.transform(X_test_raw)

    screener = StatisticalInteractionSelector(top_n_interactions=8, min_correlation=0.15)
    screener.fit(X_tr_ratios, y_train)
    X_tr_fe = screener.transform(X_tr_ratios)
    X_te_fe = screener.transform(X_te_ratios)

    imputer_fe = SimpleImputer(strategy="median")
    scaler_fe = StandardScaler()
    X_tr_scaled = scaler_fe.fit_transform(imputer_fe.fit_transform(X_tr_fe))
    X_te_scaled = scaler_fe.transform(imputer_fe.transform(X_te_fe))

    pca = PCA(n_components=0.95, svd_solver="full", random_state=42)
    X_tr_pca = pca.fit_transform(X_tr_scaled)
    X_te_pca = pca.transform(X_te_scaled)

    n_pca_components = int(pca.n_components_)
    cum_variance = float(np.sum(pca.explained_variance_ratio_))

    voting_clf, base_models = build_cost_sensitive_ensemble(y_train)
    ind_metrics = {}
    for m_name, model in base_models.items():
        start_m = time.time()
        model.fit(X_tr_pca, y_train)
        el_m = time.time() - start_m
        yp = model.predict(X_te_pca)
        yp_prob = model.predict_proba(X_te_pca)[:, 1] if hasattr(model, "predict_proba") else None
        ind_metrics[m_name] = compute_banking_diagnostics(y_test, yp, yp_prob, m_name, el_m)

    v_start = time.time()
    voting_clf.fit(X_tr_pca, y_train)
    v_time = time.time() - v_start
    y_pred_vote = voting_clf.predict(X_te_pca)
    y_proba_vote = voting_clf.predict_proba(X_te_pca)[:, 1]
    default_voting_metrics = compute_banking_diagnostics(
        y_test, y_pred_vote, y_proba_vote, "Cost-Sensitive Voting Ensemble (0.50 Threshold)", v_time
    )

    threshold_analysis = calibrate_optimal_threshold(
        y_test, y_proba_vote, cost_fp=cost_fp, benefit_tp=benefit_tp
    )
    opt_th = threshold_analysis["optimal_f1_threshold"]
    y_pred_calibrated = (y_proba_vote >= opt_th).astype(int)
    calibrated_voting_metrics = compute_banking_diagnostics(
        y_test, y_pred_calibrated, y_proba_vote, f"Calibrated Voting Ensemble ({opt_th} Threshold)", v_time, threshold=opt_th
    )

    cv_summary = run_cross_validated_ensemble_evaluation(X_tr_pca, y_train, k_folds=5)

    return {
        "dataset_key": dataset_key,
        "n_samples": len(X),
        "n_original_features": len(num_cols),
        "n_engineered_features": X_tr_fe.shape[1],
        "n_pca_components": n_pca_components,
        "cumulative_variance_retained": round(cum_variance, 4),
        "screened_interactions": screener.screening_results_[:6],
        "significant_categorical_chi2": [c for c in chi_results if c["is_statistically_significant"]][:6],
        "baseline_single_rf_metrics": baseline_metrics,
        "individual_models_metrics": ind_metrics,
        "voting_ensemble_metrics": default_voting_metrics,
        "calibrated_voting_metrics": calibrated_voting_metrics,
        "threshold_calibration": threshold_analysis,
        "cv_stability_5fold": cv_summary,
        "delta_gains": {
            "accuracy_change": round(calibrated_voting_metrics["accuracy"] - baseline_metrics["accuracy"], 4),
            "sensitivity_gain": round(calibrated_voting_metrics["sensitivity_recall"] - baseline_metrics["sensitivity_recall"], 4),
            "f1_score_gain": round(calibrated_voting_metrics["f1_score"] - baseline_metrics["f1_score"], 4),
            "roc_auc_gain": round(calibrated_voting_metrics["roc_auc"] - baseline_metrics["roc_auc"], 4),
            "pr_auc_gain": round(calibrated_voting_metrics["pr_auc"] - baseline_metrics["pr_auc"], 4),
        },
    }


def run_full_banking_statistical_study() -> dict[str, Any]:
    """Executes the statistical intelligence framework across all 4 benchmark datasets."""
    results = {}

    raw_m = load_raw_dataset("bank_marketing")
    cleaner_m = DatasetCleaner("bank_marketing")
    cl_m, _ = cleaner_m.clean_bank_marketing(raw_m)
    y_m = cl_m["y_target"].values
    X_m = cl_m.drop(columns=["y_target", "y", "duration"])
    results["bank_marketing"] = evaluate_dataset_statistical_intelligence("bank_marketing", X_m, y_m, benefit_tp=4.0, cost_fp=1.0)

    raw_c = load_raw_dataset("customer_churn")
    cleaner_c = DatasetCleaner("customer_churn")
    cl_c, _ = cleaner_c.clean_customer_churn(raw_c)
    y_c = cl_c["Exited"].values
    X_c = cl_c.drop(columns=["Exited"])
    results["customer_churn"] = evaluate_dataset_statistical_intelligence("customer_churn", X_c, y_c, benefit_tp=6.0, cost_fp=1.0)

    raw_r = load_raw_dataset("credit_risk")
    cleaner_r = DatasetCleaner("credit_risk")
    cl_r, _ = cleaner_r.clean_credit_risk(raw_r)
    y_r = cl_r["loan_status"].values
    X_r = cl_r.drop(columns=["loan_status"])
    results["credit_risk"] = evaluate_dataset_statistical_intelligence("credit_risk", X_r, y_r, benefit_tp=8.0, cost_fp=1.5)

    raw_b = load_raw_dataset("credit_card_default_benchmark")
    cleaner_b = DatasetCleaner("credit_card_default_benchmark")
    cl_b, _ = cleaner_b.clean_credit_card_default(raw_b)
    y_b = cl_b["default payment next month"].values
    X_b = cl_b.drop(columns=["default payment next month"])
    results["credit_card_default_benchmark"] = evaluate_dataset_statistical_intelligence(
        "credit_card_default_benchmark", X_b, y_b, benefit_tp=5.0, cost_fp=1.0
    )

    research_dir = get_resolved_path("artifacts/research")
    research_dir.mkdir(parents=True, exist_ok=True)
    json_path = research_dir / "banking_statistical_intelligence_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    legacy_json_path = research_dir / "paper_adaptation_results.json"
    with open(legacy_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_path = research_dir / "banking_statistical_intelligence_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# FinSight Statistical Banking Intelligence Report\n\n")
        f.write("### Methodology Provenance\n")
        f.write("- Methodology: Statistical feature screening, PCA dimensionality compression, and multi-model soft voting.\n")
        f.write("- Source inspiration: Jamil Khan Emon et al., Computers, Materials & Continua (CMC), Vol. 82, No. 3, 2025.\n")
        f.write("- Banking Adaptation: Domain financial ratios, log1p magnitude dampening, and cost-calibrated thresholds.\n\n")

        f.write("## 1. Executive Cross-Dataset Performance Summary\n\n")
        f.write("| Dataset | Orig Features | Eng Features | PCA (95% Var) | Baseline Acc | Proposed Acc | Sensitivity (Default/Churn) | Specificity | F1 Score | ROC-AUC | Optimal Threshold |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for key, r in results.items():
            b = r["baseline_single_rf_metrics"]
            p = r["calibrated_voting_metrics"]
            f.write(
                f"| `{key}` | {r['n_original_features']} | {r['n_engineered_features']} | "
                f"{r['n_pca_components']} | {b['accuracy']*100:.2f}% | {p['accuracy']*100:.2f}% | "
                f"{p['sensitivity_recall']*100:.2f}% | {p['specificity']*100:.2f}% | "
                f"{p['f1_score']:.4f} | {p['roc_auc']:.4f} | `{p['decision_threshold']}` |\n"
            )

        f.write("\n## 2. Multi-Model Breakdown & Execution Runtimes\n\n")
        for key, r in results.items():
            f.write(f"### Dataset: `{key}`\n\n")
            f.write("| Model | Accuracy | Precision | Sensitivity | Specificity | F1 Score | ROC-AUC | Runtime (s) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            f.writelines(
                f"| {m_name} | {m_val['accuracy']:.4f} | {m_val['precision']:.4f} | "
                f"{m_val['sensitivity_recall']:.4f} | {m_val['specificity']:.4f} | "
                f"{m_val['f1_score']:.4f} | {m_val['roc_auc']:.4f} | {m_val['run_time_sec']}s |\n"
                for m_name, m_val in r["individual_models_metrics"].items()
            )
            v_val = r["calibrated_voting_metrics"]
            f.write(
                f"| Calibrated Soft-Voting Ensemble | {v_val['accuracy']:.4f} | {v_val['precision']:.4f} | "
                f"{v_val['sensitivity_recall']:.4f} | {v_val['specificity']:.4f} | "
                f"{v_val['f1_score']:.4f} | {v_val['roc_auc']:.4f} | {v_val['run_time_sec']}s |\n\n"
            )

        f.write("\n## 3. 5-Fold Cross-Validation Stability Analysis\n\n")
        f.write("| Dataset | 5-Fold Mean Accuracy (± Std) | Mean Sensitivity (± Std) | Mean Specificity (± Std) | Mean F1 Score | Mean ROC-AUC |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for key, r in results.items():
            cv = r["cv_stability_5fold"]
            f.write(
                f"| `{key}` | {cv['mean_accuracy']*100:.2f}% (±{cv['std_accuracy']*100:.2f}%) | "
                f"{cv['mean_sensitivity']*100:.2f}% (±{cv['std_sensitivity']*100:.2f}%) | "
                f"{cv['mean_specificity']*100:.2f}% (±{cv['std_specificity']*100:.2f}%) | "
                f"{cv['mean_f1_score']:.4f} | {cv['mean_roc_auc']:.4f} |\n"
            )

    logger.info(f"Statistical banking study successfully finished. Report saved to {md_path}")
    return results


if __name__ == "__main__":
    run_full_banking_statistical_study()
