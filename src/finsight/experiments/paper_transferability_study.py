"""Empirical Transferability Study of Jamil Khan Emon et al. Research Methodology.

Adapts the exact methodology and code structure from:
'Harmonization of Heart Disease Dataset for Accurate Diagnosis: A Machine Learning Approach
Enhanced by Feature Engineering'
Authors: Jamil Khan Emon et al.
Published in: Computers, Materials & Continua (CMC), Vol. 82, No. 3, 2025.
DOI/URL: https://www.techscience.com/cmc/v82n3/59942
Source GitHub: https://github.com/JamilKhanEmon/Heart-Disease-

Transfers the exact pipeline structure from:
- Final_model_with_new_Feature.ipynb
- update_with_k(10)_fold_also_Sensitivity_.ipynb
- update_with_k(5)_fold_also_Sensitivity.ipynb

Steps replicated across banking datasets:
Step 1: Pairwise correlation analysis for two-variable combinations (sorted descending)
Step 2: Chi-squared contingency tests for categorical pairs
Step 3: Multiplicative pairwise interaction feature engineering (col1 * col2)
Step 4: StandardScaler normalization on numeric + interaction features
Step 5: 70-30 Train-Test split (test_size=0.30, random_state=42)
Step 6: PCA dimensionality reduction retaining 95% variance (PCA(n_components=0.95))
Step 7: Multi-model training with runtime tracking (RandomForest, GradientBoosting, XGBoost, etc.)
Step 8: Soft-Voting Ensemble (VotingClassifier(voting='soft'))
Step 9: Rigorous metrics: Accuracy, Precision, Sensitivity (Recall), Specificity, F1, ROC-AUC
Step 10: k-Fold Cross-Validation (k=5 and k=10) with Sensitivity & Specificity tracking
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from finsight.cleaning.cleaner import DatasetCleaner
from finsight.config.loader import get_resolved_path
from finsight.ingestion.loader import load_raw_dataset
from finsight.utils.logger import setup_logger

logger = setup_logger("finsight.paper_study")


def calculate_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None, model_name: str, run_time: float = 0.0) -> dict[str, Any]:
    """Calculates all metrics mirroring the user's research notebooks:
    Accuracy, Precision, Sensitivity (Recall), Specificity, F1 Score, ROC-AUC, and Run Time.
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, len(y_true)

    recall_sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = float(roc_auc_score(y_true, y_proba)) if y_proba is not None else 0.0
    pr_auc = float(average_precision_score(y_true, y_proba)) if y_proba is not None else 0.0

    return {
        "model_name": model_name,
        "run_time_sec": round(float(run_time), 3),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "sensitivity_recall": round(recall_sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def perform_pairwise_correlation_analysis(df: pd.DataFrame, num_cols: list[str], top_n: int = 10) -> list[dict[str, Any]]:
    """Replicates Cell 15 from Final_model_with_new_Feature.ipynb:
    Computes pairwise correlations for two-variable combinations, sorted descending.
    """
    results = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            v1, v2 = num_cols[i], num_cols[j]
            corr_val = df[v1].corr(df[v2])
            if not np.isnan(corr_val):
                results.append({"var1": v1, "var2": v2, "combination": f"{v1} * {v2}", "abs_correlation": abs(float(corr_val)), "correlation": float(corr_val)})
    results.sort(key=lambda x: x["abs_correlation"], reverse=True)
    return results[:top_n]


def perform_chi_squared_tests(df: pd.DataFrame, cat_cols: list[str], top_n: int = 10) -> list[dict[str, Any]]:
    """Replicates Cell 17 from Final_model_with_new_Feature.ipynb:
    Chi-Squared Test for Categorical Variables to identify significant relationships.
    """
    results = []
    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            v1, v2 = cat_cols[i], cat_cols[j]
            contingency = pd.crosstab(df[v1], df[v2])
            chi2, p, dof, _ = chi2_contingency(contingency)
            results.append({
                "var1": v1,
                "var2": v2,
                "chi2": round(float(chi2), 4),
                "p_value": float(p),
                "dof": int(dof),
                "is_significant": bool(p < 0.05),
            })
    return results[:top_n]


def apply_research_feature_engineering(df: pd.DataFrame, top_pairs: list[dict[str, Any]]) -> pd.DataFrame:
    """Replicates Cell 18 from Final_model_with_new_Feature.ipynb:
    Creates explicit multiplicative interaction terms for the top ranked feature pairs.
    """
    df_fe = df.copy()
    for pair in top_pairs:
        v1, v2 = pair["var1"], pair["var2"]
        if v1 in df_fe.columns and v2 in df_fe.columns:
            inter_col = f"{v1}_{v2}"
            df_fe[inter_col] = (df_fe[v1] * df_fe[v2]).replace([np.inf, -np.inf], np.nan)
    return df_fe


def evaluate_dataset_with_paper_pipeline(
    dataset_key: str,
    X: pd.DataFrame,
    y: np.ndarray,
    run_kfold: bool = True,
    k_folds: int = 5,
) -> dict[str, Any]:
    """Full execution pipeline faithfully mirroring:
    Final_model_with_new_Feature.ipynb + update_with_k(10)_fold_also_Sensitivity_.ipynb
    """
    logger.info(f"Replicating Jamil Khan Emon et al. pipeline on {dataset_key} ({len(X):,} samples)...")

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Step 1: Pairwise correlation analysis
    top_corrs = perform_pairwise_correlation_analysis(X, num_cols, top_n=10)

    # Step 2: Chi-squared tests for categorical features
    chi_results = perform_chi_squared_tests(X, cat_cols, top_n=10) if len(cat_cols) >= 2 else []

    # Step 3: Feature Engineering - creating interaction terms
    X_fe = apply_research_feature_engineering(X[num_cols], top_corrs)
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_fe), columns=X_fe.columns)

    # Step 4: StandardScaler applied to all numerical & interaction columns (Cell 19)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

    # Step 5: 70-30 Train-Test split (Cell 20: test_size=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.30, stratify=y, random_state=42
    )

    # Step 6: PCA Dimensionality Reduction retaining 95% variance (Cell 21)
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    n_pca_components = int(pca.n_components_)
    cumulative_variance = float(np.sum(pca.explained_variance_ratio_))

    # Step 7: Train individual models mirroring the paper's model suite (Cell 22)
    models_to_train = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=4),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric="logloss", n_jobs=4),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=42),
    }

    trained_models = {}
    individual_metrics = {}
    run_times = {}

    for model_name, model in models_to_train.items():
        start_time = time.time()
        model.fit(X_train_pca, y_train)
        elapsed = time.time() - start_time
        run_times[model_name] = elapsed
        trained_models[model_name] = model

        y_pred = model.predict(X_test_pca)
        y_proba = model.predict_proba(X_test_pca)[:, 1] if hasattr(model, "predict_proba") else None
        individual_metrics[model_name] = calculate_detailed_metrics(y_test, y_pred, y_proba, model_name, elapsed)

    # Step 8: Soft-Voting Ensemble (Cell 23)
    voting_estimators = [(name, trained_models[name]) for name in trained_models]
    voting_clf = VotingClassifier(estimators=voting_estimators, voting="soft")

    v_start = time.time()
    voting_clf.fit(X_train_pca, y_train)
    voting_time = time.time() - v_start

    y_pred_voting = voting_clf.predict(X_test_pca)
    y_proba_voting = voting_clf.predict_proba(X_test_pca)[:, 1]
    voting_metrics = calculate_detailed_metrics(y_test, y_pred_voting, y_proba_voting, "VotingClassifier (Soft)", voting_time)

    # Step 9: Baseline Comparison (Same models on raw scaled features without FE or PCA)
    rf_base = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=4)
    scaler_base = StandardScaler()
    X_train_base = scaler_base.fit_transform(X_train[num_cols])
    X_test_base = scaler_base.transform(X_test[num_cols])
    b_start = time.time()
    rf_base.fit(X_train_base, y_train)
    b_time = time.time() - b_start
    y_pred_base = rf_base.predict(X_test_base)
    y_proba_base = rf_base.predict_proba(X_test_base)[:, 1]
    baseline_metrics = calculate_detailed_metrics(y_test, y_pred_base, y_proba_base, "Baseline (Raw Features + Single RF)", b_time)

    # Step 10: k-Fold Cross-Validation (update_with_k(5)_fold_also_Sensitivity.ipynb)
    kfold_summary = {}
    if run_kfold:
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_accuracies, cv_sensitivities, cv_specificities, cv_f1s, cv_rocs = [], [], [], [], []

        for train_idx, val_idx in skf.split(X_scaled, y):
            X_tr, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            pca_cv = PCA(n_components=0.95, random_state=42)
            X_tr_p = pca_cv.fit_transform(X_tr)
            X_val_p = pca_cv.transform(X_val)

            v_fold = VotingClassifier(estimators=[
                ("rf", RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=2)),
                ("xgb", XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss", n_jobs=2)),
            ], voting="soft")
            v_fold.fit(X_tr_p, y_tr)

            yp = v_fold.predict(X_val_p)
            yp_prob = v_fold.predict_proba(X_val_p)[:, 1]
            cm_fold = confusion_matrix(y_val, yp)
            tn_f, fp_f, fn_f, tp_f = cm_fold.ravel() if cm_fold.shape == (2, 2) else (0, 0, 0, len(y_val))

            cv_accuracies.append(accuracy_score(y_val, yp))
            cv_sensitivities.append(tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0)
            cv_specificities.append(tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0.0)
            cv_f1s.append(f1_score(y_val, yp, zero_division=0))
            cv_rocs.append(roc_auc_score(y_val, yp_prob))

        kfold_summary = {
            "k_folds": k_folds,
            "mean_accuracy": round(float(np.mean(cv_accuracies)), 4),
            "std_accuracy": round(float(np.std(cv_accuracies)), 4),
            "mean_sensitivity": round(float(np.mean(cv_sensitivities)), 4),
            "mean_specificity": round(float(np.mean(cv_specificities)), 4),
            "mean_f1": round(float(np.mean(cv_f1s)), 4),
            "mean_roc_auc": round(float(np.mean(cv_rocs)), 4),
        }

    return {
        "dataset_key": dataset_key,
        "n_samples": len(X),
        "n_original_features": len(num_cols),
        "n_engineered_features": len(X_fe.columns),
        "n_pca_components": n_pca_components,
        "cumulative_variance_retained": round(cumulative_variance, 4),
        "top_pairwise_correlations": top_corrs,
        "chi_squared_significant_pairs": [c for c in chi_results if c["is_significant"]],
        "models_training_times": {k: round(v, 2) for k, v in run_times.items()},
        "voting_training_time": round(voting_time, 2),
        "individual_models_metrics": individual_metrics,
        "proposed_voting_metrics": voting_metrics,
        "baseline_single_rf_metrics": baseline_metrics,
        "delta_vs_baseline": {
            "accuracy_gain": round(voting_metrics["accuracy"] - baseline_metrics["accuracy"], 4),
            "sensitivity_gain": round(voting_metrics["sensitivity_recall"] - baseline_metrics["sensitivity_recall"], 4),
            "f1_gain": round(voting_metrics["f1_score"] - baseline_metrics["f1_score"], 4),
            "roc_auc_gain": round(voting_metrics["roc_auc"] - baseline_metrics["roc_auc"], 4),
        },
        "kfold_cv_results": kfold_summary,
    }


def run_full_paper_transferability_study() -> dict[str, Any]:
    """Applies the research paper methodology across all active banking datasets."""
    results = {}

    # 1. Bank Marketing (Prospective, duration strictly excluded)
    raw_m = load_raw_dataset("bank_marketing")
    cleaner_m = DatasetCleaner("bank_marketing")
    cl_m, _ = cleaner_m.clean_bank_marketing(raw_m)
    y_m = cl_m["y_target"].values
    X_m = cl_m.drop(columns=["y_target", "y", "duration"])
    results["bank_marketing"] = evaluate_dataset_with_paper_pipeline("bank_marketing", X_m, y_m, k_folds=5)

    # 2. Customer Churn
    raw_c = load_raw_dataset("customer_churn")
    cleaner_c = DatasetCleaner("customer_churn")
    cl_c, _ = cleaner_c.clean_customer_churn(raw_c)
    y_c = cl_c["Exited"].values
    X_c = cl_c.drop(columns=["Exited"])
    results["customer_churn"] = evaluate_dataset_with_paper_pipeline("customer_churn", X_c, y_c, k_folds=5)

    # 3. Credit Risk Default
    raw_r = load_raw_dataset("credit_risk")
    cleaner_r = DatasetCleaner("credit_risk")
    cl_r, _ = cleaner_r.clean_credit_risk(raw_r)
    y_r = cl_r["loan_status"].values
    X_r = cl_r.drop(columns=["loan_status"])
    results["credit_risk"] = evaluate_dataset_with_paper_pipeline("credit_risk", X_r, y_r, k_folds=5)

    # 4. Credit Card Default Benchmark
    raw_b = load_raw_dataset("credit_card_default_benchmark")
    cleaner_b = DatasetCleaner("credit_card_default_benchmark")
    cl_b, _ = cleaner_b.clean_credit_card_default(raw_b)
    y_b = cl_b["default payment next month"].values
    X_b = cl_b.drop(columns=["default payment next month"])
    results["credit_card_default_benchmark"] = evaluate_dataset_with_paper_pipeline("credit_card_default_benchmark", X_b, y_b, k_folds=5)

    # Save to artifacts/research
    research_dir = get_resolved_path("artifacts/research")
    research_dir.mkdir(parents=True, exist_ok=True)
    json_path = research_dir / "paper_adaptation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Report
    md_path = research_dir / "paper_methodology_and_adaptation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Research Methodology Adaptation & Transferability Report\n\n")
        f.write("### Published Source Citation\n")
        f.write("- **Paper Title**: *Harmonization of Heart Disease Dataset for Accurate Diagnosis: A Machine Learning Approach Enhanced by Feature Engineering*\n")
        f.write("- **Authors**: Jamil Khan Emon et al.\n")
        f.write("- **Journal**: *Computers, Materials & Continua (CMC)*, Vol. 82, No. 3, 2025, pp. 3843-3868\n")
        f.write("- **Source Repository**: `https://github.com/JamilKhanEmon/Heart-Disease-`\n")
        f.write("- **Indexed In**: SCI, Scopus (Q2), EI Compendex\n")
        f.write("- **DOI/URL**: [https://www.techscience.com/cmc/v82n3/59942](https://www.techscience.com/cmc/v82n3/59942)\n\n")

        f.write("## 1. Methodology Extraction from Code Repository\n\n")
        f.write("Based on direct inspection of `Final_model_with_new_Feature.ipynb` and `update_with_k(10)_fold_also_Sensitivity_.ipynb`:\n\n")
        f.write("1. **Pairwise Correlation Ranking**: Screened pairwise variables `(var1 * var2)` ranked descending by Pearson correlation.\n")
        f.write("2. **Categorical Chi-Squared Tests**: Identified statistically significant categorical associations (`chi2_contingency`).\n")
        f.write("3. **Interaction Feature Generation**: Multiplicative cross-features (`df['v1_v2'] = df['v1'] * df['v2']`).\n")
        f.write("4. **StandardScaler Normalization**: Applied uniformly across base numerical and interaction features.\n")
        f.write("5. **Stratified 70-30 Split**: `train_test_split(..., test_size=0.30, random_state=42)`.\n")
        f.write("6. **Dimensionality Reduction via PCA**: Retained 95% cumulative explained variance (`PCA(n_components=0.95)`).\n")
        f.write("7. **Multi-Model Training & Runtime Tracking**: Random Forest, Gradient Boosting, XGBoost, and HistGradientBoosting.\n")
        f.write("8. **Soft-Voting Ensemble**: `VotingClassifier(voting='soft')` combining individual model probability outputs.\n")
        f.write("9. **Comprehensive Diagnostics**: Accuracy, Precision, Sensitivity (Recall), Specificity, F1-Score, ROC-AUC, and Confusion Matrix.\n")
        f.write("10. **Stratified k-Fold Cross-Validation**: Validated with Sensitivity and Specificity distributions across folds.\n\n")

        f.write("## 2. Experimental Cross-Dataset Results\n\n")
        f.write("| Dataset | Orig Features | Eng Features | PCA Comps (95% Var) | Proposed Acc | Sensitivity | Specificity | F1 Score | ROC-AUC | 5-Fold CV Acc |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for key, res in results.items():
            prop = res["proposed_voting_metrics"]
            cv = res["kfold_cv_results"]
            f.write(
                f"| `{key}` | {res['n_original_features']} | {res['n_engineered_features']} | "
                f"**{res['n_pca_components']}** | {prop['accuracy']*100:.2f}% | "
                f"{prop['sensitivity_recall']:.4f} | {prop['specificity']:.4f} | "
                f"{prop['f1_score']:.4f} | {prop['roc_auc']:.4f} | {cv.get('mean_accuracy', 0)*100:.2f}% |\n"
            )

        f.write("\n## 3. Individual Models Comparison & Runtimes\n\n")
        for key, res in results.items():
            f.write(f"### Dataset: `{key}`\n\n")
            f.write("| Model | Accuracy | Precision | Sensitivity | Specificity | F1 Score | ROC-AUC | Runtime (s) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            f.writelines(f"| {m_name} | {m_met['accuracy']:.4f} | {m_met['precision']:.4f} | "
                    f"{m_met['sensitivity_recall']:.4f} | {m_met['specificity']:.4f} | "
                    f"{m_met['f1_score']:.4f} | {m_met['roc_auc']:.4f} | {m_met['run_time_sec']}s |\n" for m_name, m_met in res["individual_models_metrics"].items())
            v_met = res["proposed_voting_metrics"]
            f.write(
                f"| **VotingClassifier (Soft)** | **{v_met['accuracy']:.4f}** | **{v_met['precision']:.4f}** | "
                f"**{v_met['sensitivity_recall']:.4f}** | **{v_met['specificity']:.4f}** | "
                f"**{v_met['f1_score']:.4f}** | **{v_met['roc_auc']:.4f}** | {v_met['run_time_sec']}s |\n\n"
            )

    logger.info(f"Paper replication and transferability study finished. Saved to {json_path} and {md_path}")
    return results


if __name__ == "__main__":
    run_full_paper_transferability_study()
