"""Cost-Sensitive Soft-Voting Ensemble Engine with Asymmetric Threshold Calibration.

Adapts the multi-model soft voting philosophy of Jamil Khan Emon et al. (CMC 2025)
for banking scenarios characterized by high class imbalance and asymmetric financial costs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from finsight.utils.logger import setup_logger

logger = setup_logger("finsight.ensemble_engine")


def compute_banking_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    model_name: str,
    run_time: float = 0.0,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Computes full diagnostic evaluation metrics including sensitivity and specificity."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, len(y_true)

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc = float(roc_auc_score(y_true, y_proba)) if y_proba is not None else 0.0
    pr_auc = float(average_precision_score(y_true, y_proba)) if y_proba is not None else 0.0

    return {
        "model_name": model_name,
        "decision_threshold": round(float(threshold), 3),
        "run_time_sec": round(float(run_time), 3),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "sensitivity_recall": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def calibrate_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = 1.0,
    benefit_tp: float = 5.0,
) -> dict[str, Any]:
    """Finds the optimal classification threshold balancing F1 score and financial payoff."""
    thresholds = np.linspace(0.05, 0.95, 37)
    best_f1, best_f1_th = -1.0, 0.50
    best_profit, best_profit_th = -float("inf"), 0.50
    curve = []

    for th in thresholds:
        yp = (y_proba >= th).astype(int)
        cm = confusion_matrix(y_true, yp)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, len(y_true))

        f1 = float(f1_score(y_true, yp, zero_division=0))
        prec = float(precision_score(y_true, yp, zero_division=0))
        rec = float(recall_score(y_true, yp, zero_division=0))
        # Expected Monetary Value: TP * benefit - FP * cost
        profit = float(tp * benefit_tp - fp * cost_fp)

        curve.append({
            "threshold": round(float(th), 3),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "net_payoff": round(profit, 2),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_f1_th = th
        if profit > best_profit:
            best_profit = profit
            best_profit_th = th

    return {
        "optimal_f1_threshold": round(float(best_f1_th), 3),
        "max_f1_score": round(float(best_f1), 4),
        "optimal_profit_threshold": round(float(best_profit_th), 3),
        "max_net_payoff": round(float(best_profit), 2),
        "threshold_curve": curve,
    }


def build_cost_sensitive_ensemble(y_train: np.ndarray) -> tuple[VotingClassifier, dict[str, Any]]:
    """Constructs heterogeneous base models with class-imbalance weights, bundled into soft-voting."""
    n_pos = int(np.sum(y_train))
    n_neg = len(y_train) - n_pos
    pos_weight = max(1.0, float(n_neg) / float(max(1, n_pos)))

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=4,
    )
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=4,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=100,
        class_weight="balanced",
        random_state=42,
    )
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    models = {
        "RandomForest": rf,
        "XGBoost": xgb,
        "HistGradientBoosting": hgb,
        "LogisticRegression": lr,
    }

    voting = VotingClassifier(
        estimators=[(name, m) for name, m in models.items()],
        voting="soft",
    )
    return voting, models


def run_cross_validated_ensemble_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int = 5,
) -> dict[str, Any]:
    """Runs rigorous Stratified K-Fold cross-validation reporting mean and standard deviation."""
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    accs, precs, recs, specs, f1s, rocs, prs = [], [], [], [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        voting, _ = build_cost_sensitive_ensemble(y_tr)
        voting.fit(X_tr, y_tr)

        yp = voting.predict(X_val)
        yprob = voting.predict_proba(X_val)[:, 1]
        cm = confusion_matrix(y_val, yp)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, len(y_val))

        accs.append(accuracy_score(y_val, yp))
        precs.append(precision_score(y_val, yp, zero_division=0))
        recs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        f1s.append(f1_score(y_val, yp, zero_division=0))
        rocs.append(roc_auc_score(y_val, yprob))
        prs.append(average_precision_score(y_val, yprob))

    return {
        "k_folds": k_folds,
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "std_accuracy": round(float(np.std(accs)), 4),
        "mean_precision": round(float(np.mean(precs)), 4),
        "mean_sensitivity": round(float(np.mean(recs)), 4),
        "std_sensitivity": round(float(np.std(recs)), 4),
        "mean_specificity": round(float(np.mean(specs)), 4),
        "std_specificity": round(float(np.std(specs)), 4),
        "mean_f1_score": round(float(np.mean(f1s)), 4),
        "std_f1_score": round(float(np.std(f1s)), 4),
        "mean_roc_auc": round(float(np.mean(rocs)), 4),
        "mean_pr_auc": round(float(np.mean(prs)), 4),
    }
