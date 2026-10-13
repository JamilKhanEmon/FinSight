"""Unit tests for binary evaluation metrics and calibration."""

import numpy as np

from finsight.evaluation.evaluator import evaluate_binary_classifier


def test_evaluate_binary_classifier_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.05, 0.15, 0.85, 0.95])

    metrics = evaluate_binary_classifier(y_true, y_proba, "perfect_model")
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["brier_score"] < 0.05
    assert len(metrics["threshold_analysis"]) > 0


def test_evaluate_binary_classifier_imbalanced():
    y_true = np.array([0, 0, 0, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.9])

    metrics = evaluate_binary_classifier(y_true, y_proba, "imbalanced_model")
    assert metrics["roc_auc"] == 1.0
    assert metrics["positive_rate"] == 0.2
