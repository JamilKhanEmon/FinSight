# Credit Risk & Default Prediction Report

## 1. Credit Underwriting Objective

Predicting loan applicant default risk is central to capital adequacy and credit loss provisioning. This module evaluates applicant credit capacity using income ratios, interest burdens, and credit histories.

## 2. Model Performance Benchmarks

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- | --- |
| `dummy` | 0.5000 | 0.2188 | 0.0000 | 0.0000 | 0.0000 |
| `logistic_regression` | 0.8803 | 0.7299 | 0.5546 | 0.7884 | 0.6511 |
| `decision_tree` | 0.9000 | 0.8253 | 0.7231 | 0.7680 | 0.7449 |
| `random_forest` | 0.9206 | 0.8683 | 0.7500 | 0.7743 | 0.7620 |
| `xgboost` | 0.9419 | 0.8949 | 0.7798 | 0.7990 | 0.7893 |
