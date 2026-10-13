# Customer Churn & Retention Analytics Report

## 1. Problem Statement & Context

Customer defection reduces banking lifetime revenue and increases acquisition cost pressure. This module models customer churn probability using demographic and banking account attributes.

## 2. Model Evaluation Summary

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- | --- |
| `dummy` | 0.5000 | 0.2035 | 0.0000 | 0.0000 | 0.0000 |
| `logistic_regression` | 0.7882 | 0.5019 | 0.3961 | 0.7445 | 0.5171 |
| `random_forest` | 0.8618 | 0.6940 | 0.5137 | 0.7396 | 0.6062 |
| `xgboost` | 0.8669 | 0.7160 | 0.5008 | 0.7543 | 0.6020 |
