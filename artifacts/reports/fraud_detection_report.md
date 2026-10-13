# Transaction Fraud Detection Analytics Report

## 1. Executive Fraud Monitoring Overview

Transaction fraud represents severe financial loss and reputational risk. Given extreme class imbalance (~0.17%), models must be evaluated on PR-AUC, precision, and recall rather than raw accuracy.

## 2. Model Performance Benchmarks

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- | --- |
| `dummy` | 0.0017 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| `logistic_regression` | 0.6860 | 0.9480 | 0.0484 | 0.8526 | 0.0915 |
| `xgboost` | 0.7548 | 0.9777 | 0.8375 | 0.7053 | 0.7657 |
