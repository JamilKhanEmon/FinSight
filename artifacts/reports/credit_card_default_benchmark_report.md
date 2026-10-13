# Credit Card Default Benchmark Report (UCI 350)

## 1. Study Population & Benchmark Scope

This experiment benchmarks credit default risk on revolving credit accounts in Taiwan (30,000 clients). Unlike installment loan credit risk (Dataset C), this dataset evaluates monthly payment delays and revolving balance utilization.

## 2. Model Performance Benchmarks

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- | --- |
| `dummy` | 0.5000 | 0.2213 | 0.0000 | 0.0000 | 0.0000 |
| `logistic_regression` | 0.7369 | 0.5105 | 0.4292 | 0.6222 | 0.5080 |
| `random_forest` | 0.7736 | 0.5551 | 0.4593 | 0.6290 | 0.5309 |
| `xgboost` | 0.7768 | 0.5576 | 0.4601 | 0.6214 | 0.5287 |
