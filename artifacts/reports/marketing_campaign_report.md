# Bank Marketing Campaign Intelligence & Modeling Report

## 1. Executive Summary & Objective

The primary business goal is prospective identification of banking clients most likely to subscribe to term deposits during direct outbound telephone campaigns. Prospective models strictly exclude call duration, as duration cannot be known before a call takes place.

## 2. Dataset and Cleaning Summary

- Total records: 41,188
- Duplicate records removed: 12
- Cleaned records analyzed: 41,176
- Baseline subscription rate: 11.27%

## 3. Prospective Model Comparison (Leakage-Free, Pre-Call Features Only)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | Brier Score |
| --- | --- | --- | --- | --- | --- | --- |
| `dummy` | 0.5000 | 0.1127 | 0.0000 | 0.0000 | 0.0000 | 0.1000 |
| `logistic_regression` | 0.7982 | 0.4431 | 0.3589 | 0.6466 | 0.4615 | 0.1619 |
| `random_forest` | 0.8114 | 0.4777 | 0.3929 | 0.6444 | 0.4882 | 0.1547 |
| `xgboost` | 0.8128 | 0.4805 | 0.4034 | 0.6369 | 0.4939 | 0.1397 |

## 4. Retrospective Benchmark (With Post-Call Duration)

> [!WARNING]
> Call duration is a post-call operational attribute and MUST NOT be used for prospective lead ranking. This retrospective benchmark is presented strictly for diagnostic comparison.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- | --- |
| `logistic_regression (Retrospective)` | 0.9404 | 0.5921 | 0.4488 | 0.9019 | 0.5994 |
| `xgboost (Retrospective)` | 0.9510 | 0.6697 | 0.4650 | 0.9310 | 0.6202 |

## 5. Strategic Business Recommendations

- **Targeting Strategy**: Prioritize clients with prior campaign success (`poutcome == 'success'`) and individuals within favorable macroeconomic periods (lower Euribor 3M rate environments).
- **Contact Frequency Cap**: Diminishing returns occur sharply after 4 contacts; campaign managers should cap outbound attempts per client at 4 to reduce customer fatigue and call center labor expenses.
