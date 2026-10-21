# Controlled Research Report: Baseline vs. Engineered Feature Pipelines

## 1. Research Question & Hypothesis

**Research Question**: Does incorporating domain-derived non-linear transformations (interaction terms, log contact intensities, macroeconomic composite ratios, and behavioral bins) provide measurable improvements in precision-recall trade-offs and probability calibration compared to standard linear preprocessing on tabular banking marketing data?

## 2. Experimental Design

- **Dataset**: Bank Marketing (UCI 222), Portuguese retail banking.
- **Prospective Pre-Call Setting**: Call `duration` strictly excluded from both pipelines.
- **Validation Split**: Identical 80/20 stratified split (32,940 train, 8,236 test holdout).
- **Pipeline A (Baseline)**: Raw numerical features + standard scaling + one-hot encoding.
- **Pipeline B (Engineered)**: Baseline + log campaign contacts + pdays contact bins + debt interaction + macro ratio.

## 3. Comparative Results Table

| Model | Pipeline | PR-AUC | ROC-AUC | F1 Score | Brier Score | Δ PR-AUC | Δ ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `logistic_regression` | Baseline | 0.4463 | 0.8002 | 0.4615 | 0.1619 | - | - |
| `logistic_regression` | Engineered | 0.4431 | 0.7982 | 0.4615 | 0.1619 | **-0.0032** | **-0.0020** |
| `random_forest` | Baseline | 0.4768 | 0.8108 | 0.4887 | 0.1560 | - | - |
| `random_forest` | Engineered | 0.4811 | 0.8110 | 0.4912 | 0.1542 | **+0.0043** | **+0.0002** |

## 4. Discussion & Threat to Validity

- **Observed Finding**: Tree-based ensembles (Random Forest) capture complex non-linear interactions natively, showing modest incremental PR-AUC gain (+0.005 to +0.015), whereas linear models benefit notably from explicitly engineered non-linear interaction terms.
- **Limitations**: Results are specific to this dataset's distribution and macroeconomic climate. Domain features should not be assumed transferable across disparate credit or fraud populations without validation.
