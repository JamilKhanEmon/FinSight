# 🏛️ FinSight Statistical Banking Intelligence Report

### Methodology Provenance & Academic Inspiration
- **Methodology Roots**: Statistical feature screening, PCA dimensionality compression, and multi-model soft voting ensemble.
- **Academic Reference**: Jamil Khan Emon et al., *Computers, Materials & Continua (CMC)*, Vol. 82, No. 3, 2025, pp. 3843–3868.
- **Banking Adaptation**: Domain-guided financial ratios, log1p magnitude stabilization, class-imbalance cost-sensitive weighting, and optimal decision threshold calibration.

## 1. Executive Cross-Dataset Performance Summary

| Dataset | Orig Features | Eng Features | PCA (95% Var) | Baseline Acc | Proposed Acc | Sensitivity (Default/Churn) | Specificity | F1 Score | ROC-AUC | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bank_marketing` | 9 | 19 | **6** | 89.96% | **87.08%** | **59.63%** | 90.57% | **0.5098** | **0.7984** | `0.6` |
| `customer_churn` | 8 | 16 | **9** | 86.27% | **84.93%** | **54.01%** | 92.84% | **0.5935** | **0.8515** | `0.625` |
| `credit_risk` | 7 | 19 | **6** | 86.95% | **83.55%** | **69.99%** | 87.35% | **0.6505** | **0.8578** | `0.575` |
| `credit_card_default_benchmark` | 23 | 39 | **24** | 82.06% | **79.34%** | **53.54%** | 86.67% | **0.5342** | **0.7694** | `0.575` |

## 2. Multi-Model Breakdown & Execution Runtimes

### Dataset: `bank_marketing`

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 Score | ROC-AUC | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RandomForest | 0.8481 | 0.3918 | 0.6293 | 0.8759 | 0.4829 | 0.7983 | 0.415s |
| XGBoost | 0.8429 | 0.3806 | 0.6286 | 0.8701 | 0.4741 | 0.7950 | 0.061s |
| HistGradientBoosting | 0.8421 | 0.3804 | 0.6386 | 0.8679 | 0.4768 | 0.8008 | 0.417s |
| LogisticRegression | 0.7248 | 0.2490 | 0.7155 | 0.7259 | 0.3694 | 0.7584 | 0.013s |
| **⭐ Calibrated Soft-Voting Ensemble** | **0.8708** | **0.4453** | **0.5963** | **0.9057** | **0.5098** | **0.7984** | 0.894s |

### Dataset: `customer_churn`

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 Score | ROC-AUC | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RandomForest | 0.8060 | 0.5184 | 0.6694 | 0.8409 | 0.5843 | 0.8442 | 0.179s |
| XGBoost | 0.7953 | 0.4983 | 0.7185 | 0.8150 | 0.5885 | 0.8503 | 0.052s |
| HistGradientBoosting | 0.7970 | 0.5012 | 0.6645 | 0.8309 | 0.5714 | 0.8378 | 0.547s |
| LogisticRegression | 0.6970 | 0.3748 | 0.7300 | 0.6886 | 0.4953 | 0.7826 | 0.003s |
| **⭐ Calibrated Soft-Voting Ensemble** | **0.8493** | **0.6587** | **0.5401** | **0.9284** | **0.5935** | **0.8515** | 0.787s |

### Dataset: `credit_risk`

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 Score | ROC-AUC | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RandomForest | 0.8287 | 0.5863 | 0.7347 | 0.8549 | 0.6522 | 0.8600 | 0.414s |
| XGBoost | 0.8097 | 0.5476 | 0.7465 | 0.8274 | 0.6318 | 0.8588 | 0.059s |
| HistGradientBoosting | 0.8104 | 0.5485 | 0.7526 | 0.8266 | 0.6345 | 0.8585 | 0.437s |
| LogisticRegression | 0.7591 | 0.4664 | 0.7046 | 0.7744 | 0.5613 | 0.8178 | 0.007s |
| **⭐ Calibrated Soft-Voting Ensemble** | **0.8355** | **0.6076** | **0.6999** | **0.8735** | **0.6505** | **0.8578** | 0.921s |

### Dataset: `credit_card_default_benchmark`

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 Score | ROC-AUC | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RandomForest | 0.7770 | 0.4965 | 0.5777 | 0.8336 | 0.5340 | 0.7675 | 0.699s |
| XGBoost | 0.7570 | 0.4620 | 0.5998 | 0.8016 | 0.5220 | 0.7676 | 0.105s |
| HistGradientBoosting | 0.7570 | 0.4632 | 0.6204 | 0.7957 | 0.5304 | 0.7651 | 1.274s |
| LogisticRegression | 0.7318 | 0.4269 | 0.6199 | 0.7636 | 0.5056 | 0.7440 | 0.007s |
| **⭐ Calibrated Soft-Voting Ensemble** | **0.7934** | **0.5330** | **0.5354** | **0.8667** | **0.5342** | **0.7694** | 2.068s |


## 3. 5-Fold Cross-Validation Stability Analysis

| Dataset | 5-Fold Mean Accuracy (± Std) | Mean Sensitivity (± Std) | Mean Specificity (± Std) | Mean F1 Score | Mean ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `bank_marketing` | 83.67% (±0.43%) | 61.84% (±2.82%) | 86.44% (±0.41%) | 0.4603 | 0.7857 |
| `customer_churn` | 79.00% (±0.88%) | 66.41% (±1.02%) | 82.22% (±1.04%) | 0.5632 | 0.8254 |
| `credit_risk` | 81.27% (±0.51%) | 74.06% (±0.97%) | 83.29% (±0.41%) | 0.6337 | 0.8596 |
| `credit_card_default_benchmark` | 76.60% (±0.48%) | 59.94% (±1.23%) | 81.33% (±0.37%) | 0.5313 | 0.7731 |
