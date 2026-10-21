# Research Methodology Adaptation & Transferability Report

### Published Source Citation
- **Paper Title**: *Harmonization of Heart Disease Dataset for Accurate Diagnosis: A Machine Learning Approach Enhanced by Feature Engineering*
- **Authors**: Jamil Khan Emon et al.
- **Journal**: *Computers, Materials & Continua (CMC)*, Vol. 82, No. 3, 2025, pp. 3843-3868
- **Source Repository**: `https://github.com/JamilKhanEmon/Heart-Disease-`
- **Indexed In**: SCI, Scopus (Q2), EI Compendex
- **DOI/URL**: [https://www.techscience.com/cmc/v82n3/59942](https://www.techscience.com/cmc/v82n3/59942)

## 1. Methodology Extraction from Code Repository

Based on direct inspection of `Final_model_with_new_Feature.ipynb` and `update_with_k(10)_fold_also_Sensitivity_.ipynb`:

1. **Pairwise Correlation Ranking**: Screened pairwise variables `(var1 * var2)` ranked descending by Pearson correlation.
2. **Categorical Chi-Squared Tests**: Identified statistically significant categorical associations (`chi2_contingency`).
3. **Interaction Feature Generation**: Multiplicative cross-features (`df['v1_v2'] = df['v1'] * df['v2']`).
4. **StandardScaler Normalization**: Applied uniformly across base numerical and interaction features.
5. **Stratified 70-30 Split**: `train_test_split(..., test_size=0.30, random_state=42)`.
6. **Dimensionality Reduction via PCA**: Retained 95% cumulative explained variance (`PCA(n_components=0.95)`).
7. **Multi-Model Training & Runtime Tracking**: Random Forest, Gradient Boosting, XGBoost, and HistGradientBoosting.
8. **Soft-Voting Ensemble**: `VotingClassifier(voting='soft')` combining individual model probability outputs.
9. **Comprehensive Diagnostics**: Accuracy, Precision, Sensitivity (Recall), Specificity, F1-Score, ROC-AUC, and Confusion Matrix.
10. **Stratified k-Fold Cross-Validation**: Validated with Sensitivity and Specificity distributions across folds.

## 2. Experimental Cross-Dataset Results

| Dataset | Orig Features | Eng Features | PCA Comps (95% Var) | Proposed Acc | Sensitivity | Specificity | F1 Score | ROC-AUC | 5-Fold CV Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bank_marketing` | 9 | 19 | **7** | 89.98% | 0.2162 | 0.9866 | 0.3272 | 0.8008 | 89.98% |
| `customer_churn` | 8 | 18 | **8** | 85.77% | 0.4157 | 0.9707 | 0.5433 | 0.8490 | 84.72% |
| `credit_risk` | 7 | 17 | **6** | 85.05% | 0.5320 | 0.9396 | 0.6087 | 0.8557 | 84.66% |
| `credit_card_default_benchmark` | 23 | 33 | **16** | 81.51% | 0.3605 | 0.9443 | 0.4632 | 0.7650 | 81.14% |

## 3. Individual Models Comparison & Runtimes

### Dataset: `bank_marketing`

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 Score | ROC-AUC | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RandomForest | 0.8991 | 0.6478 | 0.2299 | 0.9841 | 0.3393 | 0.7976 | 0.5s |
| GradientBoosting | 0.8991 | 0.6755 | 0.2004 | 0.9878 | 0.3091 | 0.8010 | 2.611s |
| XGBoost | 0.8993 | 0.6555 | 0.2241 | 0.9850 | 0.3340 | 0.7956 | 0.065s |
| HistGradientBoosting | 0.9002 | 0.6755 | 0.2198 | 0.9866 | 0.3317 | 0.8026 | 0.193s |
| **VotingClassifier (Soft)** | **0.8998** | **0.6719** | **0.2162** | **0.9866** | **0.3272** | **0.8008** | 3.331s |

### Dataset: `customer_churn`

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 Score | ROC-AUC | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RandomForest | 0.8507 | 0.7985 | 0.3568 | 0.9770 | 0.4932 | 0.8426 | 0.15s |
| GradientBoosting | 0.8557 | 0.7947 | 0.3928 | 0.9740 | 0.5257 | 0.8408 | 0.813s |
| XGBoost | 0.8510 | 0.7253 | 0.4321 | 0.9581 | 0.5415 | 0.8464 | 0.05s |
| HistGradientBoosting | 0.8510 | 0.7228 | 0.4354 | 0.9573 | 0.5434 | 0.8418 | 0.261s |
| **VotingClassifier (Soft)** | **0.8577** | **0.7840** | **0.4157** | **0.9707** | **0.5433** | **0.8490** | 1.267s |

### Dataset: `credit_risk`

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 Score | ROC-AUC | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RandomForest | 0.8514 | 0.7224 | 0.5202 | 0.9441 | 0.6049 | 0.8539 | 0.44s |
| GradientBoosting | 0.8473 | 0.7064 | 0.5160 | 0.9400 | 0.5964 | 0.8517 | 2.253s |
| XGBoost | 0.8490 | 0.7006 | 0.5405 | 0.9354 | 0.6102 | 0.8548 | 0.06s |
| HistGradientBoosting | 0.8506 | 0.7073 | 0.5400 | 0.9375 | 0.6124 | 0.8532 | 0.146s |
| **VotingClassifier (Soft)** | **0.8505** | **0.7113** | **0.5320** | **0.9396** | **0.6087** | **0.8557** | 2.855s |

### Dataset: `credit_card_default_benchmark`

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 Score | ROC-AUC | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RandomForest | 0.8126 | 0.6402 | 0.3489 | 0.9443 | 0.4517 | 0.7629 | 0.714s |
| GradientBoosting | 0.8139 | 0.6381 | 0.3670 | 0.9409 | 0.4660 | 0.7613 | 5.266s |
| XGBoost | 0.8123 | 0.6332 | 0.3610 | 0.9406 | 0.4598 | 0.7609 | 0.08s |
| HistGradientBoosting | 0.8164 | 0.6542 | 0.3605 | 0.9459 | 0.4648 | 0.7629 | 0.174s |
| **VotingClassifier (Soft)** | **0.8151** | **0.6477** | **0.3605** | **0.9443** | **0.4632** | **0.7650** | 6.296s |

