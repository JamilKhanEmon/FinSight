# Research Methodology Extraction & Transferability Framework

## 1. Overview and Status
- **Author/Investigator**: Senior Data Scientist & Analytics Lead
- **Status**: Research Placeholder & General Benchmark Active
- **Notice**: If external PDFs, DOIs, or source repositories (such as the author's previous heart-disease data harmonization and predictive modeling research) are placed into this folder, their exact methods, data harmonization protocols, and feature engineering approaches will be extracted into this document and evaluated for domain transferability.

## 2. Transferability Analysis Framework

When adapting methodologies from clinical / biomedical research to tabular banking intelligence:

| Clinical / Biomedical Methodology | Banking Domain Analogue | Transferability Status | Rationale & Caution |
| --- | --- | --- | --- |
| Patient Risk Harmonization | Multi-Account Credit Scoring | **Transferable** | Harmonizing demographic and balance variables across multiple account types follows similar missingness and outlier handling principles. |
| Non-linear Biomarker Interactions | Financial Leverage & Solvency Ratios | **Transferable** | Computing ratios (e.g. Loan-to-Income, Payment-to-Bill) mirrors physiological ratio features (e.g. Cholesterol/HDL). |
| Prospective vs Retrospective Cohorts | Pre-call vs Post-call Campaign Analytics | **Transferable (Critical)** | Clinical models cannot use post-operative variables; similarly, prospective bank marketing models strictly exclude post-call `duration`. |
| Clinical Diagnostic Metrics | Cost-Weighted Decision Thresholds | **Transferable** | In fraud detection and credit risk, false positives carry different financial penalties than false negatives, requiring PR-AUC and threshold optimization over raw accuracy. |
| Physiological assumptions (e.g. heart rate bounds) | Banking domain features | **NOT Transferable** | Clinical domain assumptions must never be imported into credit or fraud modeling. |

## 3. Experimental Verification
The accompanying script `src/finsight/experiments/research_experiment.py` performs empirical baseline-versus-engineered feature comparisons on banking datasets with identical, leakage-free cross-validation folds.
