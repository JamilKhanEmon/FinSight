# FinSight — Business Requirements Document (BRD)

**Project Name**: FinSight — Banking Intelligence & Customer Analytics Platform  
**Document Version**: 1.0.0  
**Target Roles**: Chief Risk Officer (CRO), Chief Marketing Officer (CMO), Head of Fraud Risk, Lead Analytics Engineer, Senior Data Scientist  

---

## 1. Problem Statement
Commercial banks face multi-faceted challenges across marketing efficiency, retention, credit solvency, and fraudulent transaction loss:
1. **Inefficient Campaign Outreach**: Outbound telemarketing suffers from low conversion rates (~11%), ballooning operational call-center expenses and inducing customer contact fatigue.
2. **Customer Attrition**: Churn among retail banking customers depletes low-cost deposit balances and strains customer lifetime value (LTV).
3. **Credit Default Exposure**: Unsecured credit and personal loans require predictive underwriting models to balance loan acceptance against bad debt write-offs.
4. **Transaction Fraud Losses**: Payment networks process hundreds of thousands of transactions daily where fraudulent unauthorized charges comprise <0.2% of events but generate severe financial liability and regulatory penalties.

## 2. Key Stakeholders
- **Executive Leadership (CRO, CMO, CFO)**: Require high-level KPI oversight, portfolio risk health, and return on marketing investment.
- **Credit Risk & Underwriting**: Require probability of default calibration, debt-to-income sensitivity, and regulatory explainability.
- **Fraud Operations**: Require real-time anomaly detection, customizable decision thresholds, and low false-positive operational burdens.
- **Marketing Campaign Managers**: Require prospective lead-scoring models (strictly pre-call) to rank outreach candidates before dialing.

## 3. Scope and Exclusions
- **In Scope**:
  - Six distinct analytical datasets covering campaign conversion, churn, installment credit risk, transaction fraud, credit card default benchmarking, and transactional time-series demand.
  - Leakage-free Scikit-learn pipelines with cross-validation.
  - Interactive multi-page executive BI dashboard.
  - Local Mac M5 CPU-first deployment without cloud dependencies.
- **Out of Scope (Exclusions)**:
  - Fictional unified database claims (datasets originate from distinct institutions and jurisdictions and must not be blended into a false single bank customer base).
  - Unjustified revenue/profitability claims when cost/fee fields are absent from raw data.
  - Post-call variable (`duration`) usage in prospective lead models.

## 4. Functional Requirements
- **FR-1 (Ingestion & Schema Integrity)**: System must validate file formats, column types, delimiter structures, null values, and target distributions before downstream processing.
- **FR-2 (Clean Data Auditing)**: Every record mutation, duplicate removal, or value filtering must be logged with counts.
- **FR-3 (Prospective vs. Retrospective Modeling)**: Primary marketing model must exclude call duration; retrospective benchmark must be distinctly tagged.
- **FR-4 (Multi-Metric Evaluation)**: Evaluation must report ROC-AUC, PR-AUC, F1, Precision, Recall, and Brier Score for all classification models.
- **FR-5 (Threshold Optimization)**: Fraud and credit risk pipelines must evaluate decision thresholds from 0.05 to 0.95.
- **FR-6 (Forecasting & Segmentation)**: Transaction time-series must be evaluated on chronological holdout splits; RFM clustering must profile customer segments.

## 5. Non-Functional Requirements
- **NFR-1 (Local Reproducibility)**: Entire codebase, models, tests, and dashboard must run standalone on Apple Silicon (16 GB RAM) without external GPUs or cloud APIs.
- **NFR-2 (Latency & Throughput)**: Inference API and dashboard views must render in < 1.5 seconds.
- **NFR-3 (Auditability & Testing)**: Automated test suite (Pytest) and Ruff linting must pass 100%.

## 6. Acceptance Criteria
- Full pipeline runs via single command: `python scripts/run_pipeline.py --all`.
- All metrics are calculated from actual local experimental runs and persisted in `artifacts/metrics/`.
- Interactive Streamlit dashboard displays real models and results.
- Zero secrets or unauthorized PII committed.
