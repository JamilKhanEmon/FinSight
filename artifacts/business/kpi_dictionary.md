# FinSight — Banking KPI Dictionary

This dictionary defines core business metrics implemented across the FinSight platform. Every formula documents its required columns, mathematical definition, aggregation grain, and analytical limitations.

---

### 1. Marketing Campaign Response Rate
- **Business Definition**: Proportion of targeted clients who agreed to subscribe to a long-term deposit.
- **Formula**:
  $$\text{Response Rate} = \frac{\sum [y = \text{'yes'}]}{N_{\text{contacted}}} \times 100$$
- **Required Columns**: `y` (target variable).
- **Aggregation Level**: Campaign level, monthly cohort, or customer segment.
- **Interpretation**: Measures lead quality and marketing message resonance. Baseline is 11.27%.
- **Limitations**: Does not quantify deposit balance size or lifetime net interest margin (NIM) since deposit balance is not recorded in the marketing dataset.

---

### 2. Customer Churn Rate
- **Business Definition**: Proportion of retail account holders who discontinued their banking relationship within the observation window.
- **Formula**:
  $$\text{Churn Rate} = \frac{\sum [\text{Exited} = 1]}{N_{\text{total\_customers}}} \times 100$$
- **Required Columns**: `Exited`.
- **Aggregation Level**: Portfolio level, segmented by country (`Geography`), age cohort, and product holding.
- **Interpretation**: Baseline churn is ~20.37%. High churn indicates competitive pressure, pricing dissatisfaction, or poor engagement.
- **Limitations**: Cross-sectional snapshot; does not indicate exact calendar time-to-churn due to lack of longitudinal date timestamps.

---

### 3. Credit Default Rate
- **Business Definition**: Percentage of loan applicants who defaulted on their contractual payment obligations.
- **Formula**:
  $$\text{Default Rate} = \frac{\sum [\text{loan\_status} = 1]}{N_{\text{loans\_issued}}} \times 100$$
- **Required Columns**: `loan_status`.
- **Aggregation Level**: Loan grade (`loan_grade`), intent (`loan_intent`), and income bracket.
- **Interpretation**: Portfolio baseline default rate is 21.82%. Used for setting loan provisioning reserves under IFRS 9 / CECL frameworks.
- **Limitations**: Does not distinguish between 30/60/90 days past due (DPD) vs. total charge-off.

---

### 4. Fraud Detection Rate (Recall / Sensitivity)
- **Business Definition**: Percentage of actual fraudulent transactions successfully flagged by the scoring system.
- **Formula**:
  $$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$
- **Required Columns**: `Class` (target), predicted label $\hat{y}$.
- **Aggregation Level**: Real-time per-transaction score.
- **Interpretation**: High recall minimizes direct fraudulent monetary losses. At threshold 0.50, XGBoost captures >78% of frauds.
- **Limitations**: High recall must be balanced against False Positive Rate to prevent merchant checkout friction and customer support ticket spikes.

---

### 5. False Positive Rate (FPR)
- **Business Definition**: Percentage of legitimate transactions incorrectly flagged or blocked as suspected fraud.
- **Formula**:
  $$\text{FPR} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}$$
- **Required Columns**: `Class`, predicted label $\hat{y}$.
- **Interpretation**: Measures customer friction. In retail banking, an FPR exceeding 0.1% results in tens of thousands of rejected legitimate customer cards.
- **Limitations**: Requires cost-sensitive utility curve analysis to balance loss prevention against customer insult rate.

---

### 6. Revolving Credit Utilization Ratio
- **Business Definition**: Proportion of assigned credit card limit currently utilized by the account holder.
- **Formula**:
  $$\text{Utilization Ratio} = \frac{\text{BILL\_AMT1}}{\text{LIMIT\_BAL}}$$
- **Required Columns**: `BILL_AMT1`, `LIMIT_BAL`.
- **Interpretation**: Utilization above 80% is a primary early-warning indicator of prospective default.
- **Limitations**: Excludes intra-month peak spend if billing cycle snapshot does not capture intermediate payoffs.

---

### 7. Average Daily Transaction Revenue
- **Business Definition**: Mean monetary gross sales generated per active operating day.
- **Formula**:
  $$\text{Mean Daily Revenue} = \frac{\sum_{i=1}^{T} \text{Daily Revenue}}{T}$$
- **Required Columns**: `Quantity`, `UnitPrice`, `InvoiceDate`.
- **Interpretation**: Benchmark metric for time-series forecasting. Used for working capital management and cash flow planning.
