# Data Quality Audit: Credit Risk

- **Source**: https://www.kaggle.com/datasets/laotse/credit-risk-dataset
- **License**: CC0: Public Domain
- **Dimensions**: 32,581 rows × 12 columns
- **Duplicate Rows**: 165 (0.51%)
- **Missing Cells**: 4,011

## Target Variable Distribution

**Target Column**: `loan_status`

| Class | Count | Proportion |
| --- | --- | --- |
| `0` | 25,473 | 78.18% |
| `1` | 7,108 | 21.82% |

## Column-Level Quality Metrics

| Column Name | Type | Missing (Count) | Missing (%) | Unique Values | Details |
| --- | --- | --- | --- | --- | --- |
| `person_age` | `int64` | 0 | 0.0% | 58 | Mean: 27.73, Range: [20.0, 144.0] |
| `person_income` | `int64` | 0 | 0.0% | 4295 | Mean: 66074.85, Range: [4000.0, 6000000.0] |
| `person_home_ownership` | `str` | 0 | 0.0% | 4 | Top: RENT, MORTGAGE, OWN |
| `person_emp_length` | `float64` | 895 | 2.75% | 36 | Mean: 4.79, Range: [0.0, 123.0] |
| `loan_intent` | `str` | 0 | 0.0% | 6 | Top: EDUCATION, MEDICAL, VENTURE |
| `loan_grade` | `str` | 0 | 0.0% | 7 | Top: A, B, C |
| `loan_amnt` | `int64` | 0 | 0.0% | 753 | Mean: 9589.37, Range: [500.0, 35000.0] |
| `loan_int_rate` | `float64` | 3116 | 9.56% | 348 | Mean: 11.01, Range: [5.42, 23.22] |
| `loan_status` | `int64` | 0 | 0.0% | 2 | Mean: 0.22, Range: [0.0, 1.0] |
| `loan_percent_income` | `float64` | 0 | 0.0% | 77 | Mean: 0.17, Range: [0.0, 0.83] |
| `cb_person_default_on_file` | `str` | 0 | 0.0% | 2 | Top: N, Y |
| `cb_person_cred_hist_length` | `int64` | 0 | 0.0% | 29 | Mean: 5.80, Range: [2.0, 30.0] |
