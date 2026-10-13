# Data Quality Audit: Customer Churn

- **Source**: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
- **License**: Open Data Commons Open Database License (ODbL) / Public
- **Dimensions**: 10,000 rows × 14 columns
- **Duplicate Rows**: 0 (0.0%)
- **Missing Cells**: 0

## Target Variable Distribution

**Target Column**: `Exited`

| Class | Count | Proportion |
| --- | --- | --- |
| `0` | 7,963 | 79.63% |
| `1` | 2,037 | 20.37% |

## Column-Level Quality Metrics

| Column Name | Type | Missing (Count) | Missing (%) | Unique Values | Details |
| --- | --- | --- | --- | --- | --- |
| `RowNumber` | `int64` | 0 | 0.0% | 10000 | Mean: 5000.50, Range: [1.0, 10000.0] |
| `CustomerId` | `int64` | 0 | 0.0% | 10000 | Mean: 15690940.57, Range: [15565701.0, 15815690.0] |
| `Surname` | `str` | 0 | 0.0% | 2932 | Top: Smith, Scott, Martin |
| `CreditScore` | `int64` | 0 | 0.0% | 460 | Mean: 650.53, Range: [350.0, 850.0] |
| `Geography` | `str` | 0 | 0.0% | 3 | Top: France, Germany, Spain |
| `Gender` | `str` | 0 | 0.0% | 2 | Top: Male, Female |
| `Age` | `int64` | 0 | 0.0% | 70 | Mean: 38.92, Range: [18.0, 92.0] |
| `Tenure` | `int64` | 0 | 0.0% | 11 | Mean: 5.01, Range: [0.0, 10.0] |
| `Balance` | `float64` | 0 | 0.0% | 6382 | Mean: 76485.89, Range: [0.0, 250898.09] |
| `NumOfProducts` | `int64` | 0 | 0.0% | 4 | Mean: 1.53, Range: [1.0, 4.0] |
| `HasCrCard` | `int64` | 0 | 0.0% | 2 | Mean: 0.71, Range: [0.0, 1.0] |
| `IsActiveMember` | `int64` | 0 | 0.0% | 2 | Mean: 0.52, Range: [0.0, 1.0] |
| `EstimatedSalary` | `float64` | 0 | 0.0% | 9999 | Mean: 100090.24, Range: [11.58, 199992.48] |
| `Exited` | `int64` | 0 | 0.0% | 2 | Mean: 0.20, Range: [0.0, 1.0] |
