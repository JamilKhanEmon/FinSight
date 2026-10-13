# Data Quality Audit: Credit Card Default Benchmark

- **Source**: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Dimensions**: 30,000 rows × 25 columns
- **Duplicate Rows**: 0 (0.0%)
- **Missing Cells**: 0

## Target Variable Distribution

**Target Column**: `default payment next month`

| Class | Count | Proportion |
| --- | --- | --- |
| `0` | 23,364 | 77.88% |
| `1` | 6,636 | 22.12% |

## Column-Level Quality Metrics

| Column Name | Type | Missing (Count) | Missing (%) | Unique Values | Details |
| --- | --- | --- | --- | --- | --- |
| `ID` | `int64` | 0 | 0.0% | 30000 | Mean: 15000.50, Range: [1.0, 30000.0] |
| `LIMIT_BAL` | `int64` | 0 | 0.0% | 81 | Mean: 167484.32, Range: [10000.0, 1000000.0] |
| `SEX` | `int64` | 0 | 0.0% | 2 | Mean: 1.60, Range: [1.0, 2.0] |
| `EDUCATION` | `int64` | 0 | 0.0% | 7 | Mean: 1.85, Range: [0.0, 6.0] |
| `MARRIAGE` | `int64` | 0 | 0.0% | 4 | Mean: 1.55, Range: [0.0, 3.0] |
| `AGE` | `int64` | 0 | 0.0% | 56 | Mean: 35.49, Range: [21.0, 79.0] |
| `PAY_0` | `int64` | 0 | 0.0% | 11 | Mean: -0.02, Range: [-2.0, 8.0] |
| `PAY_2` | `int64` | 0 | 0.0% | 11 | Mean: -0.13, Range: [-2.0, 8.0] |
| `PAY_3` | `int64` | 0 | 0.0% | 11 | Mean: -0.17, Range: [-2.0, 8.0] |
| `PAY_4` | `int64` | 0 | 0.0% | 11 | Mean: -0.22, Range: [-2.0, 8.0] |
| `PAY_5` | `int64` | 0 | 0.0% | 10 | Mean: -0.27, Range: [-2.0, 8.0] |
| `PAY_6` | `int64` | 0 | 0.0% | 10 | Mean: -0.29, Range: [-2.0, 8.0] |
| `BILL_AMT1` | `int64` | 0 | 0.0% | 22723 | Mean: 51223.33, Range: [-165580.0, 964511.0] |
| `BILL_AMT2` | `int64` | 0 | 0.0% | 22346 | Mean: 49179.08, Range: [-69777.0, 983931.0] |
| `BILL_AMT3` | `int64` | 0 | 0.0% | 22026 | Mean: 47013.15, Range: [-157264.0, 1664089.0] |
| `BILL_AMT4` | `int64` | 0 | 0.0% | 21548 | Mean: 43262.95, Range: [-170000.0, 891586.0] |
| `BILL_AMT5` | `int64` | 0 | 0.0% | 21010 | Mean: 40311.40, Range: [-81334.0, 927171.0] |
| `BILL_AMT6` | `int64` | 0 | 0.0% | 20604 | Mean: 38871.76, Range: [-339603.0, 961664.0] |
| `PAY_AMT1` | `int64` | 0 | 0.0% | 7943 | Mean: 5663.58, Range: [0.0, 873552.0] |
| `PAY_AMT2` | `int64` | 0 | 0.0% | 7899 | Mean: 5921.16, Range: [0.0, 1684259.0] |
| `PAY_AMT3` | `int64` | 0 | 0.0% | 7518 | Mean: 5225.68, Range: [0.0, 896040.0] |
| `PAY_AMT4` | `int64` | 0 | 0.0% | 6937 | Mean: 4826.08, Range: [0.0, 621000.0] |
| `PAY_AMT5` | `int64` | 0 | 0.0% | 6897 | Mean: 4799.39, Range: [0.0, 426529.0] |
| `PAY_AMT6` | `int64` | 0 | 0.0% | 6939 | Mean: 5215.50, Range: [0.0, 528666.0] |
| `default payment next month` | `int64` | 0 | 0.0% | 2 | Mean: 0.22, Range: [0.0, 1.0] |
