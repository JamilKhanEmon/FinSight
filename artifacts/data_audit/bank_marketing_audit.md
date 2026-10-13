# Data Quality Audit: Bank Marketing

- **Source**: https://archive.ics.uci.edu/dataset/222/bank+marketing
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Dimensions**: 41,188 rows × 21 columns
- **Duplicate Rows**: 12 (0.03%)
- **Missing Cells**: 0

## Target Variable Distribution

**Target Column**: `y`

| Class | Count | Proportion |
| --- | --- | --- |
| `no` | 36,548 | 88.73% |
| `yes` | 4,640 | 11.27% |

## Column-Level Quality Metrics

| Column Name | Type | Missing (Count) | Missing (%) | Unique Values | Details |
| --- | --- | --- | --- | --- | --- |
| `age` | `int64` | 0 | 0.0% | 78 | Mean: 40.02, Range: [17.0, 98.0] |
| `job` | `str` | 0 | 0.0% | 12 | Top: admin., blue-collar, technician |
| `marital` | `str` | 0 | 0.0% | 4 | Top: married, single, divorced |
| `education` | `str` | 0 | 0.0% | 8 | Top: university.degree, high.school, basic.9y |
| `default` | `str` | 0 | 0.0% | 3 | Top: no, unknown, yes |
| `housing` | `str` | 0 | 0.0% | 3 | Top: yes, no, unknown |
| `loan` | `str` | 0 | 0.0% | 3 | Top: no, yes, unknown |
| `contact` | `str` | 0 | 0.0% | 2 | Top: cellular, telephone |
| `month` | `str` | 0 | 0.0% | 10 | Top: may, jul, aug |
| `day_of_week` | `str` | 0 | 0.0% | 5 | Top: thu, mon, wed |
| `duration` | `int64` | 0 | 0.0% | 1544 | Mean: 258.29, Range: [0.0, 4918.0] |
| `campaign` | `int64` | 0 | 0.0% | 42 | Mean: 2.57, Range: [1.0, 56.0] |
| `pdays` | `int64` | 0 | 0.0% | 27 | Mean: 962.48, Range: [0.0, 999.0] |
| `previous` | `int64` | 0 | 0.0% | 8 | Mean: 0.17, Range: [0.0, 7.0] |
| `poutcome` | `str` | 0 | 0.0% | 3 | Top: nonexistent, failure, success |
| `emp.var.rate` | `float64` | 0 | 0.0% | 10 | Mean: 0.08, Range: [-3.4, 1.4] |
| `cons.price.idx` | `float64` | 0 | 0.0% | 26 | Mean: 93.58, Range: [92.201, 94.767] |
| `cons.conf.idx` | `float64` | 0 | 0.0% | 26 | Mean: -40.50, Range: [-50.8, -26.9] |
| `euribor3m` | `float64` | 0 | 0.0% | 316 | Mean: 3.62, Range: [0.634, 5.045] |
| `nr.employed` | `float64` | 0 | 0.0% | 11 | Mean: 5167.04, Range: [4963.6, 5228.1] |
| `y` | `str` | 0 | 0.0% | 2 | Top: no, yes |
