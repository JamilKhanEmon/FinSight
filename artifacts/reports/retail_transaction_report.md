# Online Retail Transaction Analytics & Forecasting Benchmark

> [!NOTE]
> This dataset reflects e-commerce retail transactions and serves as a supplementary transactional and time-series forecasting benchmark.

## 1. Time-Series Revenue Forecasting (28-Day Chronological Test Holdout)

| Model | MAE | RMSE | MAPE |
| --- | --- | --- | --- |
| `naive_seasonal_7d` | £18,124.90 | £33,874.70 | 30.24% |
| `moving_average_7d` | £25,728.35 | £39,449.83 | 33.34% |
| `holt_winters_exponential_smoothing` | £17,603.63 | £31,459.96 | 24.88% |

## 2. RFM Customer Segmentation Profiles

| Cluster ID | Customer Count | Share (%) | Avg Recency (Days) | Avg Frequency | Avg Monetary Spend |
| --- | --- | --- | --- | --- | --- |
| `0` | 713 | 16.44% | 12.2d | 13.8 | £8,088.02 |
| `1` | 1,622 | 37.39% | 181.5d | 1.3 | £341.00 |
| `2` | 837 | 19.29% | 17.7d | 2.2 | £557.32 |
| `3` | 1,166 | 26.88% | 71.6d | 4.1 | £1,801.78 |
