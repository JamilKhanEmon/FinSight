"""Transaction Analysis, Time-Series Forecasting, and RFM Segmentation Module."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from finsight.cleaning.cleaner import DatasetCleaner
from finsight.config.loader import get_resolved_path
from finsight.ingestion.loader import load_raw_dataset
from finsight.utils.logger import setup_logger

logger = setup_logger("finsight.analytics.retail")


def prepare_daily_retail_series(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned transactions into daily sales, order count, and customer metrics."""
    df = cleaned_df.copy()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["Date"] = df["InvoiceDate"].dt.date

    daily = df.groupby("Date").agg(
        total_revenue=("Revenue", "sum"),
        order_count=("InvoiceNo", "nunique"),
        units_sold=("Quantity", "sum"),
        active_customers=("CustomerID", "nunique"),
    ).reset_index()

    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values("Date").reset_index(drop=True)

    # Reindex to full daily frequency to fill gaps (e.g. non-trading days filled with 0)
    full_idx = pd.date_range(start=daily["Date"].min(), end=daily["Date"].max(), freq="D")
    daily = daily.set_index("Date").reindex(full_idx, fill_value=0.0).reset_index()
    daily = daily.rename(columns={"index": "Date"})

    return daily


def run_time_series_forecasting(
    daily_df: pd.DataFrame,
    target_col: str = "total_revenue",
    test_days: int = 28,
) -> dict[str, Any]:
    """Evaluates time-series forecasting models using strict chronological hold-out splits."""
    series = daily_df.set_index("Date")[target_col].astype(float)
    train_series = series.iloc[:-test_days]
    test_series = series.iloc[-test_days:]

    # 1. Baseline: 7-day Naive Seasonal
    naive_pred = train_series.iloc[-test_days:].values  # seasonal repeater

    # 2. Baseline: 7-day Moving Average
    ma_val = train_series.rolling(7).mean().iloc[-1]
    ma_pred = np.full(test_days, ma_val)

    # 3. Holt-Winters Exponential Smoothing
    hw_pred = np.zeros(test_days)
    try:
        # Add small constant to avoid zero issues in multiplicative trend
        adj_train = train_series + 1.0
        hw_model = ExponentialSmoothing(
            adj_train,
            trend="add",
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated",
        ).fit()
        hw_pred = np.maximum(0, hw_model.forecast(test_days).values - 1.0)
    except Exception as exc:
        logger.warning(f"Exponential Smoothing fit exception: {exc}. Using Moving Average fallback.")
        hw_pred = ma_pred

    def calc_ts_metrics(y_true, y_pred, model_name):
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(root_mean_squared_error(y_true, y_pred))
        non_zero = y_true > 0
        mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]))) if non_zero.sum() > 0 else 0.0
        return {
            "model_name": model_name,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 4),
        }

    results = {
        "naive_seasonal_7d": calc_ts_metrics(test_series.values, naive_pred, "naive_seasonal_7d"),
        "moving_average_7d": calc_ts_metrics(test_series.values, ma_pred, "moving_average_7d"),
        "holt_winters_exponential_smoothing": calc_ts_metrics(test_series.values, hw_pred, "holt_winters_exponential_smoothing"),
        "train_period": f"{train_series.index.min().strftime('%Y-%m-%d')} to {train_series.index.max().strftime('%Y-%m-%d')}",
        "test_period": f"{test_series.index.min().strftime('%Y-%m-%d')} to {test_series.index.max().strftime('%Y-%m-%d')}",
        "test_days": test_days,
    }

    return results


def run_rfm_customer_segmentation(cleaned_df: pd.DataFrame, n_clusters: int = 4) -> dict[str, Any]:
    """Performs Recency, Frequency, Monetary (RFM) segmentation on verified customer records."""
    df = cleaned_df.dropna(subset=["CustomerID"]).copy()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary=("Revenue", "sum"),
    ).reset_index()

    # Filter invalid non-positive monetary
    rfm = rfm[rfm["monetary"] > 0].reset_index(drop=True)

    # Log-transform to handle skew
    rfm_log = pd.DataFrame({
        "recency_log": np.log1p(rfm["recency"]),
        "frequency_log": np.log1p(rfm["frequency"]),
        "monetary_log": np.log1p(rfm["monetary"]),
    })

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm_log)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(scaled_features)

    # Segment summaries
    cluster_profiles = []
    for c in range(n_clusters):
        subset = rfm[rfm["cluster"] == c]
        profile = {
            "cluster_id": c,
            "customer_count": len(subset),
            "pct_of_customers": round(len(subset) / len(rfm) * 100, 2),
            "avg_recency_days": round(float(subset["recency"].mean()), 1),
            "avg_frequency_orders": round(float(subset["frequency"].mean()), 1),
            "avg_monetary_spend": round(float(subset["monetary"].mean()), 2),
        }
        cluster_profiles.append(profile)

    return {
        "total_segmented_customers": len(rfm),
        "n_clusters": n_clusters,
        "clusters": cluster_profiles,
    }


def run_retail_pipeline(force_retrain: bool = False) -> dict[str, Any]:
    """Execute Online Retail analytics, time-series forecasting, and RFM segmentation."""
    metrics_dir = get_resolved_path("artifacts/metrics")
    metrics_file = metrics_dir / "retail_analytics_metrics.json"

    if metrics_file.exists() and not force_retrain:
        logger.info(f"Existing retail metrics found at {metrics_file}. Loading cached results.")
        with open(metrics_file, "r") as f:
            return json.load(f)

    logger.info("=== Starting Online Retail Transaction Analytics Execution ===")
    raw_df = load_raw_dataset("online_retail")

    cleaner = DatasetCleaner("online_retail")
    cleaned_df, clean_summary = cleaner.clean_online_retail(raw_df)

    daily_df = prepare_daily_retail_series(cleaned_df)
    forecasting_results = run_time_series_forecasting(daily_df, test_days=28)
    segmentation_results = run_rfm_customer_segmentation(cleaned_df, n_clusters=4)

    payload = {
        "forecasting": forecasting_results,
        "segmentation": segmentation_results,
        "clean_summary": clean_summary,
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Save markdown summary
    reports_dir = get_resolved_path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "retail_transaction_report.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Online Retail Transaction Analytics & Forecasting Benchmark\n\n")
        f.write("> [!NOTE]\n")
        f.write(
            "> This dataset reflects e-commerce retail transactions and serves as a supplementary transactional "
            "and time-series forecasting benchmark.\n\n"
        )
        f.write("## 1. Time-Series Revenue Forecasting (28-Day Chronological Test Holdout)\n\n")
        f.write("| Model | MAE | RMSE | MAPE |\n")
        f.write("| --- | --- | --- | --- |\n")
        for m_name in ["naive_seasonal_7d", "moving_average_7d", "holt_winters_exponential_smoothing"]:
            res = forecasting_results[m_name]
            f.write(f"| `{m_name}` | £{res['mae']:,.2f} | £{res['rmse']:,.2f} | {res['mape']:.2%} |\n")
        f.write("\n")
        f.write("## 2. RFM Customer Segmentation Profiles\n\n")
        f.write("| Cluster ID | Customer Count | Share (%) | Avg Recency (Days) | Avg Frequency | Avg Monetary Spend |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        f.writelines(f"| `{c['cluster_id']}` | {c['customer_count']:,} | {c['pct_of_customers']}% | "
                f"{c['avg_recency_days']}d | {c['avg_frequency_orders']} | £{c['avg_monetary_spend']:,.2f} |\n" for c in segmentation_results["clusters"])

    logger.info(f"Saved retail analytics report to {report_file}")
    return payload
