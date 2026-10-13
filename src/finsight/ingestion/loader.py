"""Data Ingestion Module for FinSight Platform.

Loads datasets safely respecting memory constraints on macOS Apple Silicon.
Supports CSV, Excel (xls/xlsx), and Parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from finsight.config.loader import get_datasets_config, get_resolved_path
from finsight.utils.logger import setup_logger

logger = setup_logger("finsight.ingestion")


def get_file_info(file_path: Path) -> dict[str, Any]:
    """Inspect file existence, size, and modification timestamp."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    stat = file_path.stat()
    return {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
    }


def load_raw_dataset(
    dataset_key: str,
    nrows: int | None = None,
    use_polars: bool = False,
) -> pd.DataFrame | pl.DataFrame:
    """Load a raw dataset by its key from configs/datasets.yaml.

    Args:
        dataset_key: Registry key for the dataset (e.g. 'bank_marketing')
        nrows: Optional limit on number of rows to load (useful for exploration/tests)
        use_polars: If True, returns a Polars DataFrame where applicable

    Returns:
        pd.DataFrame or pl.DataFrame
    """
    datasets_cfg = get_datasets_config()
    if dataset_key not in datasets_cfg["datasets"]:
        raise KeyError(f"Dataset key '{dataset_key}' not defined in datasets.yaml")

    d_cfg = datasets_cfg["datasets"][dataset_key]
    raw_subpath = d_cfg["raw_subpath"]
    full_path = get_resolved_path("data/raw") / raw_subpath

    file_info = get_file_info(full_path)
    logger.info(
        f"Loading dataset '{dataset_key}' from {file_info['file_name']} "
        f"({file_info['size_mb']} MB, nrows={nrows or 'all'})..."
    )

    fmt = d_cfg.get("file_format", "csv").lower()
    header_row = d_cfg.get("header_row", 0)
    delimiter = d_cfg.get("delimiter", ",")

    if fmt == "csv":
        if use_polars and nrows is None and delimiter == ",":
            return pl.read_csv(full_path)
        # Use pandas
        sep = delimiter if delimiter else ","
        df = pd.read_csv(
            full_path,
            sep=sep,
            header=header_row,
            nrows=nrows,
            low_memory=False,
        )
    elif fmt in ("xls", "xlsx"):
        df = pd.read_excel(
            full_path,
            header=header_row,
            nrows=nrows,
        )
    else:
        raise ValueError(f"Unsupported file format '{fmt}' for dataset '{dataset_key}'")

    logger.info(f"Loaded '{dataset_key}' successfully with shape {df.shape}.")
    return df
