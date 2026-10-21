from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from ai.config.loader import get_datasets_config, get_resolved_path
from ai.utils.logger import setup_logger

logger = setup_logger("ai.data.ingestion")


def get_file_info(file_path: Path) -> dict[str, Any]:
    """Inspects file existence, size, and metadata."""
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
    """Loads a raw dataset by its registry key defined in configs/datasets.yaml."""
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

    fmt = d_cfg.get("format", "csv").lower()
    sep = d_cfg.get("delimiter", ",")

    if fmt == "csv":
        if use_polars:
            return pl.read_csv(full_path, n_rows=nrows, separator=sep)
        return pd.read_csv(full_path, nrows=nrows, sep=sep)
    elif fmt in ("excel", "xls", "xlsx"):
        header_row = d_cfg.get("header_row", 0)
        return pd.read_excel(full_path, nrows=nrows, header=header_row)
    elif fmt == "parquet":
        if use_polars:
            return pl.read_parquet(full_path, n_rows=nrows)
        return pd.read_parquet(full_path)
    else:
        raise ValueError(f"Unsupported format '{fmt}' for dataset '{dataset_key}'")
