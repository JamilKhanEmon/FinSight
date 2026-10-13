"""Configuration loader for FinSight Platform."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
BASE_DIR = Path(__file__).resolve().parents[3]


def load_yaml(file_path: Path) -> dict[str, Any]:
    """Safely load a YAML configuration file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_project_config() -> dict[str, Any]:
    """Load core project configuration."""
    return load_yaml(CONFIG_DIR / "project.yaml")


def get_datasets_config() -> dict[str, Any]:
    """Load dataset registry and task specifications."""
    return load_yaml(CONFIG_DIR / "datasets.yaml")


def get_models_config() -> dict[str, Any]:
    """Load model architectures, parameters, and tuning configs."""
    return load_yaml(CONFIG_DIR / "models.yaml")


def get_resolved_path(rel_path: str) -> Path:
    """Resolve a relative project path against project root."""
    return BASE_DIR / rel_path
