from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "configs"


def load_yaml(file_path: Path) -> dict[str, Any]:
    """Safely parse a YAML configuration file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def get_project_config() -> dict[str, Any]:
    """Load core platform configuration."""
    return load_yaml(CONFIG_DIR / "project.yaml")


def get_datasets_config() -> dict[str, Any]:
    """Load dataset registry, license metadata, and column schemas."""
    return load_yaml(CONFIG_DIR / "datasets.yaml")


def get_models_config() -> dict[str, Any]:
    """Load model architectures, hyperparameters, and evaluation settings."""
    return load_yaml(CONFIG_DIR / "models.yaml")


def get_resolved_path(rel_path: str) -> Path:
    """Resolve a project-relative path against the repository root."""
    return BASE_DIR / rel_path
