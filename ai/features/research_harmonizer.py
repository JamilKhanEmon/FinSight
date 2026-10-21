from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai.utils.logger import setup_logger

logger = setup_logger("ai.features.research_harmonizer")


class ResearchInteractionGenerator(BaseEstimator, TransformerMixin):
    """Generates pairwise numerical interaction and ratio features."""

    def __init__(self, top_n_interactions: int = 15):
        self.top_n_interactions = top_n_interactions
        self.selected_pairs: list[tuple[str, str]] = []

    def fit(self, X: pd.DataFrame, y=None):
        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if len(numeric_cols) < 2:
            return self

        corr_matrix = X[numeric_cols].corr().abs()
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                val = corr_matrix.loc[col1, col2]
                if not np.isnan(val):
                    pairs.append(((col1, col2), val))

        pairs.sort(key=lambda x: x[1], reverse=True)
        self.selected_pairs = [p[0] for p in pairs[:self.top_n_interactions]]
        logger.info(f"Selected {len(self.selected_pairs)} interaction pairs from feature correlations.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col1, col2 in self.selected_pairs:
            if col1 in df.columns and col2 in df.columns:
                inter_col = f"inter_{col1}_{col2}"
                df[inter_col] = (df[col1] * df[col2]).replace([np.inf, -np.inf], np.nan)

                denom = df[col2].replace(0, np.nan)
                ratio_col = f"ratio_{col1}_{col2}"
                df[ratio_col] = (df[col1] / denom).replace([np.inf, -np.inf], np.nan)

        return df.replace([np.inf, -np.inf], np.nan)


def build_research_harmonized_pipeline(
    apply_pca: bool = True,
    pca_variance: float = 0.95,
    top_n_interactions: int = 12,
) -> Pipeline:
    """Builds an interaction-augmented dimensionality reduction pipeline."""
    steps = [
        ("interaction_fe", ResearchInteractionGenerator(top_n_interactions=top_n_interactions)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if apply_pca:
        steps.append(("pca", PCA(n_components=pca_variance, random_state=42)))

    return Pipeline(steps)
