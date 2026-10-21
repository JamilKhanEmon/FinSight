"""Unit tests for the research-harmonized feature engineering and PCA pipeline."""

import numpy as np
import pandas as pd

from ai.features.research_harmonizer import (
    ResearchInteractionGenerator,
    build_research_harmonized_pipeline,
)


def test_research_interaction_generator():
    np.random.seed(42)
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat_b": [2.0, 4.0, 6.0, 8.0, 10.0],
        "feat_c": [5.0, 4.0, 3.0, 2.0, 1.0],
        "cat_d": ["x", "y", "x", "y", "x"],
    })

    gen = ResearchInteractionGenerator(top_n_interactions=2)
    gen.fit(df)

    assert len(gen.selected_pairs) > 0
    transformed = gen.transform(df)

    # Check that interaction and ratio columns were created
    inter_cols = [c for c in transformed.columns if c.startswith("inter_")]
    ratio_cols = [c for c in transformed.columns if c.startswith("ratio_")]
    assert len(inter_cols) > 0
    assert len(ratio_cols) > 0
    assert len(transformed) == len(df)


def test_research_harmonized_pipeline_with_pca():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 6), columns=[f"col_{i}" for i in range(6)])

    pipe = build_research_harmonized_pipeline(apply_pca=True, pca_variance=0.95, top_n_interactions=3)
    X_trans = pipe.fit_transform(X)

    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == 50
    # PCA retaining 95% variance should reduce dimensions while maintaining information
    assert X_trans.shape[1] > 0
