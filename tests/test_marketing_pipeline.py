"""Unit and integration tests for Bank Marketing feature engineering and training."""

import pandas as pd

from finsight.features.marketing import MarketingDomainFeatureGenerator


def test_prospective_excludes_duration():
    df = pd.DataFrame({
        "age": [25, 55],
        "duration": [120, 450],
        "campaign": [2, 1],
        "pdays": [999, 6],
        "housing": ["yes", "no"],
        "loan": ["yes", "no"],
        "euribor3m": [4.8, 1.2],
        "nr_employed": [5191.0, 5000.0],
        "cons_price_idx": [93.9, 92.8],
        "cons_conf_idx": [-36.4, -40.0],
    })

    gen = MarketingDomainFeatureGenerator(include_duration=False)
    transformed = gen.transform(df)
    assert "duration" not in transformed.columns
    assert "previously_contacted" in transformed.columns
    assert transformed["previously_contacted"].tolist() == [0, 1]


def test_retrospective_keeps_duration():
    df = pd.DataFrame({
        "age": [25, 55],
        "duration": [120, 450],
        "campaign": [2, 1],
        "pdays": [999, 6],
        "housing": ["yes", "no"],
        "loan": ["yes", "no"],
        "euribor3m": [4.8, 1.2],
        "nr_employed": [5191.0, 5000.0],
        "cons_price_idx": [93.9, 92.8],
        "cons_conf_idx": [-36.4, -40.0],
    })

    gen = MarketingDomainFeatureGenerator(include_duration=True)
    transformed = gen.transform(df)
    assert "duration" in transformed.columns
