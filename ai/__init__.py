"""FinSight AI Core Engine.

Unifies prospective marketing intelligence, retention modeling, credit risk underwriting,
real-time transaction fraud screening, and statistical feature engineering.
"""

from __future__ import annotations

import sys

__version__ = "0.1.0"


def _register_legacy_shims() -> None:
    """Maps legacy 'finsight' namespace to 'ai', 'backend', and 'research' in sys.modules.

    This ensures serialized Joblib pipelines and legacy scripts load without requiring
    duplicate physical wrapper files on disk.
    """
    from ai import config, data, evaluation, features, models, pipelines
    from ai.config import loader as config_loader
    from ai.data import cleaner, ingestion, validation
    from ai.evaluation import evaluator
    from ai.features import (
        churn,
        credit_card_default,
        credit_risk,
        fraud,
        marketing,
        research_harmonizer,
        statistical_engine,
    )
    from ai.models import ensemble_engine, trainer

    legacy_map = {
        "finsight": sys.modules[__name__],
        "finsight.config": config,
        "finsight.config.loader": config_loader,
        "finsight.data": data,
        "finsight.cleaning": data,
        "finsight.cleaning.cleaner": cleaner,
        "finsight.ingestion": data,
        "finsight.ingestion.loader": ingestion,
        "finsight.validation": data,
        "finsight.validation.schema": validation,
        "finsight.features": features,
        "finsight.features.marketing": marketing,
        "finsight.features.churn": churn,
        "finsight.features.credit_risk": credit_risk,
        "finsight.features.fraud": fraud,
        "finsight.features.credit_card_default": credit_card_default,
        "finsight.features.statistical_feature_engine": statistical_engine,
        "finsight.features.research_harmonizer": research_harmonizer,
        "finsight.models": models,
        "finsight.models.trainer": trainer,
        "finsight.models.ensemble_engine": ensemble_engine,
        "finsight.evaluation": evaluation,
        "finsight.evaluation.evaluator": evaluator,
        "finsight.pipeline": pipelines,
    }

    for key, mod in legacy_map.items():
        sys.modules.setdefault(key, mod)


_register_legacy_shims()

