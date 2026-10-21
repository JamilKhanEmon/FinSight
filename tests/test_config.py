from ai.config.loader import get_datasets_config, get_models_config, get_project_config


def test_project_config_loaded():
    cfg = get_project_config()
    assert "project" in cfg
    assert cfg["project"]["name"] == "FinSight"
    assert "hardware" in cfg
    assert cfg["hardware"]["memory_ram_gb"] == 16


def test_datasets_config_loaded():
    cfg = get_datasets_config()
    assert "datasets" in cfg
    assert "bank_marketing" in cfg["datasets"]
    assert "customer_churn" in cfg["datasets"]
    assert "credit_risk" in cfg["datasets"]
    assert "fraud_detection" in cfg["datasets"]


def test_models_config_loaded():
    cfg = get_models_config()
    assert "marketing_response" in cfg
    assert "customer_churn" in cfg
    assert "credit_risk" in cfg
    assert "fraud_detection" in cfg
