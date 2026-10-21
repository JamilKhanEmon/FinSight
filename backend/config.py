from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendSettings:
    title: str = "FinSight Banking Intelligence API"
    description: str = (
        "Enterprise predictive inference microservice for prospective marketing, "
        "credit underwriting risk, and customer retention scoring."
    )
    version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000


settings = BackendSettings()
