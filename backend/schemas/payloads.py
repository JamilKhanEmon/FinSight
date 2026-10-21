from __future__ import annotations

from pydantic import BaseModel, Field


class MarketingLeadInput(BaseModel):
    age: int = Field(..., ge=18, le=100, json_schema_extra={"example": 35})
    job: str = Field(..., json_schema_extra={"example": "admin."})
    marital: str = Field(..., json_schema_extra={"example": "married"})
    education: str = Field(..., json_schema_extra={"example": "university.degree"})
    default: str = Field("no", json_schema_extra={"example": "no"})
    housing: str = Field("yes", json_schema_extra={"example": "yes"})
    loan: str = Field("no", json_schema_extra={"example": "no"})
    contact: str = Field("cellular", json_schema_extra={"example": "cellular"})
    month: str = Field("may", json_schema_extra={"example": "may"})
    day_of_week: str = Field("mon", json_schema_extra={"example": "mon"})
    campaign: int = Field(1, ge=1, le=50, json_schema_extra={"example": 1})
    pdays: int = Field(999, ge=0, le=999, json_schema_extra={"example": 999})
    previous: int = Field(0, ge=0, le=20, json_schema_extra={"example": 0})
    poutcome: str = Field("nonexistent", json_schema_extra={"example": "nonexistent"})
    emp_var_rate: float = Field(1.1, json_schema_extra={"example": 1.1})
    cons_price_idx: float = Field(93.994, json_schema_extra={"example": 93.994})
    cons_conf_idx: float = Field(-36.4, json_schema_extra={"example": -36.4})
    euribor3m: float = Field(4.857, json_schema_extra={"example": 4.857})
    nr_employed: float = Field(5191.0, json_schema_extra={"example": 5191.0})


class CreditRiskInput(BaseModel):
    person_age: int = Field(..., ge=18, le=100, json_schema_extra={"example": 28})
    person_income: float = Field(..., ge=0, json_schema_extra={"example": 65000.0})
    person_home_ownership: str = Field(..., json_schema_extra={"example": "RENT"})
    person_emp_length: float = Field(2.0, ge=0, le=60, json_schema_extra={"example": 3.0})
    loan_intent: str = Field(..., json_schema_extra={"example": "PERSONAL"})
    loan_grade: str = Field(..., json_schema_extra={"example": "B"})
    loan_amnt: float = Field(..., ge=500, json_schema_extra={"example": 10000.0})
    loan_int_rate: float = Field(11.14, ge=1.0, le=40.0, json_schema_extra={"example": 11.14})
    loan_percent_income: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.15})
    cb_person_default_on_file: str = Field("N", json_schema_extra={"example": "N"})
    cb_person_cred_hist_length: int = Field(4, ge=0, le=50, json_schema_extra={"example": 4})


class ChurnInput(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850, json_schema_extra={"example": 650})
    Geography: str = Field("France", json_schema_extra={"example": "France"})
    Gender: str = Field("Female", json_schema_extra={"example": "Female"})
    Age: int = Field(..., ge=18, le=100, json_schema_extra={"example": 42})
    Tenure: int = Field(..., ge=0, le=10, json_schema_extra={"example": 5})
    Balance: float = Field(..., ge=0, json_schema_extra={"example": 75000.0})
    NumOfProducts: int = Field(..., ge=1, le=4, json_schema_extra={"example": 1})
    HasCrCard: int = Field(1, ge=0, le=1, json_schema_extra={"example": 1})
    IsActiveMember: int = Field(1, ge=0, le=1, json_schema_extra={"example": 1})
    EstimatedSalary: float = Field(..., ge=0, json_schema_extra={"example": 60000.0})


class FraudTxInput(BaseModel):
    Amount: float = Field(..., ge=0.0, json_schema_extra={"example": 125.50})
    Time: float = Field(..., ge=0.0, json_schema_extra={"example": 3600.0})
    features: dict[str, float] = Field(default_factory=dict, description="V1 to V28 PCA features")
