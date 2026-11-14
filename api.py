from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import credit_model as cm

app = FastAPI(
    title="Credit Risk PD Scoring Service",
    description=(
        "Calibrated 12-month probability of default (PD) for retail applicants, trained on the "
        "Statlog (German Credit) dataset. Returns a PD, a rating grade and an approve/decline "
        "decision at a cost-optimised threshold, plus SHAP explanations."
    ),
    version="2.0.0",
)


class CreditRequest(BaseModel):
    """All 20 Statlog attributes. Categorical values must match the dataset labels (see /categories)."""

    status: str = Field(description="Checking account status, e.g. 'no checking account'.")
    duration: int = Field(ge=1, le=120, description="Loan duration in months.")
    credit_history: str
    purpose: str
    amount: float = Field(gt=0, description="Loan amount (DM).")
    savings: str
    employment_duration: str
    installment_rate: int = Field(ge=1, le=4, description="Instalment as % of disposable income (1 low to 4 high).")
    personal_status_sex: str
    other_debtors: str
    present_residence: int = Field(ge=1, le=4)
    property: str
    age: int = Field(ge=18, le=100)
    other_installment_plans: str
    housing: str
    number_credits: int = Field(ge=1, le=4)
    job: str
    people_liable: int = Field(ge=1, le=2)
    telephone: str
    foreign_worker: str


class CreditResponse(BaseModel):
    probability_default: float = Field(description="Calibrated PD; the PD in EL = PD x LGD x EAD.")
    rating_grade: Literal["A", "B", "C", "D", "E"] = Field(
        description="A <5%, B 5-10%, C 10-20%, D 20-35%, E >=35% PD."
    )
    decision: Literal["approve", "decline"]
    decision_threshold: float = Field(description="Decline when PD >= this value (chosen on training data).")


@app.get("/health")
def health() -> dict:
    art = cm.load_artifact()
    return {
        "status": "ok",
        "model_version": art["version"],
        "model": art["selected_model"],
        "calibration": art["calibration"],
    }


@app.get("/categories", summary="Allowed values for each categorical field")
def categories() -> dict[str, list[str]]:
    return cm.load_artifact()["categories"]


@app.post("/predict", response_model=CreditResponse, summary="Score a credit applicant")
def predict(request: CreditRequest) -> CreditResponse:
    try:
        return CreditResponse(**cm.predict(request.model_dump()))
    except cm.CategoryValidationError as e:
        raise HTTPException(status_code=422, detail={"field": e.field, "value": e.value, "allowed": e.allowed})


@app.get("/explain", summary="Top SHAP drivers for an applicant in the reference dataset")
def explain(user_id: int) -> dict:
    try:
        return cm.explain(user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
