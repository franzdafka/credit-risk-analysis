from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

codex/update-api-and-readme-for-credit-risk-assessment-hblksp
from credit_model import (
    CategoryValidationError,
    explain_user_risk,
    get_model_version,
    predict_risk,
    validate_and_normalize_categories,
)
 
from credit_model import explain_user_risk, get_model_version, predict_risk
main

app = FastAPI(title="Credit Risk Assessment Service", version="2.0.0")


class CreditRequest(BaseModel):
    duration: int = Field(ge=4, le=72)
    amount: float = Field(gt=0, description="Loan amount")
    income: float = Field(gt=0, description="Applicant monthly income")
    age: int = Field(ge=18, le=100)
    installment_rate: int = Field(ge=1, le=4)
    number_credits: int = Field(ge=1, le=4)
    people_liable: int = Field(ge=1, le=2)
    purpose: str = Field(default="furniture/equipment")
    credit_history: str = Field(default="existing paid")
    employment_duration: str = Field(default="1 <= ... < 4 yrs")


class CreditResponse(BaseModel):
    probability_default: float
    predicted_default: bool
    decision: str
    risk_band: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": get_model_version()}


@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest) -> CreditResponse:
    payload = request.model_dump()
    try:
        normalized_payload = validate_and_normalize_categories(payload)
    except CategoryValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Invalid category value.",
                "field": exc.field,
                "received": exc.value,
                "allowed_values": exc.allowed,
            },
        ) from exc

    prediction = predict_risk(normalized_payload)
    decision = "reject" if prediction.predicted_default else "approve"
    return CreditResponse(
        probability_default=prediction.probability_default,
        predicted_default=prediction.predicted_default,
        decision=decision,
        risk_band=prediction.risk_band,
    )


@app.get("/explain")
def explain(user_id: int) -> dict:
    try:
        explanation = explain_user_risk(user_id=user_id, top_k=3)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"user_id": user_id, **explanation}
