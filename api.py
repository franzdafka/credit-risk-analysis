from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from credit_model import predict_risk, get_model_version, explain_user_risk, validate_and_normalize_categories, CategoryValidationError

app = FastAPI(title="Fintech Credit Risk API", version="1.0.0")

class CreditRequest(BaseModel):
    duration: int = Field(ge=1, le=120)
    amount: float = Field(gt=0)
    income: float = Field(gt=0)
    age: int = Field(ge=18, le=100)
    installment_rate: int = Field(ge=1, le=4)
    number_credits: int = Field(ge=1, le=10)
    people_liable: int = Field(ge=1, le=2)
    purpose: str = "car"
    credit_history: str = "existing paid"
    employment_duration: str = "1 <= ... < 4 yrs"

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
    try:
        payload = validate_and_normalize_categories(request.model_dump())
    except CategoryValidationError as e:
        raise HTTPException(status_code=422, detail={"field": e.field, "value": e.value, "allowed": e.allowed})
    prediction = predict_risk(payload)
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
        return explain_user_risk(user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
