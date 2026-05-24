from fastapi import FastAPI
from pydantic import BaseModel, Field
from credit_model import predict_risk, get_model_version, get_model_metrics

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
    prediction = predict_risk(request.model_dump())
    decision = "reject" if prediction.predicted_default else "approve"
    return CreditResponse(
        probability_default=prediction.probability_default,
        predicted_default=prediction.predicted_default,
        decision=decision,
        risk_band=prediction.risk_band,
    )
