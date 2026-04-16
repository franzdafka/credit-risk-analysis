from fastapi import FastAPI
from pydantic import BaseModel, Field

from credit_model import predict_risk

app = FastAPI(title="Fintech Credit Risk API", version="1.0.0")


class CreditRequest(BaseModel):
    age: int = Field(ge=18, le=100)
    annual_income: float = Field(gt=0)
    loan_amount: float = Field(gt=0)


class CreditResponse(BaseModel):
    probability_default: float
    predicted_default: bool
    decision: str
    risk_band: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


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
