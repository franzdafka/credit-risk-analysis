from fastapi import FastAPI
from pydantic import BaseModel, Field

from credit_model import get_model_metrics, predict_risk

app = FastAPI(title="Fintech Credit Risk API", version="1.1.0")


class CreditRequest(BaseModel):
    duration: int = Field(ge=4, le=72)
    amount: float = Field(gt=0)
    age: int = Field(ge=18, le=100)
    installment_rate: int = Field(ge=1, le=4)
    number_credits: int = Field(ge=1, le=4)
    people_liable: int = Field(ge=1, le=2)


class CreditResponse(BaseModel):
    probability_default: float
    predicted_default: bool
    decision: str
    risk_band: str


class MetricsResponse(BaseModel):
    auc_roc: float
    gini_coefficient: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    model_metrics = get_model_metrics()
    return MetricsResponse(
        auc_roc=model_metrics.auc_roc,
        gini_coefficient=model_metrics.gini_coefficient,
    )


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
