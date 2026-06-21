from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from credit_model import (
    predict_risk,
    get_model_version,
    explain_user_risk,
    validate_and_normalize_categories,
    CategoryValidationError,
)

app = FastAPI(
    title="Credit Risk Scoring Service -- Retail IRB",
    description=(
        "PD scorecard for retail credit risk assessment under the Basel III "
        "Internal Ratings-Based (IRB) approach. Returns a 12-month Probability "
        "of Default (PD) estimate and SHAP-based decision explanation for each applicant."
    ),
    version="1.0.0",
)


class CreditRequest(BaseModel):
    duration: int = Field(
        ge=1, le=120,
        description="Loan duration in months."
    )
    amount: float = Field(
        gt=0,
        description="Loan amount in DM (Deutsche Mark, dataset currency)."
    )
    age: int = Field(
        ge=18, le=100,
        description="Applicant age in years."
    )
    installment_rate: int = Field(
        ge=1, le=4,
        description="Installment rate as a percentage of disposable income (1=low, 4=high)."
    )
    number_credits: int = Field(
        ge=1, le=10,
        description="Number of existing credits at this bank."
    )
    people_liable: int = Field(
        ge=1, le=2,
        description="Number of people liable for maintenance (1 or 2)."
    )
    purpose: str = Field(
        default="car",
        description="Purpose of the loan (e.g. car, furniture/equipment, education, business)."
    )
    credit_history: str = Field(
        default="existing paid",
        description=(
            "Applicant credit history status. "
            "One of: 'existing paid', 'all paid', 'critical/other existing credit', "
            "'delay in paying off in the past', 'no credits/all paid'."
        )
    )
    employment_duration: str = Field(
        default="1 <= ... < 4 yrs",
        description=(
            "Duration of current employment. "
            "One of: 'unemployed', '< 1 yr', '1 <= ... < 4 yrs', '4 <= ... < 7 yrs', '>= 7 yrs'."
        )
    )


class CreditResponse(BaseModel):
    probability_default: float = Field(
        description="Estimated 12-month Probability of Default (PD). Maps to the PD parameter in EL = PD x LGD x EAD."
    )
    predicted_default: bool = Field(
        description="Classification at the 0.5 threshold. True = predicted default."
    )
    rating_grade: str = Field(
        description="Internal rating grade derived from PD: low (<0.30), medium (0.30-0.60), high (>0.60)."
    )
    underwriting_decision: str = Field(
        description="Indicative underwriting decision: approve or decline."
    )


@app.get("/health", summary="Service health and model version")
def health() -> dict:
    return {"status": "ok", "model_version": get_model_version()}


@app.post(
    "/predict",
    response_model=CreditResponse,
    summary="Score a credit applicant",
    description=(
        "Returns a PD estimate, internal rating grade, and indicative underwriting decision "
        "for a retail credit applicant. Inputs correspond to the German Credit Data (Statlog) "
        "feature set (UCI Machine Learning Repository, Hofmann 1994)."
    ),
)
def predict(request: CreditRequest) -> CreditResponse:
    try:
        payload = validate_and_normalize_categories(request.model_dump())
    except CategoryValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={"field": e.field, "value": e.value, "allowed": e.allowed},
        )
    prediction = predict_risk(payload)
    underwriting_decision = "decline" if prediction.predicted_default else "approve"
    return CreditResponse(
        probability_default=prediction.probability_default,
        predicted_default=prediction.predicted_default,
        rating_grade=prediction.risk_band,
        underwriting_decision=underwriting_decision,
    )


@app.get(
    "/explain",
    summary="SHAP-based decision explanation",
    description=(
        "Returns the top positive and negative SHAP factors driving the PD estimate "
        "for a given applicant in the reference dataset, using TreeSHAP for the "
        "underlying XGBoost model."
    ),
)
def explain(user_id: int) -> dict:
    try:
        return explain_user_risk(user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
