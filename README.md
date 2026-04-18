# Credit Risk Assessment Service

Production-oriented credit scoring service built with **FastAPI**, a serialized **scikit-learn** model artifact, and SHAP-based decision explanations.

## Service Overview

The service exposes three primary endpoints:

- `POST /predict` — returns probability of default and underwriting decision.
- `GET /explain?user_id=<id>` — returns top 3 positive and top 3 negative SHAP factors for the selected user profile.
- `GET /health` — returns service status and deployed model version.

## Architecture

- **Model training:** `train_model.py`
- **Model runtime utilities:** `credit_model.py`
- **API service:** `api.py`
- **Automated tests:** `tests/test_api.py`

## MLOps and Model Versioning

The API does **not retrain on startup**. Instead, the application loads a serialized artifact from:

- `artifacts/credit_risk_model.joblib`

The artifact contains:

- Fitted preprocessing + classifier pipeline
- Model metrics (AUC-ROC, Gini)
- Version string (example: `credit-risk-20260418102030`)
- Reference feature frame with `user_id` used by `/explain`

### Train and Version a Model Artifact

```bash
python train_model.py
```

### Run API

```bash
uvicorn api:app --reload --port 8000
```

## Testing Strategy

Tests are written in `tests/test_api.py` and include:

1. **Smoke tests**
   - `/health` connectivity and version visibility
   - `/predict` basic response contract
   - `/explain` top-k output contract

2. **Boundary tests**
   - Extreme financial inputs (`income=1`, `amount=10_000_000`)

3. **Invariance / Monotonicity tests**
   - Ceteris paribus check that higher income generally lowers risk score

Run tests:

```bash
pytest
```

## Example Requests

### Predict

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 24,
    "amount": 5000,
    "income": 4200,
    "age": 35,
    "installment_rate": 2,
    "number_credits": 1,
    "people_liable": 1,
    "purpose": "car",
    "credit_history": "existing paid",
    "employment_duration": "1 <= ... < 4 yrs"
  }'
```

### Explain

```bash
curl "http://127.0.0.1:8000/explain?user_id=0"
```

### Health

```bash
curl "http://127.0.0.1:8000/health"
```

## Limitations

- **No macroeconomic features yet:** The model does not include unemployment, inflation, interest rates, or region-level shocks.
- **No production drift monitoring:** There is no live data drift / concept drift detection pipeline in this repository.
- **Fairness assessment pending:** A formal bias and disparate impact audit has not yet been implemented.

## Optional UI

A Streamlit app is included in `app.py` for local exploration.

```bash
streamlit run app.py
```
