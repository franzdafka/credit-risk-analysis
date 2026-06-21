# Credit Risk Assessment Service

A deployable credit scoring service built with FastAPI, scikit-learn, and SHAP. Exposes a REST API for probability-of-default predictions with per-decision explanations, and includes a Streamlit UI for interactive underwriting simulations.

---

## Model Performance

Trained on the [German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) — 1,000 loan applicants, 20 features.

| Metric | Value |
|---|---|
| AUC-ROC | 0.674 |
| Gini Coefficient | 0.348 |
| Algorithm | Logistic Regression (class-weighted) |
| Train / Test split | 80 / 20, stratified |

Gini = 2 × AUC − 1. Values above 0.30 are considered acceptable for a baseline credit scoring model.

---

## Architecture

```
Streamlit UI (app.py)
      │
      │  HTTP / JSON
      ▼
FastAPI service (api.py)
      │
      ▼
credit_model.py
  ├── feature engineering pipeline
  ├── LogisticRegression classifier
  └── SHAP LinearExplainer
```

| File | Role |
|---|---|
| `credit_model.py` | Model training, feature engineering, SHAP inference |
| `api.py` | REST API — `/predict`, `/explain`, `/health` |
| `app.py` | Streamlit UI, calls the API with local model fallback |
| `train_model.py` | Trains and serializes the model artifact to `artifacts/` |
| `tests/test_api.py` | Smoke, boundary, and monotonicity tests |

---

## API Endpoints

### POST /predict

Returns probability of default and a risk band.

```bash
curl -X POST http://localhost:8000/predict \
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

```json
{
  "probability_default": 0.18,
  "predicted_default": false,
  "risk_band": "low"
}
```

### GET /explain?user_id=\<id\>

Returns the top 3 risk-increasing and top 3 risk-reducing SHAP factors for a given applicant.

```bash
curl "http://localhost:8000/explain?user_id=0"
```

```json
{
  "top_positive": [
    { "feature": "num__dti_ratio", "shap_value": 0.42 },
    { "feature": "num__duration",  "shap_value": 0.31 }
  ],
  "top_negative": [
    { "feature": "num__age", "shap_value": -0.27 }
  ]
}
```

### GET /health

```json
{
  "status": "ok",
  "model_version": "credit-risk-20260621170834"
}
```

---

## Feature Engineering

The model uses 14 features: 9 from the raw dataset plus 5 engineered ones.

| Feature | Description |
|---|---|
| `dti_ratio` | Loan amount divided by estimated income proxy |
| `employment_length` | Ordinal encoding of employment duration band |
| `number_of_delinquencies` | Binary flag derived from credit history text |
| `credit_history_length` | Age-based proxy for length of credit track record |
| `loan_purpose` | Passthrough of `purpose` for one-hot encoding |

---

## Quick Start

**Local Python**

```bash
pip install -r requirements.txt
python3 train_model.py

uvicorn api:app --reload --port 8000   # terminal 1
streamlit run app.py                   # terminal 2
```

- API docs: http://localhost:8000/docs
- UI: http://localhost:8501

**Docker Compose**

```bash
docker compose up --build
```

---

## Tests

```bash
pytest
```

Covers smoke tests (`/health`, `/predict`, `/explain`), boundary inputs (`income=1`, `amount=10_000_000`), and a monotonicity check that higher income generally reduces the predicted default probability.

---

## Known Limitations

- No macroeconomic features (interest rates, unemployment, regional data)
- No data drift or concept drift monitoring
- Fairness audit across demographic proxies not yet implemented
- Model is retrained offline; no online learning loop
