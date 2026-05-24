# Credit Risk Assessment Service

![CI](https://github.com/franzdafka/credit-risk-analysis/actions/workflows/ci.yml/badge.svg)

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

## Modelling Decisions

**Why Logistic Regression?**  
Logistic regression was chosen as the primary classifier for three reasons: (1) its outputs are calibrated probabilities by construction, which is critical for threshold-based underwriting decisions; (2) it satisfies the interpretability requirements typical of regulated credit environments (cf. GDPR Art. 22 on automated decision-making); (3) it serves as a well-understood baseline — any complexity added via ensemble methods should demonstrably improve AUC-ROC to justify the interpretability trade-off.

**Model performance**  
The fitted model achieves a Gini coefficient of approximately **0.45–0.55** on the German Credit holdout set (AUC-ROC ≈ 0.72–0.78). In retail credit scoring, a Gini above 0.40 is generally considered acceptable for a single-bureau scorecard; above 0.60 is strong. The current score reflects the limited feature set — no bureau data, no behavioural signals.

**Why SHAP for explanations?**  
For a linear model, SHAP values are equivalent to the product of feature coefficients and mean-centred inputs — they are mathematically exact, not approximate. This means the `/explain` endpoint produces auditable, legally defensible explanations rather than post-hoc approximations.

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

```md
Run tests locally:

```bash
pytest tests/ -v

## CI/CD

The repository includes a GitHub Actions CI pipeline that runs on every push and pull request to `main`.

The pipeline:
- installs Python dependencies
- trains the model artifact
- runs the API test suite

This ensures the model service is reproducible in a clean environment, not only on a local machine.

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
