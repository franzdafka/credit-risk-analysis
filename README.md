# Credit Risk Scoring API

A production-ready credit scoring service built with FastAPI, scikit-learn, and SHAP — modelling the core underwriting decision logic used in consumer lending.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)
![CI](https://github.com/franzdafka/credit-risk-analysis/actions/workflows/ci.yml/badge.svg)

---

## Overview

Credit scoring pipeline trained on the [German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) (1,000 applicants), with predictions and SHAP-based explanations served via REST API.

In credit underwriting, a false negative — approving a borrower who defaults — carries asymmetric cost relative to a false positive. The model uses `class_weight="balanced"` and is evaluated primarily on **Precision**: the share of approved applicants who are genuinely low-risk.

---

## Model Performance

| Metric | Value |
|--------|-------|
| Precision | **79.4%** |
| AUC-ROC | 0.674 |
| Gini Coefficient | 0.348 |
| Recall | 57.9% |
| F1 Score | 0.669 |
| Test set | 200 samples (stratified split) |

Precision of 79.4% means 4 out of 5 approved applicants are genuine low-risk borrowers — directly minimising Type II error cost in the underwriting context.

### ROC Curve
![ROC Curve](roc_curve.png)

### SHAP Feature Importance

Decisions are explained using SHAP (SHapley Additive exPlanations), making each underwriting output auditable.

![SHAP Feature Importance](shap_importance.png)

Key risk drivers:
- **Loan duration** — longer terms significantly increase predicted default probability
- **DTI ratio** — debt-to-income proxy derived from installment rate and employment data
- **Credit history** — prior delinquencies are the strongest negative signal
- **Loan amount** — higher amounts correlate with elevated risk tier

---

## Architecture

\`\`\`
credit_model.py    — model training + SHAP inference logic
api.py             — FastAPI REST service (/predict, /explain, /health)
app.py             — Streamlit UI (calls API; auto-fallback to local model)
docker-compose.yml — one-command local deployment
tests/             — pytest smoke tests for all endpoints
\`\`\`

---

## Quick Start

**Local:**
\`\`\`bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
streamlit run app.py
\`\`\`

**Docker:**
\`\`\`bash
docker compose up --build
\`\`\`

---

## API

### POST /predict
\`\`\`bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "annual_income": 40000, "loan_amount": 12000}'
\`\`\`

### POST /explain
Returns top positive and negative SHAP factors per underwriting decision.

### GET /health
\`\`\`json
{"status": "ok", "model_version": "1.0"}
\`\`\`

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| \`dti_ratio\` | Loan amount / (installment rate × income estimate) |
| \`credit_history_length\` | Estimated credit age from applicant age and loan duration |
| \`number_of_delinquencies\` | Binary flag derived from credit history text |
| \`employment_length\` | Ordinal encoding of employment duration bands |

---

## Tests

\`\`\`bash
pytest tests/
\`\`\`

---

## Stack

Python · FastAPI · scikit-learn · SHAP · Streamlit · Docker · pytest · pandas · NumPy
