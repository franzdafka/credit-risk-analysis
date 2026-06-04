# Credit Risk Scoring API

A production-ready credit scoring service built with FastAPI, scikit-learn, and SHAP — modelling the core underwriting decision logic used in consumer lending.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)
![CI](https://github.com/franzdafka/credit-risk-analysis/actions/workflows/ci.yml/badge.svg)

---

## Overview

Credit scoring pipeline trained on the German Credit Dataset (1,000 applicants), with predictions and SHAP-based explanations served via REST API.

In credit underwriting, a false negative — approving a borrower who defaults — carries asymmetric cost relative to a false positive. The model uses class_weight="balanced" and is evaluated primarily on Precision and AUC-ROC.

---

## Model Benchmarking

Four classifiers benchmarked on identical train/test splits (80/20, stratified):

| Model | AUC-ROC | Gini | Precision | Recall | F1 |
|-------|---------|------|-----------|--------|-----|
| Logistic Regression | 0.674 | 0.348 | 0.794 | 0.579 | 0.669 |
| Random Forest | 0.693 | 0.385 | 0.719 | 0.857 | 0.782 |
| Gradient Boosting | 0.750 | 0.500 | 0.744 | 0.829 | 0.784 |
| XGBoost | 0.730 | 0.460 | 0.756 | 0.886 | 0.816 |

Gradient Boosting achieves the best AUC (0.750) and Gini (0.500). XGBoost achieves the best F1 (0.816) and Recall. The production API uses Logistic Regression for interpretability; the full benchmark is available via model_benchmark.py.

### ROC Curve
![ROC Curve](roc_curve.png)

### SHAP Feature Importance
![SHAP Feature Importance](shap_importance.png)

Key risk drivers:
- Loan duration — longer terms significantly increase predicted default probability
- DTI ratio — debt-to-income proxy derived from installment rate and employment data
- Credit history — prior delinquencies are the strongest negative signal
- Loan amount — higher amounts correlate with elevated risk tier

---

## Architecture

credit_model.py    — model training + SHAP inference logic
model_benchmark.py — four-model comparison (LR, RF, GBM, XGBoost)
api.py             — FastAPI REST service (/predict, /explain, /health)
app.py             — Streamlit UI
docker-compose.yml — one-command local deployment
tests/             — pytest smoke tests

---

## Quick Start

pip install -r requirements.txt
uvicorn api:app --reload --port 8000
streamlit run app.py

Docker:
docker compose up --build

Run benchmark:
python model_benchmark.py

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| dti_ratio | Loan amount divided by installment rate times income estimate |
| credit_history_length | Estimated credit age from applicant age and loan duration |
| number_of_delinquencies | Binary flag derived from credit history text |
| employment_length | Ordinal encoding of employment duration bands |

---

## Tests

pytest tests/

---

## Stack

Python · FastAPI · scikit-learn · XGBoost · SHAP · Streamlit · Docker · pytest · pandas · NumPy
