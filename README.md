# 🏦 Deployable Fintech Credit Risk Demo

This demo now uses the **real German Credit dataset** (UCI/OpenML mirror) instead of a synthetic toy sample.

## What’s included
- **FastAPI scoring service** (`/predict`, `/health`, `/metrics`)
- **Streamlit underwriting UI** with API-first scoring and local fallback
- **Docker Compose** deployment for end-to-end demo startup
- **Automated API smoke tests**

## Dataset
- Source: German Credit data (`GermanCredit.csv` mirror)
- Loader behavior:
  1. Use local cache at `data/GermanCredit.csv` if available
  2. Otherwise download from: `https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv`

## Model
- Algorithm: Logistic Regression
- Features used:
  - `duration`
  - `amount`
  - `age`
  - `installment_rate`
  - `number_credits`
  - `people_liable`

## Model metrics
Metrics are computed on an 80/20 stratified split and exposed from the running service:
- **AUC-ROC**: available via `GET /metrics`
- **Gini coefficient**: `2 * AUC - 1`, available via `GET /metrics`

You can retrieve them directly:

```bash
curl http://localhost:8000/metrics
```

## Quick Start (Local Python)

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
# in another terminal
streamlit run app.py
```

- API docs: `http://localhost:8000/docs`
- Front-end: `http://localhost:8501`

## Quick Start (Docker Compose)

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Front-end: `http://localhost:8501`

## Example prediction request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 24,
    "amount": 5000,
    "age": 35,
    "installment_rate": 2,
    "number_credits": 1,
    "people_liable": 1
  }'
```

## Tests

```bash
pytest
```
