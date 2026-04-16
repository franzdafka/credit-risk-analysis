# 🏦 Deployable Fintech Credit Risk Demo

This project is now structured as a **deployable fintech demo** with:
- A **FastAPI scoring service** (`/predict`, `/health`)
- A **Streamlit front-end** for underwriting simulations
- **Docker Compose** setup for one-command local deployment
- Basic **API tests** for reliability

## Architecture

- `credit_model.py` — shared model training + risk inference logic
- `api.py` — REST API used by front-end or external systems
- `app.py` — Streamlit UI (calls API; auto-fallback to local model)
- `docker-compose.yml` — runs API + front-end together
- `tests/test_api.py` — smoke tests for health and predict endpoints

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

## Example API Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":35,"annual_income":40000,"loan_amount":12000}'
```

## Run tests

```bash
pytest
```

## Demo Positioning

This repository can be shown as an MVP for:
- Embedded lending decision support
- Risk scoring API prototyping
- Internal credit analyst dashboard demos
