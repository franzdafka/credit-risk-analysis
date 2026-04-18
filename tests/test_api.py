from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _base_payload() -> dict:
    return {
        "duration": 24,
        "amount": 5000,
        "income": 4200,
        "age": 35,
        "installment_rate": 2,
        "number_credits": 1,
        "people_liable": 1,
        "purpose": "car",
        "credit_history": "existing paid",
        "employment_duration": "1 <= ... < 4 yrs",
    }


# Smoke tests

def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_predict_returns_decision() -> None:
    response = client.post("/predict", json=_base_payload())
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability_default"] <= 1
    assert body["decision"] in {"approve", "reject"}
    assert body["risk_band"] in {"low", "medium", "high"}


def test_explain_returns_top_3_positive_and_negative_features() -> None:
    response = client.get("/explain", params={"user_id": 0})
    assert response.status_code == 200
    body = response.json()
    assert len(body["top_positive"]) == 3
    assert len(body["top_negative"]) == 3


# Boundary tests

def test_predict_handles_extreme_values() -> None:
    payload = _base_payload()
    payload["income"] = 1
    payload["amount"] = 10_000_000
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    probability_default = response.json()["probability_default"]
    assert 0 <= probability_default <= 1


# Invariance / monotonicity test

def test_income_monotonicity_generally_decreases_risk() -> None:
    base_payload = _base_payload()

    low_income_payload = dict(base_payload)
    high_income_payload = dict(base_payload)
    low_income_payload["income"] = 1500
    high_income_payload["income"] = 9000

    low_income_response = client.post("/predict", json=low_income_payload)
    high_income_response = client.post("/predict", json=high_income_payload)

    assert low_income_response.status_code == 200
    assert high_income_response.status_code == 200

    low_risk = low_income_response.json()["probability_default"]
    high_risk = high_income_response.json()["probability_default"]

    assert high_risk <= low_risk
