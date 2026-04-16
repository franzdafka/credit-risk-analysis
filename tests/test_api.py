from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["auc_roc"] <= 1.0
    assert -1.0 <= body["gini_coefficient"] <= 1.0


def test_predict_returns_decision() -> None:
    payload = {
        "duration": 24,
        "amount": 5000,
        "age": 35,
        "installment_rate": 2,
        "number_credits": 1,
        "people_liable": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability_default"] <= 1
    assert body["decision"] in {"approve", "reject"}
    assert body["risk_band"] in {"low", "medium", "high"}