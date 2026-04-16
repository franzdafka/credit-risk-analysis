from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_decision() -> None:
    payload = {"age": 34, "annual_income": 47000, "loan_amount": 9500}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability_default"] <= 1
    assert body["decision"] in {"approve", "reject"}
    assert body["risk_band"] in {"low", "medium", "high"}
