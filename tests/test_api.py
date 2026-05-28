from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _base_payload() -> dict:
    return {
        "duration": 24,
        "amount": 5000,
        "age": 35,
        "installment_rate": 2,
        "number_credits": 1,
        "people_liable": 1,
        "purpose": "car (new)",
        "credit_history": "existing credits paid back duly till now",
        "employment_duration": "1 <= ... < 4 years",
    }


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
    assert "underwriting_decision" in body
    assert body["underwriting_decision"] in ("approve", "decline")
    assert "probability_default" in body
    assert "rating_grade" in body
    assert body["rating_grade"] in ("low", "medium", "high")


def test_explain_returns_top_3_positive_and_negative_features() -> None:
    response = client.get("/explain", params={"user_id": 0})
    assert response.status_code == 200
    body = response.json()
    assert len(body["top_positive"]) == 3
    assert len(body["top_negative"]) == 3


def test_predict_rejects_unknown_category_with_422() -> None:
    payload = _base_payload()
    payload["credit_history"] = "totally unknown category"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["detail"]["field"] == "credit_history"


def test_predict_accepts_known_alias_category_values() -> None:
    payload = _base_payload()
    payload["credit_history"] = "existing credits paid back duly till now"
    payload["employment_duration"] = "... >= 7 years"
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_handles_extreme_values() -> None:
    payload = _base_payload()
    payload["amount"] = 10_000_000
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_pd_is_valid_probability() -> None:
    response = client.post("/predict", json=_base_payload())
    assert response.status_code == 200
    pd = response.json()["probability_default"]
    assert 0.0 <= pd <= 1.0
