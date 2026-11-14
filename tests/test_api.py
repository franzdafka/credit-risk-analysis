import pandas as pd
from fastapi.testclient import TestClient

import credit_model as cm
from api import app

client = TestClient(app)


def _payload(row: int = 0) -> dict:
    df = pd.read_csv(cm.DATA_PATH)
    record = df.loc[row, cm.RAW_FEATURES].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()}


def test_health() -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_categories_cover_all_categorical_fields() -> None:
    body = client.get("/categories").json()
    assert set(body) == set(cm.CATEGORICAL_FEATURES)


def test_predict_returns_valid_pd_grade_and_decision() -> None:
    response = client.post("/predict", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability_default"] <= 1.0
    assert body["rating_grade"] == cm.rating_grade(body["probability_default"])
    expected = "decline" if body["probability_default"] >= body["decision_threshold"] else "approve"
    assert body["decision"] == expected


def test_predict_matches_model_directly() -> None:
    payload = _payload(5)
    api_pd = client.post("/predict", json=payload).json()["probability_default"]
    frame = cm.add_features(pd.DataFrame([payload]))[cm.FEATURE_COLUMNS]
    direct_pd = cm.load_artifact()["model"].predict_proba(frame)[0, 1]
    assert abs(api_pd - direct_pd) < 1e-9


def test_unknown_category_returns_422_with_allowed_values() -> None:
    payload = _payload()
    payload["purpose"] = "spaceship"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["field"] == "purpose"
    assert "business" in detail["allowed"]


def test_out_of_range_numeric_is_rejected() -> None:
    payload = _payload()
    payload["installment_rate"] = 9
    assert client.post("/predict", json=payload).status_code == 422


def test_explain_returns_three_drivers_each_side() -> None:
    body = client.get("/explain", params={"user_id": 0}).json()
    assert len(body["risk_increasing"]) == 3
    assert len(body["risk_reducing"]) == 3
    assert all(d["shap_value"] > 0 for d in body["risk_increasing"])
    assert all(d["shap_value"] < 0 for d in body["risk_reducing"])


def test_explain_unknown_user_is_404() -> None:
    assert client.get("/explain", params={"user_id": 999_999}).status_code == 404
