from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class RiskPrediction:
    probability_default: float
    predicted_default: bool
    risk_band: str


def _training_dataframe() -> pd.DataFrame:
    data = {
        "age": [
            25,
            45,
            35,
            50,
            23,
            40,
            60,
            28,
            33,
            55,
            29,
            48,
            37,
            52,
            24,
            41,
            62,
            30,
            34,
            57,
        ],
        "annual_income": [
            25000,
            60000,
            40000,
            80000,
            20000,
            55000,
            90000,
            30000,
            45000,
            75000,
            27000,
            63000,
            42000,
            82000,
            22000,
            58000,
            95000,
            32000,
            47000,
            78000,
        ],
        "loan_amount": [
            5000,
            15000,
            8000,
            20000,
            3000,
            12000,
            25000,
            6000,
            9000,
            18000,
            5500,
            16000,
            8500,
            21000,
            3500,
            13000,
            27000,
            6500,
            9500,
            19000,
        ],
        "default": [
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
        ],
    }
    return pd.DataFrame(data)


def train_model() -> LogisticRegression:
    df = _training_dataframe()
    model = LogisticRegression(max_iter=1000)
    model.fit(df[["age", "annual_income", "loan_amount"]], df["default"])
    return model


_MODEL = train_model()


def predict_risk(payload: Dict[str, float]) -> RiskPrediction:
    features = pd.DataFrame([payload], columns=["age", "annual_income", "loan_amount"])
    probability_default = float(_MODEL.predict_proba(features)[0][1])
    predicted_default = bool(_MODEL.predict(features)[0])

    if probability_default > 0.6:
        risk_band = "high"
    elif probability_default > 0.3:
        risk_band = "medium"
    else:
        risk_band = "low"

    return RiskPrediction(
        probability_default=probability_default,
        predicted_default=predicted_default,
        risk_band=risk_band,
    )
