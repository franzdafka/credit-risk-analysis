from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "data"
LOCAL_DATASET_PATH = DATA_DIR / "GermanCredit.csv"
REMOTE_DATASET_URL = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
FEATURE_COLUMNS = [
    "duration",
    "amount",
    "age",
    "installment_rate",
    "number_credits",
    "people_liable",
]


@dataclass(frozen=True)
class RiskPrediction:
    probability_default: float
    predicted_default: bool
    risk_band: str


@dataclass(frozen=True)
class ModelMetrics:
    auc_roc: float
    gini_coefficient: float


def _load_german_credit_dataset() -> pd.DataFrame:
    """Load real German Credit data from local cache, falling back to source URL."""
    if LOCAL_DATASET_PATH.exists():
        return pd.read_csv(LOCAL_DATASET_PATH)

    DATA_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(REMOTE_DATASET_URL)
    df.to_csv(LOCAL_DATASET_PATH, index=False)
    return df


def _build_training_data() -> tuple[pd.DataFrame, pd.Series]:
    df = _load_german_credit_dataset()
    y = df["credit_risk"].astype(int)
    X = df[FEATURE_COLUMNS].copy()
    return X, y


def _train_model_and_metrics() -> tuple[LogisticRegression, ModelMetrics]:
    X, y = _build_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    prob_good = model.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, prob_good)
    gini_coefficient = (2 * auc_roc) - 1

    return model, ModelMetrics(auc_roc=auc_roc, gini_coefficient=gini_coefficient)


_MODEL, _METRICS = _train_model_and_metrics()


def get_model_metrics() -> ModelMetrics:
    return _METRICS


def predict_risk(payload: Dict[str, float]) -> RiskPrediction:
    features = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    prob_good = float(_MODEL.predict_proba(features)[0][1])
    probability_default = 1 - prob_good
    predicted_default = probability_default >= 0.5

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