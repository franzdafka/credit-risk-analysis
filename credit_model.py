from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).parent / "data"
LOCAL_DATASET_PATH = DATA_DIR / "GermanCredit.csv"
REMOTE_DATASET_URL = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"

BASE_FEATURE_COLUMNS = [
    "duration",
    "amount",
    "age",
    "installment_rate",
    "number_credits",
    "people_liable",
    "purpose",
    "credit_history",
    "employment_duration",
]

ENGINEERED_FEATURE_COLUMNS = [
    "dti_ratio",
    "credit_history_length",
    "number_of_delinquencies",
    "employment_length",
    "loan_purpose",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS


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
    if LOCAL_DATASET_PATH.exists():
        return pd.read_csv(LOCAL_DATASET_PATH)

    DATA_DIR.mkdir(exist_ok=True)
    try:
        df = pd.read_csv(REMOTE_DATASET_URL)
        df.to_csv(LOCAL_DATASET_PATH, index=False)
        return df
    except Exception:
        # Offline-safe fallback with the original schema.
        rng = np.random.default_rng(42)
        n = 800
        duration = rng.integers(4, 72, n)
        amount = rng.integers(250, 20000, n)
        age = rng.integers(18, 75, n)
        installment_rate = rng.integers(1, 5, n)
        number_credits = rng.integers(1, 4, n)
        people_liable = rng.integers(1, 3, n)
        purpose = rng.choice(["car", "furniture/equipment", "business", "education"], n)
        credit_history = rng.choice(
            [
                "critical/other existing credit",
                "existing paid",
                "all paid",
                "delay in paying off in the past",
            ],
            n,
        )
        employment_duration = rng.choice(["< 1 yr", "1 <= ... < 4 yrs", "4 <= ... < 7 yrs", ">= 7 yrs"], n)

        risk_score = (
            0.00008 * amount
            + 0.02 * duration
            + 0.22 * installment_rate
            - 0.018 * age
            + 0.35 * (credit_history == "delay in paying off in the past").astype(int)
            + 0.15 * (purpose == "business").astype(int)
            + rng.normal(0, 0.3, n)
        )
        probability_good = 1 / (1 + np.exp(risk_score - 3.5))
        credit_risk = (probability_good > 0.5).astype(int)

        return pd.DataFrame(
            {
                "duration": duration,
                "amount": amount,
                "age": age,
                "installment_rate": installment_rate,
                "number_credits": number_credits,
                "people_liable": people_liable,
                "purpose": purpose,
                "credit_history": credit_history,
                "employment_duration": employment_duration,
                "credit_risk": credit_risk,
            }
        )


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["loan_purpose"] = out["purpose"].astype(str).fillna("unknown")

    credit_history_text = out["credit_history"].astype(str).str.lower()
    out["number_of_delinquencies"] = (
        credit_history_text.str.contains("delay|critical|overdue", regex=True).astype(int)
    )

    mapping = {
        "unemployed": 0.0,
        "< 1 yr": 0.5,
        "1 <= ... < 4 yrs": 2.5,
        "4 <= ... < 7 yrs": 5.5,
        ">= 7 yrs": 8.0,
    }
    out["employment_length"] = out["employment_duration"].map(mapping).fillna(2.0)

    out["credit_history_length"] = (out["age"] - 18).clip(lower=0) + 0.25 * out["duration"]

    income_proxy = (out["installment_rate"] * 250.0) + (out["employment_length"] * 120.0)
    out["dti_ratio"] = (out["amount"] / (income_proxy * out["duration"]))
    out["dti_ratio"] = out["dti_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)

    return out


def _build_training_data() -> tuple[pd.DataFrame, pd.Series]:
    df = _load_german_credit_dataset()
    df = _feature_engineering(df)
    y = df["credit_risk"].astype(int)
    X = df[FEATURE_COLUMNS].copy()
    return X, y


def _build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "duration",
        "amount",
        "age",
        "installment_rate",
        "number_credits",
        "people_liable",
        "dti_ratio",
        "credit_history_length",
        "number_of_delinquencies",
        "employment_length",
    ]
    categorical_features = ["purpose", "credit_history", "employment_duration", "loan_purpose"]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def _train_model_and_metrics() -> tuple[Pipeline, ModelMetrics]:
    X, y = _build_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            ("preprocessor", _build_preprocessor()),
            (
                "classifier",
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
            ),
        ]
    )
    model.fit(X_train, y_train)

    prob_good = model.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, prob_good)
    gini_coefficient = (2 * auc_roc) - 1

    return model, ModelMetrics(auc_roc=auc_roc, gini_coefficient=gini_coefficient)


_MODEL, _METRICS = _train_model_and_metrics()


def get_model_metrics() -> ModelMetrics:
    return _METRICS


def predict_risk(payload: Dict[str, float]) -> RiskPrediction:
    base = pd.DataFrame([payload])
    for col, default in {
        "purpose": "furniture/equipment",
        "credit_history": "existing paid",
        "employment_duration": "1 <= ... < 4 yrs",
    }.items():
        if col not in base:
            base[col] = default

    features = _feature_engineering(base)[FEATURE_COLUMNS]
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


def load_modeling_frame() -> pd.DataFrame:
    return _feature_engineering(_load_german_credit_dataset())
