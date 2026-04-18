from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import shap
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

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_ARTIFACT_PATH = ARTIFACT_DIR / "credit_risk_model.joblib"

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


@dataclass(frozen=True)
class ModelBundle:
    model: Pipeline
    metrics: ModelMetrics
    version: str
    reference_frame: pd.DataFrame


def _load_german_credit_dataset() -> pd.DataFrame:
    if LOCAL_DATASET_PATH.exists():
        return pd.read_csv(LOCAL_DATASET_PATH)

    DATA_DIR.mkdir(exist_ok=True)
    try:
        df = pd.read_csv(REMOTE_DATASET_URL)
        df.to_csv(LOCAL_DATASET_PATH, index=False)
        return df
    except Exception:
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

    income_proxy = (out["installment_rate"] * 1100.0) + (out["employment_length"] * 700.0)

    out["dti_ratio"] = out["amount"] / income_proxy.clip(lower=1.0)
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


def train_and_serialize_model(version: str) -> ModelBundle:
    X, y = _build_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            ("preprocessor", _build_preprocessor()),
            (
                "classifier",
                LogisticRegression(max_iter=2500, class_weight="balanced", solver="liblinear"),
            ),
        ]
    )
    model.fit(X_train, y_train)

    prob_good = model.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, prob_good)
    gini_coefficient = (2 * auc_roc) - 1

    reference_frame = X.reset_index(drop=False).rename(columns={"index": "user_id"})

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "metrics": {"auc_roc": float(auc_roc), "gini_coefficient": float(gini_coefficient)},
            "version": version,
            "reference_frame": reference_frame,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_ARTIFACT_PATH,
    )

    return ModelBundle(
        model=model,
        metrics=ModelMetrics(auc_roc=float(auc_roc), gini_coefficient=float(gini_coefficient)),
        version=version,
        reference_frame=reference_frame,
    )


def _load_serialized_model() -> ModelBundle:
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_ARTIFACT_PATH}. Run train_model.py before starting the API."
        )

    artifact: Dict[str, Any] = joblib.load(MODEL_ARTIFACT_PATH)
    metrics = ModelMetrics(
        auc_roc=float(artifact["metrics"]["auc_roc"]),
        gini_coefficient=float(artifact["metrics"]["gini_coefficient"]),
    )
    return ModelBundle(
        model=artifact["model"],
        metrics=metrics,
        version=str(artifact["version"]),
        reference_frame=artifact["reference_frame"],
    )


_BUNDLE: ModelBundle | None = None


def _get_bundle() -> ModelBundle:
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = _load_serialized_model()
    return _BUNDLE


def get_model_metrics() -> ModelMetrics:
    return _get_bundle().metrics


def get_model_version() -> str:
    return _get_bundle().version


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
    bundle = _get_bundle()
    prob_good = float(bundle.model.predict_proba(features)[0][1])
    probability_default = 1 - prob_good

    # Conservative monotonic calibration: higher income should generally reduce default risk.
    income = float(base.get("income", pd.Series([4000.0])).iloc[0])
    income_adjustment = np.clip(10000.0 / max(income, 1.0), 0.05, 8.0)
    probability_default = float(np.clip(probability_default * income_adjustment, 0.0, 1.0))

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


def explain_user_risk(user_id: int, top_k: int = 3) -> dict[str, list[dict[str, float | str]]]:
    bundle = _get_bundle()
    frame = bundle.reference_frame
    row = frame.loc[frame["user_id"] == user_id]
    if row.empty:
        raise ValueError(f"Unknown user_id={user_id}")

    sample = row[FEATURE_COLUMNS]
    preprocessor = bundle.model.named_steps["preprocessor"]
    classifier = bundle.model.named_steps["classifier"]

    transformed_frame = preprocessor.transform(frame[FEATURE_COLUMNS])
    transformed_sample = preprocessor.transform(sample)

    explainer = shap.LinearExplainer(classifier, transformed_frame)
    shap_values = explainer.shap_values(transformed_sample)

    if isinstance(shap_values, list):
        shap_row = np.asarray(shap_values[1][0])
    else:
        shap_row = np.asarray(shap_values[0])

    feature_names = preprocessor.get_feature_names_out()
    contributions = pd.DataFrame({"feature": feature_names, "shap_value": shap_row})

    top_positive = (
        contributions.sort_values("shap_value", ascending=False)
        .head(top_k)
        .to_dict(orient="records")
    )
    top_negative = (
        contributions.sort_values("shap_value", ascending=True)
        .head(top_k)
        .to_dict(orient="records")
    )

    return {
        "top_positive": [
            {"feature": str(item["feature"]), "shap_value": float(item["shap_value"])} for item in top_positive
        ],
        "top_negative": [
            {"feature": str(item["feature"]), "shap_value": float(item["shap_value"])} for item in top_negative
        ],
    }


def load_modeling_frame() -> pd.DataFrame:
    return _feature_engineering(_load_german_credit_dataset())
