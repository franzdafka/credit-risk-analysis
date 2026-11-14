"""Probability-of-default model for the Statlog (German Credit) dataset.

One pipeline is used everywhere (evaluation, training, API):

1. Hold out a stratified 20% test set. It is used once, for the final report.
2. Compare candidate models by repeated stratified 5-fold CV AUC on the
   training set only, and select the best mean CV AUC.
3. Calibrate the selected model (Platt / isotonic, chosen by CV Brier score),
   so that the output can be read as a probability of default.
4. Choose the approve / decline threshold from out-of-fold PDs on the
   training set, under an explicit cost matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "GermanCredit.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_ARTIFACT_PATH = ARTIFACT_DIR / "credit_risk_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Cost matrix (Statlog convention): approving a borrower who defaults costs 5,
# declining a borrower who would have repaid costs 1 (lost margin).
COST_BAD_APPROVED = 5.0
COST_GOOD_DECLINED = 1.0

NUMERIC_FEATURES = [
    "duration",
    "amount",
    "installment_rate",
    "present_residence",
    "age",
    "number_credits",
    "people_liable",
    "monthly_amount",
]
CATEGORICAL_FEATURES = [
    "status",
    "credit_history",
    "purpose",
    "savings",
    "employment_duration",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]
RAW_FEATURES = [c for c in NUMERIC_FEATURES if c != "monthly_amount"] + CATEGORICAL_FEATURES
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Rating grades on the calibrated PD (upper bounds, exclusive).
RATING_GRADES = [("A", 0.05), ("B", 0.10), ("C", 0.20), ("D", 0.35), ("E", 1.01)]


# ---------------------------------------------------------------- data


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the single engineered feature: amount repaid per month of the loan."""
    out = df.copy()
    out["monthly_amount"] = out["amount"] / out["duration"].clip(lower=1)
    return out


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return features and the default flag (1 = bad credit risk)."""
    df = pd.read_csv(DATA_PATH)
    df = add_features(df)
    y = (1 - df["credit_risk"]).astype(int).rename("default")
    return df[FEATURE_COLUMNS], y


def split(X: pd.DataFrame, y: pd.Series):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


# ---------------------------------------------------------------- models


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def candidate_models() -> dict[str, Pipeline]:
    classifiers = {
        "Logistic Regression": LogisticRegression(C=0.3, max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }
    return {name: Pipeline([("prep", build_preprocessor()), ("clf", clf)]) for name, clf in classifiers.items()}


def cross_validate_candidates(X_train, y_train) -> pd.DataFrame:
    """Repeated stratified 5-fold CV AUC on the training set only."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    rows = []
    for name, model in candidate_models().items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        rows.append({"model": name, "cv_auc_mean": scores.mean(), "cv_auc_std": scores.std()})
    return pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)


def choose_calibration(base: Pipeline, X_train, y_train) -> tuple[str, dict[str, float]]:
    """Pick the calibration method with the lowest out-of-fold Brier score."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    briers: dict[str, float] = {}
    oof = cross_val_predict(clone(base), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    briers["none"] = brier_score_loss(y_train, oof)
    for method in ("sigmoid", "isotonic"):
        model = CalibratedClassifierCV(clone(base), method=method, cv=5)
        oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        briers[method] = brier_score_loss(y_train, oof)
    return min(briers, key=briers.get), briers


def make_calibrated(base: Pipeline, method: str):
    if method == "none":
        return clone(base)
    return CalibratedClassifierCV(clone(base), method=method, cv=5)


# ---------------------------------------------------------------- decisions


def expected_cost(y_true, pd_hat, threshold: float) -> float:
    """Average cost per applicant when declining every applicant with PD >= threshold."""
    y_true = np.asarray(y_true)
    decline = np.asarray(pd_hat) >= threshold
    bad_approved = np.sum(~decline & (y_true == 1))
    good_declined = np.sum(decline & (y_true == 0))
    return (COST_BAD_APPROVED * bad_approved + COST_GOOD_DECLINED * good_declined) / len(y_true)


def bayes_threshold() -> float:
    """With calibrated PDs, decline when PD * C_bad > (1 - PD) * C_good."""
    return COST_GOOD_DECLINED / (COST_GOOD_DECLINED + COST_BAD_APPROVED)


def choose_threshold(y_train, oof_pd) -> float:
    """Cost-minimising threshold on out-of-fold training PDs (never on the test set)."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [expected_cost(y_train, oof_pd, t) for t in grid]
    return float(grid[int(np.argmin(costs))])


def rating_grade(pd_value: float) -> str:
    for grade, upper in RATING_GRADES:
        if pd_value < upper:
            return grade
    return RATING_GRADES[-1][0]


def ks_statistic(y_true, scores) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    from scipy.stats import ks_2samp

    return float(ks_2samp(scores[y_true == 1], scores[y_true == 0]).statistic)


# ---------------------------------------------------------------- training


@dataclass
class TrainingResult:
    cv_table: pd.DataFrame
    selected_model: str
    calibration: str
    calibration_briers: dict[str, float]
    threshold: float
    oof_pd: np.ndarray
    model: Any
    explain_model: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_metrics: dict[str, float] = field(default_factory=dict)


def train(verbose: bool = False) -> TrainingResult:
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)

    cv_table = cross_validate_candidates(X_train, y_train)
    selected = cv_table.loc[0, "model"]
    base = candidate_models()[selected]
    if verbose:
        print(cv_table.to_string(index=False))
        print(f"Selected by CV AUC: {selected}")

    method, briers = choose_calibration(base, X_train, y_train)

    oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1)
    oof_pd = cross_val_predict(
        make_calibrated(base, method), X_train, y_train, cv=oof_cv, method="predict_proba"
    )[:, 1]
    threshold = choose_threshold(y_train, oof_pd)

    model = make_calibrated(base, method).fit(X_train, y_train)
    explain_model = clone(base).fit(X_train, y_train)

    pd_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pd_test)
    decline = pd_test >= threshold
    test_metrics = {
        "auc_roc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_test, pd_test),
        "brier": float(brier_score_loss(y_test, pd_test)),
        "brier_base_rate": float(brier_score_loss(y_test, np.full(len(y_test), y_train.mean()))),
        "threshold": threshold,
        "approval_rate": float(1 - decline.mean()),
        "default_rate_approved": float(y_test[~decline].mean()),
        "cost_model": expected_cost(y_test, pd_test, threshold),
        "cost_approve_all": expected_cost(y_test, pd_test, 1.01),
        "cost_decline_all": expected_cost(y_test, pd_test, -0.01),
    }

    return TrainingResult(
        cv_table=cv_table,
        selected_model=selected,
        calibration=method,
        calibration_briers=briers,
        threshold=threshold,
        oof_pd=oof_pd,
        model=model,
        explain_model=explain_model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        test_metrics=test_metrics,
    )


def save_artifact(result: TrainingResult, version: str) -> Path:
    X_all = pd.concat([result.X_train, result.X_test]).sort_index()
    categories = {c: sorted(X_all[c].unique().tolist()) for c in CATEGORICAL_FEATURES}
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "version": version,
            "model": result.model,
            "explain_model": result.explain_model,
            "selected_model": result.selected_model,
            "calibration": result.calibration,
            "threshold": result.threshold,
            "test_metrics": result.test_metrics,
            "categories": categories,
            "background": result.X_train.sample(100, random_state=RANDOM_STATE),
            "reference_frame": X_all,
        },
        MODEL_ARTIFACT_PATH,
    )
    return MODEL_ARTIFACT_PATH


# ---------------------------------------------------------------- serving


class CategoryValidationError(ValueError):
    def __init__(self, field_name: str, value: str, allowed: list[str]):
        super().__init__(f"Unknown value {value!r} for {field_name}")
        self.field = field_name
        self.value = value
        self.allowed = allowed


_ARTIFACT: dict[str, Any] | None = None


def load_artifact() -> dict[str, Any]:
    global _ARTIFACT
    if _ARTIFACT is None:
        if not MODEL_ARTIFACT_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_ARTIFACT_PATH}. Run `python train_model.py` first."
            )
        _ARTIFACT = joblib.load(MODEL_ARTIFACT_PATH)
    return _ARTIFACT


def validate_categories(payload: dict[str, Any]) -> None:
    allowed = load_artifact()["categories"]
    for col in CATEGORICAL_FEATURES:
        if payload[col] not in allowed[col]:
            raise CategoryValidationError(col, str(payload[col]), allowed[col])


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    validate_categories(payload)
    art = load_artifact()
    frame = add_features(pd.DataFrame([payload]))[FEATURE_COLUMNS]
    pd_value = float(art["model"].predict_proba(frame)[0, 1])
    decline = pd_value >= art["threshold"]
    return {
        "probability_default": pd_value,
        "rating_grade": rating_grade(pd_value),
        "decision": "decline" if decline else "approve",
        "decision_threshold": art["threshold"],
    }


def readable_feature_name(raw: str) -> str:
    kind, _, name = raw.partition("__")
    if kind == "cat":
        for col in CATEGORICAL_FEATURES:
            if name.startswith(col + "_"):
                return f"{col} = {name[len(col) + 1:]}"
    return name


def _shap_values(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    import shap

    art = load_artifact()
    pipe: Pipeline = art["explain_model"]
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    x = prep.transform(frame)
    names = [readable_feature_name(n) for n in prep.get_feature_names_out()]
    if isinstance(clf, LogisticRegression):
        background = prep.transform(art["background"])
        explainer = shap.LinearExplainer(clf, background)
        values = explainer.shap_values(x)
    else:
        values = shap.TreeExplainer(clf).shap_values(x)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
    return np.asarray(values), names


def explain(user_id: int, top_k: int = 3) -> dict[str, list[dict[str, float | str]]]:
    """Top risk-increasing and risk-reducing SHAP contributions (log-odds of default)."""
    frame = load_artifact()["reference_frame"]
    if user_id not in frame.index:
        raise ValueError(f"Unknown user_id={user_id}")
    row = frame.loc[[user_id]]
    values, names = _shap_values(row)
    onehot = load_artifact()["explain_model"].named_steps["prep"].transform(row)[0]
    # A one-hot column that is 0 for this applicant contributes through its absence.
    names = [n.replace(" = ", " ≠ ") if " = " in n and onehot[i] == 0 else n for i, n in enumerate(names)]
    contrib = pd.DataFrame({"feature": names, "shap_value": values[0]})
    contrib = contrib[contrib["shap_value"].abs() > 1e-9]
    up = contrib.sort_values("shap_value", ascending=False).head(top_k)
    down = contrib.sort_values("shap_value").head(top_k)
    to_list = lambda d: [{"feature": r.feature, "shap_value": float(r.shap_value)} for r in d.itertuples()]
    return {"risk_increasing": to_list(up), "risk_reducing": to_list(down)}
