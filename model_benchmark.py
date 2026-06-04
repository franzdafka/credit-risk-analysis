"""
Model benchmarking — compares Logistic Regression, Random Forest,
Gradient Boosting, and XGBoost on the German Credit Dataset.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"

FEATURE_COLUMNS = [
    "duration", "amount", "age", "installment_rate", "number_credits",
    "people_liable", "purpose", "credit_history", "employment_duration",
    "dti_ratio", "credit_history_length", "number_of_delinquencies",
    "employment_length", "loan_purpose",
]

NUMERIC = [
    "duration", "amount", "age", "installment_rate", "number_credits",
    "people_liable", "dti_ratio", "credit_history_length",
    "number_of_delinquencies", "employment_length",
]

CATEGORICAL = ["purpose", "credit_history", "employment_duration", "loan_purpose"]


def feature_engineering(df):
    out = df.copy()
    out["loan_purpose"] = out["purpose"].astype(str).fillna("unknown")
    text = out["credit_history"].astype(str).str.lower()
    out["number_of_delinquencies"] = text.str.contains(
        "delay|critical|overdue", regex=True
    ).astype(int)
    mapping = {
        "unemployed": 0.0, "< 1 yr": 0.5,
        "1 <= ... < 4 yrs": 2.5, "4 <= ... < 7 yrs": 5.5, ">= 7 yrs": 8.0,
    }
    out["employment_length"] = out["employment_duration"].map(mapping).fillna(2.0)
    out["credit_history_length"] = (out["age"] - 18).clip(lower=0) + 0.25 * out["duration"]
    income_proxy = (out["installment_rate"] * 1100.0) + (out["employment_length"] * 700.0)
    out["dti_ratio"] = (out["amount"] / income_proxy.clip(lower=1.0)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)
    return out


def build_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])


MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=2500, class_weight="balanced", solver="liblinear"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, learning_rate=0.05, scale_pos_weight=2.3,
        random_state=42, eval_metric="logloss",
    ),
}


def run_benchmark():
    df = feature_engineering(pd.read_csv(DATA_URL))
    X = df[FEATURE_COLUMNS]
    y = df["credit_risk"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rows = []
    for name, clf in MODELS.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "AUC-ROC": round(roc_auc_score(y_test, prob), 3),
            "Gini": round(2 * roc_auc_score(y_test, prob) - 1, 3),
            "Precision": round(precision_score(y_test, pred), 3),
            "Recall": round(recall_score(y_test, pred), 3),
            "F1": round(f1_score(y_test, pred), 3),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(run_benchmark().to_string(index=False))
