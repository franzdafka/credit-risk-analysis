from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_model import FEATURE_COLUMNS, load_modeling_frame

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    cat_cols = [c for c in x.columns if c not in num_cols]

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
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def evaluate_model(name: str, model, x_train, x_test, y_train, y_test) -> dict:
    model.fit(x_train, y_train)
    y_proba = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    gini = 2 * auc - 1

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"{name} AP={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({name})")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"pr_curve_{name.lower().replace(' ', '_')}.png", dpi=140)
    plt.close()

    return {"model": name, "auc_roc": auc, "pr_auc": pr_auc, "gini": gini}


def run_shap(best_model, x_test) -> None:
    pre = best_model.named_steps["preprocessor"]
    clf = best_model.named_steps["classifier"]

    xt = pre.transform(x_test)
    feature_names = pre.get_feature_names_out()
    xt_df = pd.DataFrame(xt.toarray() if hasattr(xt, "toarray") else xt, columns=feature_names)

    sample = xt_df.sample(min(200, len(xt_df)), random_state=42)

    explainer = shap.Explainer(clf, sample)
    shap_values = explainer(sample)

    shap.plots.beeswarm(shap_values, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary_beeswarm.png", dpi=140)
    plt.close()

    shap.plots.waterfall(shap_values[0], show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_waterfall_first_prediction.png", dpi=140)
    plt.close()


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_modeling_frame()
    x = df[FEATURE_COLUMNS]
    y = df["credit_risk"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(x)

    models = [
        (
            "Logistic Regression",
            Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
                    ),
                ]
            ),
        ),
        (
            "Random Forest",
            Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=8,
                            min_samples_leaf=4,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
    ]

    if HAS_XGBOOST:
        models.append(
            (
                "XGBoost",
                Pipeline(
                    [
                        ("preprocessor", preprocessor),
                        (
                            "classifier",
                            XGBClassifier(
                                n_estimators=300,
                                max_depth=4,
                                learning_rate=0.05,
                                subsample=0.9,
                                colsample_bytree=0.9,
                                eval_metric="logloss",
                                random_state=42,
                            ),
                        ),
                    ]
                ),
            )
        )
    else:
        models.append(
            (
                "Gradient Boosting (XGBoost fallback)",
                Pipeline(
                    [
                        ("preprocessor", preprocessor),
                        ("classifier", GradientBoostingClassifier(random_state=42)),
                    ]
                ),
            )
        )

    results = [evaluate_model(name, model, x_train, x_test, y_train, y_test) for name, model in models]
    results_df = pd.DataFrame(results).sort_values("auc_roc", ascending=False)
    results_df.to_csv(REPORT_DIR / "benchmark_results.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = dict(models)[best_name]
    best_model.fit(x_train, y_train)
    run_shap(best_model, x_test)

    print("Saved benchmark report to reports/benchmark_results.csv")


if __name__ == "__main__":
    main()
