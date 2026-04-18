import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from credit_model import FEATURE_COLUMNS, load_modeling_frame
from sklearn.linear_model import LogisticRegression


def main() -> None:
    df = load_modeling_frame()
    x = df[FEATURE_COLUMNS]
    y = df["credit_risk"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    model.fit(pd.get_dummies(x_train), y_train)

    test_encoded = pd.get_dummies(x_test)
    test_encoded = test_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    y_pred = model.predict(test_encoded)
    y_proba = model.predict_proba(test_encoded)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    gini = (2 * auc) - 1

    print("=== German Credit Baseline (with feature engineering) ===")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Gini coefficient: {gini:.4f}")


if __name__ == "__main__":
    main()
