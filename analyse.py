import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from credit_model import FEATURE_COLUMNS, REMOTE_DATASET_URL


def main() -> None:
    df = pd.read_csv(REMOTE_DATASET_URL)
    X = df[FEATURE_COLUMNS]
    y = df["credit_risk"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    gini = (2 * auc) - 1

    print("=== German Credit Baseline ===")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Gini coefficient: {gini:.4f}")


if __name__ == "__main__":
    main()
