from __future__ import annotations

from datetime import datetime, timezone

from credit_model import MODEL_ARTIFACT_PATH, train_and_serialize_model


if __name__ == "__main__":
    version = datetime.now(timezone.utc).strftime("credit-risk-%Y%m%d%H%M%S")
    bundle = train_and_serialize_model(version=version)
    print(
        f"Saved model artifact to {MODEL_ARTIFACT_PATH} | "
        f"version={bundle.version} | auc={bundle.metrics.auc_roc:.4f}"
    )
