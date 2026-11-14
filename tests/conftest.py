import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import credit_model as cm  # noqa: E402


def pytest_sessionstart(session) -> None:  # type: ignore[no-untyped-def]
    """Train a model artifact once if none exists (it is not committed)."""
    if not cm.MODEL_ARTIFACT_PATH.exists():
        cm.save_artifact(cm.train(), version="credit-risk-test-fixture")
