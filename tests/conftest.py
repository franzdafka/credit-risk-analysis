import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from credit_model import MODEL_ARTIFACT_PATH, train_and_serialize_model  # noqa: E402


def pytest_sessionstart(session) -> None:  # type: ignore[no-untyped-def]
    """Ensure a serialized model exists for API tests without committing binaries."""
    if not MODEL_ARTIFACT_PATH.exists():
        train_and_serialize_model(version="credit-risk-test-fixture")
