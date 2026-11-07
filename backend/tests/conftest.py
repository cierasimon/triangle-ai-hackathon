import os
from pathlib import Path

import pytest

# tests never read the real .env (no key, no LLM calls)
os.environ["PAPERPHYSICS_NO_DOTENV"] = "1"
os.environ["LLM_PROVIDER"] = "none"

from fastapi.testclient import TestClient  # noqa: E402

from app import store  # noqa: E402
from app.main import app  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


@pytest.fixture
def client():
    store.reset()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def synthetic_images() -> list[Path]:
    images = sorted((FIXTURES / "synthetic").glob("*.png"))
    assert images, "fixtures/synthetic is empty"
    return images
