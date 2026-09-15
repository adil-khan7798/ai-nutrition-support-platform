"""Shared fixtures for the intake_api test suite.

The intake_api module keeps global state (`food_service` and `store`) that is
populated from environment variables at import time. To get isolated state per
test we point FOOD_CSV at a tiny fixture file and reload the module, then drive
the app through FastAPI's TestClient (which also runs the lifespan startup, so
`food_service.load()` happens exactly as it does in production).
"""

import importlib
import sys

import pytest
from fastapi.testclient import TestClient

# A minimal stand-in for the real USDA food.csv. Column names MUST match the
# values in intake_api.FOOD_COLUMN_MAP. Per-100g values are chosen to make the
# grams-scaling arithmetic easy to verify by hand.
#   apple raw  -> protein 0.3, carbs 14, sugar 10, fiber 2.4, fat 0.2,
#                 sat fat 0.03, cholesterol 0, sodium 1   (per 100g)
#   beef patty -> protein 26, carbs 0, sugar 0, fiber 0, fat 20,
#                 sat fat 8, cholesterol 90, sodium 75    (per 100g)
FIXTURE_CSV = (
    "Category,Description,Data.Protein,Data.Carbohydrate,Data.Sugar Total,"
    "Data.Fiber,Data.Fat.Total Lipid,Data.Fat.Saturated Fat,"
    "Data.Cholesterol,Data.Major Minerals.Sodium\n"
    "Fruits,Apple raw,0.3,14,10,2.4,0.2,0.03,0,1\n"
    "Meats,Beef patty cooked,26,0,0,0,20,8,90,75\n"
)


@pytest.fixture
def intake_api(tmp_path, monkeypatch):
    """Reload intake_api with a fixture CSV and a fresh in-memory store."""
    csv_path = tmp_path / "food.csv"
    csv_path.write_text(FIXTURE_CSV)
    monkeypatch.setenv("FOOD_CSV", str(csv_path))
    # Keep the ML URL at a deterministic value so respx routes can match it.
    monkeypatch.setenv("ML_API_URL", "http://ml-api.test")

    # Drop any previously imported copy so module-level globals re-initialise.
    sys.modules.pop("intake_api", None)
    module = importlib.import_module("intake_api")
    return module


@pytest.fixture
def food_service(intake_api):
    """A loaded FoodService backed by the fixture CSV (no app/HTTP needed)."""
    intake_api.food_service.load()
    return intake_api.food_service


@pytest.fixture
def client(intake_api):
    """TestClient with lifespan run (loads the fixture food dataset)."""
    with TestClient(intake_api.app) as c:
        yield c


@pytest.fixture
def auth_headers(client):
    """Bearer token for the built-in admin account, valid for one test."""
    resp = client.post(
        "/v1/login", json={"username": "admin", "password": "admin-password"}
    )
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['token']}"}


@pytest.fixture
def valid_profile():
    """A profile that satisfies every UserProfile constraint."""
    return {
        "name": "Test User",
        "age": 40,
        "gender": 1,
        "race_ethnicity": 3,
        "education_level": 3,
        "weight_kg": 80.0,
        "height_cm": 175.0,
    }
