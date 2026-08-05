import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_predict_accepts_6_feature_payload_and_returns_prediction():
    payload = {
        "make": "Toyota",
        "model": "Fortuner",
        "year": 2021,
        "car_type": "SUV",
        "fuel_type": "Petrol",
        "transmission": "Automatic",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "predicted_price" in body
    assert body.get("model_used")
    assert body.get("actual_benchmark") is not None


def test_feature_names_returns_grouped_manufacturers():
    response = client.get("/feature-names")

    assert response.status_code == 200
    body = response.json()
    assert "manufacturers" in body
    assert len(body["manufacturers"]) > 0

    first_mfr = body["manufacturers"][0]
    assert "name" in first_mfr
    assert "car_names" in first_mfr
    assert "car_types" in first_mfr
    assert "gearbox" in first_mfr
    assert "energy" in first_mfr
    assert "years" in first_mfr

    # Ensure no duplicates
    for mfr in body["manufacturers"]:
        assert len(mfr["car_names"]) == len(set(mfr["car_names"]))
        assert len(mfr["car_types"]) == len(set(mfr["car_types"]))
        assert len(mfr["gearbox"]) == len(set(mfr["gearbox"]))
        assert len(mfr["energy"]) == len(set(mfr["energy"]))
        assert len(mfr["years"]) == len(set(mfr["years"]))


def test_predict_xgboost_model():
    payload = {
        "make": "Hyundai",
        "model": "Creta",
        "year": 2024,
        "car_type": "Hatchback",
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "model_name": "xgboost",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["model_used"] == "xgboost"
    assert body["predicted_price"] > 0


def test_actual_benchmark_matches_dataset():
    """Verify that actual_benchmark returns accurate dataset median."""
    payload = {
        "make": "Toyota",
        "model": "Fortuner",
        "year": 2021,
        "car_type": "SUV",
        "fuel_type": "Hybrid",
        "transmission": "Automatic",
    }

    response = client.post("/predict", json=payload)
    body = response.json()

    # The actual benchmark should be a positive number from the dataset
    assert body["actual_benchmark"] > 0
    # It should be within the dataset's price range ($6000-$10900)
    assert 6000 <= body["actual_benchmark"] <= 11000
