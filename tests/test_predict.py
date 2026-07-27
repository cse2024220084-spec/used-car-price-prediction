import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_predict_accepts_frontend_payload_and_returns_prediction():
    payload = {
        "year": 2019,
        "mileage": 45000,
        "horsepower": 150,
        "number_of_seats": 5,
        "number_of_doors": 4,
        "brand": "Toyota",
        "model": "Camry",
        "fuel_type": "Petrol",
        "transmission": "Automatic",
        "location": "Dubai",
        "car_type": "Sedan",
        "color": "White",
        "sale_status": "Used",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "predicted_price" in body or "prediction" in body
    assert body.get("model_used")
