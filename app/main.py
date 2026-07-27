from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(
    title="Used Car Price Prediction API",
    description="Predicts the resale price of a used car from its features.",
    version="1.0.0",
)

# Allow the Next.js dev server (and any deployed frontend URL you add) to
# call this API from the browser. Without this, fetch() from the frontend
# is blocked by CORS and silently fails — the UI would only ever show its
# placeholder fallback, never a real prediction.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # TODO: add your deployed frontend URL here too, e.g.
        # "https://your-app.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Use the model exported from the Random Forest notebook.
DEFAULT_MODEL_NAME = "random_forest"

_loaded_model = None


def get_model():
    """Lazy-load the model once, on first request, instead of at import time."""
    global _loaded_model
    if _loaded_model is None:
        model_path = MODELS_DIR / f"{DEFAULT_MODEL_NAME}.joblib"
        if not model_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model file not found at {model_path}. "
                       f"Run the corresponding model notebook first to generate it.",
            )
        _loaded_model = joblib.load(model_path)
    return _loaded_model


class CarFeatures(BaseModel):
    """
    Raw feature schema that mirrors the car dataset used in preprocessing.
    """
    model_config = ConfigDict(populate_by_name=True)

    year: int = Field(..., alias="Manufactured Year", ge=1980, le=2026, example=2018)
    mileage: float = Field(..., alias="Mileage-KM", ge=0, example=45000)
    horsepower: float = Field(..., alias="Engine Power-HP", ge=0, example=150)
    number_of_seats: int = Field(..., alias="Number of Seats", ge=1, example=5)
    number_of_doors: int = Field(..., alias="Number of Doors", ge=1, example=4)
    brand: str = Field(..., alias="Manufacturer Name", example="Toyota")
    model: str = Field(..., alias="Car Name", example="Camry")
    fuel_type: str = Field(..., alias="Energy", example="Petrol")
    transmission: str = Field(..., alias="Gearbox", example="Automatic")
    location: str = Field(..., alias="Location", example="Dubai")
    car_type: str = Field(..., alias="Car Type", example="Sedan")
    color: str = Field(..., alias="Color", example="White")
    sale_status: str = Field(..., alias="Car Sale Status", example="Used")


class PredictionResponse(BaseModel):
    predicted_price: float
    model_used: str


@app.get("/")
def root():
    """Simple health check / landing message."""
    return {
        "message": "Used Car Price Prediction API is running.",
        "docs": "Visit /docs for the interactive Swagger UI.",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(car: CarFeatures):
    """
    Predict the price of a used car.

    NOTE: This endpoint currently expects the same raw fields used by the
    preprocessing notebook. The loaded model must be trained with the same
    feature pipeline or wrapped in a saved preprocessing pipeline.
    """
    model = get_model()

    input_df = pd.DataFrame([car.model_dump(by_alias=True)])

    try:
        prediction = model.predict(input_df)[0]
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed — input likely needs preprocessing "
                   f"to match training format. Error: {e}",
        )

    return PredictionResponse(predicted_price=round(float(prediction), 2), model_used=DEFAULT_MODEL_NAME)
