from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.preprocess import preprocess_data

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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "used_car_sales.csv"

# Use the model exported from the Random Forest notebook.
DEFAULT_MODEL_NAME = "random_forest"
TREE_CATEGORICAL_COLUMNS = ["Manufacturer Name", "Car Type", "Energy", "Gearbox"]

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
    Frontend-friendly payload that can be expanded into the model's raw schema.
    """
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    make: str = Field(..., validation_alias=AliasChoices("make", "brand"), example="Toyota")
    model: str = Field(..., example="Camry")
    year: int = Field(..., ge=1980, le=2026, example=2018)
    mileage: float = Field(..., ge=0, example=45000)
    fuel_type: str = Field(..., validation_alias=AliasChoices("fuel", "fuel_type"), example="Petrol")
    transmission: str = Field(..., example="Automatic")
    horsepower: float | None = Field(None, ge=0, validation_alias=AliasChoices("horsepower", "engine_size"), example=150)
    number_of_seats: int | None = Field(None, ge=1, example=5)
    number_of_doors: int | None = Field(None, ge=1, example=4)
    location: str | None = Field(None, example="Dubai")
    car_type: str | None = Field(None, example="Sedan")
    color: str | None = Field(None, example="White")
    sale_status: str | None = Field(None, example="Used")


class PredictionResponse(BaseModel):
    predicted_price: float
    model_used: str


@lru_cache(maxsize=1)
def get_preprocessors():
    """Load and cache the raw-data preprocessor and the tree-model preprocessor."""
    raw_df = pd.read_csv(RAW_DATA_PATH)
    _, _, _, _, raw_preprocessor = preprocess_data(raw_df)

    processed_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    processed_columns = list(processed_train.columns)

    tree_training_frame = processed_train.copy()
    for column in TREE_CATEGORICAL_COLUMNS:
        tree_training_frame[column] = "NA"

    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), TREE_CATEGORICAL_COLUMNS)
        ],
        remainder="passthrough",
    )
    tree_preprocessor.fit(tree_training_frame)

    normalized_df = raw_df.rename(columns={
        "Price-$": "price",
        "Manufactured Year": "year",
        "Mileage-KM": "mileage",
        "Engine Power-HP": "horsepower",
        "Manufacturer Name": "brand",
        "Car Name": "model",
        "Energy": "fuel_type",
        "Gearbox": "transmission",
        "Number of Seats": "number_of_seats",
        "Number of Doors": "number_of_doors",
        "Car Type": "car_type",
        "Car Sale Status": "sale_status",
    })

    defaults = {
        "horsepower": int(round(normalized_df["horsepower"].median())),
        "number_of_seats": int(round(normalized_df["number_of_seats"].median())),
        "number_of_doors": int(round(normalized_df["number_of_doors"].median())),
        "Location": str(normalized_df["Location"].mode().iat[0]),
        "car_type": str(normalized_df["car_type"].mode().iat[0]),
        "Color": str(normalized_df["Color"].mode().iat[0]),
        "sale_status": str(normalized_df["sale_status"].mode().iat[0]),
    }

    brand_model_defaults = (
        normalized_df.groupby(["brand", "model"])[["horsepower", "number_of_seats", "number_of_doors"]]
        .median()
        .reset_index()
    )
    brand_defaults = (
        normalized_df.groupby("brand")[["horsepower", "number_of_seats", "number_of_doors"]]
        .median()
        .reset_index()
    )

    return {
        "raw_preprocessor": raw_preprocessor,
        "tree_preprocessor": tree_preprocessor,
        "processed_columns": processed_columns,
        "defaults": defaults,
        "brand_model_defaults": brand_model_defaults,
        "brand_defaults": brand_defaults,
    }


def _pick_default(frame: pd.DataFrame, filter_columns: list[str], filter_values: list[str], value_column: str):
    matches = frame
    for column, value in zip(filter_columns, filter_values):
        matches = matches[matches[column].astype(str).str.lower() == str(value).lower()]
    if not matches.empty:
        return matches[value_column].iat[0]
    return None


def _build_model_input(car: CarFeatures):
    artifacts = get_preprocessors()
    defaults = artifacts["defaults"]

    horsepower = car.horsepower
    if horsepower is None:
        horsepower = _pick_default(
            artifacts["brand_model_defaults"],
            ["brand", "model"],
            [car.make, car.model],
            "horsepower",
        )
    if horsepower is None:
        horsepower = _pick_default(
            artifacts["brand_defaults"],
            ["brand"],
            [car.make],
            "horsepower",
        )
    if horsepower is None:
        horsepower = defaults["horsepower"]

    number_of_seats = car.number_of_seats
    if number_of_seats is None:
        number_of_seats = _pick_default(
            artifacts["brand_model_defaults"],
            ["brand", "model"],
            [car.make, car.model],
            "number_of_seats",
        )
    if number_of_seats is None:
        number_of_seats = defaults["number_of_seats"]

    number_of_doors = car.number_of_doors
    if number_of_doors is None:
        number_of_doors = _pick_default(
            artifacts["brand_model_defaults"],
            ["brand", "model"],
            [car.make, car.model],
            "number_of_doors",
        )
    if number_of_doors is None:
        number_of_doors = defaults["number_of_doors"]

    raw_row = pd.DataFrame([
        {
            "year": car.year,
            "mileage": car.mileage,
            "horsepower": float(horsepower),
            "number_of_seats": int(number_of_seats),
            "number_of_doors": int(number_of_doors),
            "brand": car.make,
            "model": car.model,
            "fuel_type": car.fuel_type,
            "transmission": car.transmission,
            "Location": car.location or defaults["Location"],
            "car_type": car.car_type or defaults["car_type"],
            "Color": car.color or defaults["Color"],
            "sale_status": car.sale_status or defaults["sale_status"],
        }
    ])

    raw_features = artifacts["raw_preprocessor"].transform(raw_row)
    if hasattr(raw_features, "toarray"):
        raw_features = raw_features.toarray()

    processed_row = pd.DataFrame(raw_features, columns=artifacts["processed_columns"])
    for column in TREE_CATEGORICAL_COLUMNS:
        processed_row[column] = "NA"

    return artifacts["tree_preprocessor"].transform(processed_row)


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
    input_df = _build_model_input(car)

    try:
        prediction = model.predict(input_df)[0]
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed while transforming the request into the trained model format. Error: {e}",
        )

    return PredictionResponse(predicted_price=round(float(prediction), 2), model_used=DEFAULT_MODEL_NAME)
