from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset(path="../data/raw/used_car_sales.csv"):
    df = pd.read_csv(path)
    return df


def preprocess_data(df, test_size=0.2, random_state=42):

    # --------------------------
    # Rename columns
    # --------------------------
    df = df.rename(columns={
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
        "Car Sale Status": "sale_status"
    })

    # --------------------------
    # Remove duplicate rows
    # --------------------------
    df = df.drop_duplicates()

    # --------------------------
    # Convert numeric columns
    # --------------------------
    numeric_cols = [
        "year",
        "mileage",
        "horsepower",
        "number_of_seats",
        "number_of_doors",
        "price"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------
    # Remove rows without target
    # --------------------------
    df = df.dropna(subset=["price"])

    # --------------------------
    # Features
    # --------------------------
    numerical_features = [
        "year",
        "mileage",
        "horsepower",
        "number_of_seats",
        "number_of_doors"
    ]

    categorical_features = [
        "brand",
        "model",
        "fuel_type",
        "transmission",
        "Location",
        "car_type",
        "Color",
        "sale_status"
    ]

    numerical_features = [c for c in numerical_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    X = df[numerical_features + categorical_features]
    y = df["price"]

    # --------------------------
    # Pipelines
    # --------------------------
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    X = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test, preprocessor


def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test,
    output_dir="../data/processed"
):
    """
    Save processed training and testing data to CSV files.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert sparse matrices to DataFrames
    if hasattr(X_train, "toarray"):
        X_train = pd.DataFrame(X_train.toarray())
        X_test = pd.DataFrame(X_test.toarray())
    else:
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

    y_train = pd.DataFrame(y_train, columns=["price"])
    y_test = pd.DataFrame(y_test, columns=["price"])

    # Save files
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print(f"Processed data saved to: {output_dir.resolve()}")