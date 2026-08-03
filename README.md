# Project Workflow: Used Car Price Prediction Enhancement

This document outlines the complete, step-by-step technical workflow used to enhance the **Used Car Price Prediction** project. 

The goal of this enhancement was to train and compare two separate machine learning models (Random Forest and XGBoost) and integrate them into a modern, highly interactive UI that explains its predictions using SHAP values.

---

## Phase 1: Model Engineering & Training (`src/train.py`)

1. **Pipeline Verification**: Validated the `preprocess_data` function in `src/preprocess.py` to ensure it correctly handled categorical one-hot encoding and numerical scaling. The final output of the preprocessor yielded exactly **61 features** for training.
2. **Algorithm Selection**:
   * **Random Forest Regressor**: Chosen for its robust, ensemble-based averaging approach (`n_estimators=100`), which prevents overfitting and provides stable, conservative price predictions.
   * **XGBoost Regressor**: Chosen for its sequential error-correction algorithm (`learning_rate=0.1`, `max_depth=6`). XGBoost is highly sensitive to complex patterns and often outperforms standard decision trees on varied datasets.
3. **Execution**: Created `src/train.py` to seamlessly load the raw dataset (`data/raw/used_car_sales.csv`), fit the preprocessor, train both models concurrently, and persist them as `random_forest.joblib` and `xgboost.joblib` in the `models/` directory. 

---

## Phase 2: Backend API Refactoring (`app/main.py`)

1. **Static File Serving**: Configured the FastAPI application to explicitly mount and serve the frontend UI by adding the `StaticFiles` middleware, allowing seamless browser access via `http://127.0.0.1:8000/app/static/index.html`.
2. **Resolving Input Shape Mismatches**: 
   * **The Issue**: The previous implementation artificially padded the incoming data with 4 extra one-hot encoded "dummy" columns (turning 61 features into 65), which caused the new models to instantly crash with a shape mismatch error upon prediction.
   * **The Fix**: Refactored the `_build_model_input` function to safely bypass the unnecessary `tree_preprocessor` padding, directly returning the raw 61-feature numpy array expected by the newly trained `.joblib` models.
3. **Graceful Defaults**: Handled missing features safely. Because the UI was later simplified, the backend was updated to automatically inject statistical medians (e.g., 5 seats, 4 doors) into the prediction payload if those specific fields were omitted.

---

## Phase 3: Frontend UI Modernization (`app/static/index.html`)

1. **Aesthetic Overhaul**: Completely redesigned the UI using a sleek, premium dark-mode aesthetic with interactive hover states, responsive CSS grids, and smooth micro-animations. 
2. **Dual-Model Inference**: Rewrote the javascript payload handler (`handlePrediction`) to execute two simultaneous asynchronous API requests (`Promise.all`)—one targeting Random Forest and one targeting XGBoost.
3. **SHAP Feature Visualization**: Engineered dynamic HTML progress bars (`.contrib-bar`) to visually explain the SHAP (SHapley Additive exPlanations) values returned by the backend. This allows users to easily see which features positively (`+`) or negatively (`-`) impacted the base price of the car.
4. **Form Optimization (UX Improvements)**:
   * **Decluttering**: Stripped out low-impact fields that only cluttered the UI (Seats, Doors, Location, Color, Car Type).
   * **Curated Dropdowns**: Replaced raw number inputs for **Year**, **Horsepower**, and **Mileage** with curated `<select>` dropdowns featuring real-world suggestions (e.g., *150,000 KM (Very High Mileage)*) to guide the user and prevent validation errors.
5. **Feature Explainability Detail**: Integrated clear, human-readable labels for the mathematical features that most heavily impact a car's price:
   * **Make & Model**: The base brand and specific model tier, which dictate the car's initial MSRP and general depreciation curve.
   * **Manufactured Year**: Represents the age of the vehicle. Newer cars hold a strong positive price impact due to less aging and modern technology.
   * **Mileage Impact**: Quantifies the wear-and-tear on the car. High mileage severely drops the predicted price below the average baseline.
   * **Fuel Type**: Highlights market demand differences between Petrol, Diesel, Hybrid, and Electric engines.
   * **Transmission**: Represents whether the car is Automatic or Manual, with automatics generally commanding a higher resale value in modern markets.
   * **Engine Power**: The horsepower output. Higher engine power typically correlates with luxury or sports packages, significantly boosting the resale price.

---

## Phase 4: Final Execution

1. **Environment Setup**: Activated the isolated Python virtual environment containing the necessary dependencies (`scikit-learn`, `xgboost`, `fastapi`, `shap`, etc.).
2. **Server Initialization**: Launched the ASGI server using Uvicorn (`uvicorn app.main:app --reload`), successfully exposing the `/predict` POST endpoint.
3. **Verification**: Executed live predictions via the browser to confirm that both models return distinct, mathematically sound prices and SHAP contributions based on their underlying architectural differences.

---

## Key Technologies Used

- **Machine Learning**: `scikit-learn` (Random Forest, ColumnTransformer), `xgboost` (XGBRegressor)
- **Model Explainability**: `shap` (SHapley Additive exPlanations)
- **Backend API**: `FastAPI`, `Uvicorn`, `Pydantic`
- **Frontend**: Vanilla HTML5, CSS3, and JavaScript (No external frameworks for maximum speed and simplicity)

---

## Quick Start Guide

To run this project locally, follow these steps in your terminal:

1. **Activate the Virtual Environment**:
   ```bash
   venv\Scripts\activate
   ```
2. **Start the FastAPI Server**:
   ```bash
   uvicorn app.main:app --reload
   ```
3. **Access the Application**:
   Open your browser and navigate to: `http://127.0.0.1:8000/app/static/index.html`

---

## Future Enhancement Opportunities

- **Cloud Deployment**: Containerize the application using Docker and deploy to a cloud provider (e.g., AWS, Render, Heroku) to make the API publicly accessible.
- **Database Integration**: Connect a PostgreSQL or SQLite database to record user prediction requests and monitor model accuracy over time.
- **Expanded Dataset**: Acquire more recent car sales data (2024-2025) to further improve model accuracy and support modern EV (Electric Vehicle) brands like Tesla or Rivian.
