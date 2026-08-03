import os
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.preprocess import load_dataset, preprocess_data, save_processed_data

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "used_car_sales.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

def main():
    print(f"Loading data from {RAW_DATA_PATH}...")
    df = load_dataset(path=str(RAW_DATA_PATH))
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    print("Saving processed data...")
    save_processed_data(X_train, X_test, y_train, y_test, output_dir=str(PROCESSED_DIR))
    
    # --------------------------
    # Train Random Forest
    # --------------------------
    print("\nTraining Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    rf_preds = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    import numpy as np
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    
    print(f"Random Forest Performance:")
    print(f"  MAE:  ${rf_mae:,.2f}")
    print(f"  RMSE: ${rf_rmse:,.2f}")
    
    # Ensure models dir exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    rf_model_path = MODELS_DIR / "random_forest.joblib"
    joblib.dump(rf_model, rf_model_path)
    print(f"Saved Random Forest model to {rf_model_path}")

    # --------------------------
    # Train XGBoost
    # --------------------------
    print("\nTraining XGBoost Regressor...")
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    xgb_preds = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    
    print(f"XGBoost Performance:")
    print(f"  MAE:  ${xgb_mae:,.2f}")
    print(f"  RMSE: ${xgb_rmse:,.2f}")
    
    xgb_model_path = MODELS_DIR / "xgboost.joblib"
    joblib.dump(xgb_model, xgb_model_path)
    print(f"Saved XGBoost model to {xgb_model_path}")
    
    print("\nTraining complete!")
    if rf_mae < xgb_mae:
        print("-> Random Forest has a lower MAE on the test set.")
    else:
        print("-> XGBoost has a lower MAE on the test set.")

if __name__ == "__main__":
    main()
