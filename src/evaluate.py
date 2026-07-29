from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    """Compute standard regression metrics for one model's predictions."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "model_name": model_name,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

def save_metrics(metrics: dict, file_path: str  = "../reports/model_metrics.csv"):
    """Save the metrics dictionary to a CSV file."""
    df_new = pd.DataFrame([metrics])
    try:
        df_existing = pd.read_csv(file_path)
        df_existing = df_existing[df_existing['model_name'] != metrics['model_name']]
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_all = df_new
    df_all.to_csv(file_path, index=False)
    return df_all


def load_all_metrics(file_path: str = "../reports/model_metrics.csv") -> pd.DataFrame:
    """Load all metrics and normalize column names for analysis notebooks."""
    project_root = Path(__file__).resolve().parents[1]
    default_path = project_root / "reports" / "model_metrics.csv"

    if file_path is None:
        resolved_path = default_path
    else:
        path = Path(file_path)
        resolved_path = path if path.is_absolute() else (project_root / path)

    try:
        df = pd.read_csv(resolved_path)
    except FileNotFoundError:
        print(f"No metrics file found at {resolved_path}.")
        return pd.DataFrame(columns=["model", "MSE", "RMSE", "MAE", "R2"])

    rename_map = {}
    if "model_name" in df.columns:
        rename_map["model_name"] = "model"
    if "model" in df.columns:
        rename_map["model"] = "model"
    if "mse" in df.columns:
        rename_map["mse"] = "MSE"
    if "rmse" in df.columns:
        rename_map["rmse"] = "RMSE"
    if "mae" in df.columns:
        rename_map["mae"] = "MAE"
    if "r2" in df.columns:
        rename_map["r2"] = "R2"

    return df.rename(columns=rename_map)
