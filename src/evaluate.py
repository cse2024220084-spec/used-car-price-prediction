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


def load_all_matrics(file_path: str = "../reports/model_metrics.csv") -> pd.DataFrame:
    """Load all metrics from the CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"No metrics file found at {file_path}.")
        return pd.read_csv(file_path, index_col=0)  # Return an empty DataFrame if the file doesn't exist
