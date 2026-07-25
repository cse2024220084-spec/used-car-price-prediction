import matplotlib.pyplot as plt
import joblib

def save_model(model, name: str, folder: str = "../models"):
    """
    Save a trained model to disk with joblib so it can be loaded later by
    the FastAPI app without retraining. `name` should be simple, e.g.
    'random_forest' -> saved as ../models/random_forest.joblib
    """
    path = f"{folder}/{name}.joblib"
    joblib.dump(model, path)
    print(f"Saved model: {path}")
    return path

def set_plot_style():
    """Apply a consistent plotting style across all notebooks."""
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3