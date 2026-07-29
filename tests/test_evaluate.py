from pathlib import Path

from src.evaluate import load_all_metrics


def test_load_all_metrics_standardizes_column_names():
    metrics_path = Path(__file__).resolve().parents[1] / "reports" / "model_metrics.csv"

    metrics_df = load_all_metrics(str(metrics_path))

    assert "model" in metrics_df.columns
    assert "RMSE" in metrics_df.columns
    assert "MAE" in metrics_df.columns
