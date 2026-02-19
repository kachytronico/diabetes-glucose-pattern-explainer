"""Train a glucose prediction model."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from diabetes_explainer import data_schema, synth_data
from diabetes_explainer.features import build_features

MODEL_PATH = Path("models/glucose_model.pkl")
METRICS_PATH = Path("models/metrics.json")


def train(
    data_path: str | Path | None = None,
    model_path: str | Path = MODEL_PATH,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Train gradient boosting model on glucose data.

    Parameters
    ----------
    data_path:
        Path to CSV. If None, generates synthetic data.
    model_path:
        Where to save the trained model.
    test_size:
        Fraction of data for testing.
    seed:
        Random seed.

    Returns
    -------
    dict with MAE and RMSE metrics.
    """
    if data_path is None:
        print("No data path provided — generating synthetic data...")
        df = synth_data.generate(seed=seed)
    else:
        df = pd.read_csv(data_path)

    df = data_schema.validate(df)
    X, y = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))

    metrics = {
        "MAE_mgdl": round(mae, 2),
        "RMSE_mgdl": round(rmse, 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_names": list(X.columns)}, f)

    metrics_path = model_path.parent / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Test MAE: {mae:.2f} mg/dL | Test RMSE: {rmse:.2f} mg/dL")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train glucose prediction model.")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV data file")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(data_path=args.data, model_path=args.model, seed=args.seed)


if __name__ == "__main__":
    main()
