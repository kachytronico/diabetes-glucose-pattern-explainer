"""Run inference with a trained glucose prediction model."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from diabetes_explainer import data_schema
from diabetes_explainer.features import build_features
from diabetes_explainer.train import MODEL_PATH


def predict(
    data_path: str | Path,
    model_path: str | Path = MODEL_PATH,
    n_last: int = 1,
) -> pd.DataFrame:
    """Return predictions for the last `n_last` rows of data_path.

    Parameters
    ----------
    data_path:
        Path to CSV file with recent glucose readings.
    model_path:
        Path to saved model pickle.
    n_last:
        Number of most recent predictions to return.

    Returns
    -------
    pd.DataFrame with column: predicted_glucose_30min
    """
    df = pd.read_csv(data_path)
    df = data_schema.validate(df)

    with open(model_path, "rb") as f:
        artefact = pickle.load(f)
    model = artefact["model"]
    feature_names = artefact["feature_names"]

    X, _ = build_features(df)
    X = X[feature_names]

    predictions = model.predict(X)
    result = pd.DataFrame(
        {
            "predicted_glucose_30min": predictions.round(1),
        }
    )
    return result.tail(n_last)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict future glucose values.")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--n", type=int, default=5, help="Number of recent predictions to show")
    args = parser.parse_args()
    result = predict(data_path=args.data, model_path=args.model, n_last=args.n)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
