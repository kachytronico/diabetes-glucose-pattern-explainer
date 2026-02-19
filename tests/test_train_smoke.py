"""Smoke tests for training pipeline."""

import pickle

from diabetes_explainer.train import train


def test_train_smoke(tmp_path):
    """Training should complete without errors on synthetic data."""
    model_path = tmp_path / "model.pkl"
    metrics = train(data_path=None, model_path=model_path, seed=0)
    assert model_path.exists()
    assert "MAE_mgdl" in metrics
    assert "RMSE_mgdl" in metrics
    assert metrics["MAE_mgdl"] > 0


def test_train_model_loadable(tmp_path):
    model_path = tmp_path / "model.pkl"
    train(data_path=None, model_path=model_path, seed=1)
    with open(model_path, "rb") as f:
        artefact = pickle.load(f)
    assert "model" in artefact
    assert "feature_names" in artefact


def test_train_mae_reasonable(tmp_path):
    """MAE should be below 30 mg/dL on synthetic data (sanity check)."""
    model_path = tmp_path / "model.pkl"
    metrics = train(data_path=None, model_path=model_path, seed=42)
    assert metrics["MAE_mgdl"] < 30.0, f"MAE too high: {metrics['MAE_mgdl']}"
