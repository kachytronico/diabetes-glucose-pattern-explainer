"""Tests for feature engineering."""

import pandas as pd
import pytest

from diabetes_explainer import data_schema
from diabetes_explainer.features import FEATURE_DESCRIPTIONS, LAG_STEPS, build_features
from diabetes_explainer.synth_data import generate


@pytest.fixture
def sample_df():
    df = generate(n_days=3, seed=42)
    return data_schema.validate(df)


def test_build_features_returns_tuple(sample_df):
    X, y = build_features(sample_df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_build_features_no_nan(sample_df):
    X, y = build_features(sample_df)
    assert not X.isna().any().any()
    assert not y.isna().any()


def test_build_features_lag_columns(sample_df):
    X, _ = build_features(sample_df)
    for lag in LAG_STEPS:
        assert f"glucose_lag_{lag}" in X.columns


def test_build_features_length_consistent(sample_df):
    X, y = build_features(sample_df)
    assert len(X) == len(y)


def test_feature_descriptions_cover_all_features(sample_df):
    X, _ = build_features(sample_df)
    for col in X.columns:
        assert col in FEATURE_DESCRIPTIONS, f"Missing description for feature: {col}"
