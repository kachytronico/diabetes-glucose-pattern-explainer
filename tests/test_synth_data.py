"""Tests for synthetic data generation."""

import pandas as pd

from diabetes_explainer.data_schema import OPTIONAL_COLUMNS, REQUIRED_COLUMNS
from diabetes_explainer.synth_data import generate


def test_generate_returns_dataframe():
    df = generate(n_days=1, seed=0)
    assert isinstance(df, pd.DataFrame)


def test_generate_has_correct_columns():
    df = generate(n_days=1, seed=0)
    for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_generate_glucose_range():
    df = generate(n_days=2, seed=1)
    assert df["glucose_mgdl"].between(40, 400).all()


def test_generate_n_days():
    df = generate(n_days=3, interval_minutes=5, seed=0)
    expected_rows = 3 * 24 * 12  # 3 days × 288 readings/day
    assert len(df) == expected_rows


def test_generate_reproducible():
    df1 = generate(n_days=1, seed=99)
    df2 = generate(n_days=1, seed=99)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_saves_csv(tmp_path):
    out = tmp_path / "test_glucose.csv"
    generate(n_days=1, output_path=out)
    assert out.exists()
    df_loaded = pd.read_csv(out)
    assert "glucose_mgdl" in df_loaded.columns
