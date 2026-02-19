"""Data schema and validation for glucose CSV files."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "glucose_mgdl"]
OPTIONAL_COLUMNS = ["carbs_g", "insulin_units", "activity_steps", "heart_rate_bpm"]
ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

GLUCOSE_MIN = 40.0
GLUCOSE_MAX = 400.0


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce a glucose DataFrame.

    Raises ValueError if required columns are missing or glucose values are out of range.
    Returns cleaned DataFrame with proper dtypes.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    out_of_range = df[(df["glucose_mgdl"] < GLUCOSE_MIN) | (df["glucose_mgdl"] > GLUCOSE_MAX)]
    if len(out_of_range) > 0:
        raise ValueError(
            f"{len(out_of_range)} glucose readings outside valid range "
            f"[{GLUCOSE_MIN}, {GLUCOSE_MAX}] mg/dL"
        )

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    return df[["timestamp", "glucose_mgdl"] + OPTIONAL_COLUMNS]
