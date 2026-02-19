"""Feature engineering for glucose prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_STEPS = 6  # predict 30 minutes ahead (5-min intervals × 6)
LAG_STEPS = [1, 2, 3, 6, 12]  # 5, 10, 15, 30, 60 minutes


def build_features(
    df: pd.DataFrame, horizon: int = HORIZON_STEPS
) -> tuple[pd.DataFrame, pd.Series]:
    """Build lag features and return (X, y) for supervised learning.

    Parameters
    ----------
    df:
        Validated glucose DataFrame (from data_schema.validate).
    horizon:
        Number of 5-min steps ahead to predict.

    Returns
    -------
    X : pd.DataFrame of features
    y : pd.Series of target glucose values
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    feat = pd.DataFrame(index=df.index)

    # Lag features for glucose
    for lag in LAG_STEPS:
        feat[f"glucose_lag_{lag}"] = df["glucose_mgdl"].shift(lag)

    # Rolling statistics
    feat["glucose_roll_mean_12"] = df["glucose_mgdl"].rolling(12, min_periods=1).mean()
    feat["glucose_roll_std_12"] = df["glucose_mgdl"].rolling(12, min_periods=1).std().fillna(0)

    # Rate of change
    feat["glucose_roc_1"] = df["glucose_mgdl"].diff(1).fillna(0)
    feat["glucose_roc_3"] = df["glucose_mgdl"].diff(3).fillna(0)

    # Event features
    feat["carbs_sum_past_3"] = df["carbs_g"].rolling(3, min_periods=1).sum()
    feat["insulin_sum_past_6"] = df["insulin_units"].rolling(6, min_periods=1).sum()
    feat["steps_sum_past_6"] = df["activity_steps"].rolling(6, min_periods=1).sum()
    feat["hr_mean_past_3"] = df["heart_rate_bpm"].rolling(3, min_periods=1).mean()

    # Time-of-day features
    feat["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)

    # Target: glucose value `horizon` steps in the future
    target = df["glucose_mgdl"].shift(-horizon)

    # Drop rows where target or features are NaN
    valid = feat.notna().all(axis=1) & target.notna()
    return feat[valid].reset_index(drop=True), target[valid].reset_index(drop=True)


FEATURE_DESCRIPTIONS = {
    "glucose_lag_1": "Glucose 5 min ago (mg/dL)",
    "glucose_lag_2": "Glucose 10 min ago (mg/dL)",
    "glucose_lag_3": "Glucose 15 min ago (mg/dL)",
    "glucose_lag_6": "Glucose 30 min ago (mg/dL)",
    "glucose_lag_12": "Glucose 60 min ago (mg/dL)",
    "glucose_roll_mean_12": "Rolling mean glucose (60 min, mg/dL)",
    "glucose_roll_std_12": "Rolling std glucose (60 min, mg/dL)",
    "glucose_roc_1": "Rate of change 5 min (mg/dL)",
    "glucose_roc_3": "Rate of change 15 min (mg/dL)",
    "carbs_sum_past_3": "Total carbs consumed in past 15 min (g)",
    "insulin_sum_past_6": "Total insulin in past 30 min (units)",
    "steps_sum_past_6": "Total steps in past 30 min",
    "hr_mean_past_3": "Mean heart rate in past 15 min (bpm)",
    "hour_sin": "Time-of-day sine encoding",
    "hour_cos": "Time-of-day cosine encoding",
}
