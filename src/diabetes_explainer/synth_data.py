"""Generate synthetic glucose time-series data for demonstration purposes.

The synthetic data captures qualitative patterns (post-meal rise, insulin dip,
circadian drift) but is NOT derived from real patients and does NOT represent
any medical ground truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(
    n_days: int = 14,
    interval_minutes: int = 5,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate a synthetic glucose DataFrame.

    Parameters
    ----------
    n_days:
        Number of days of data to generate.
    interval_minutes:
        CGM reading interval in minutes (default 5, typical CGM rate).
    seed:
        Random seed for reproducibility.
    output_path:
        If provided, save the DataFrame as CSV to this path.

    Returns
    -------
    pd.DataFrame with columns: timestamp, glucose_mgdl, carbs_g,
        insulin_units, activity_steps, heart_rate_bpm
    """
    rng = np.random.default_rng(seed)
    n_points = int(n_days * 24 * 60 / interval_minutes)
    timestamps = pd.date_range("2024-01-01", periods=n_points, freq=f"{interval_minutes}min")

    # baseline circadian rhythm (sinusoidal, peak around 8am)
    t_hours = np.arange(n_points) * interval_minutes / 60
    circadian = 10 * np.sin(2 * np.pi * (t_hours - 6) / 24)

    # baseline glucose with random walk component
    noise = rng.normal(0, 1.5, n_points).cumsum() * 0.05
    baseline = 100 + circadian + noise
    baseline = np.clip(baseline, 70, 160)

    glucose = baseline.copy()

    # simulate meal events
    carbs = np.zeros(n_points)
    insulin = np.zeros(n_points)

    meals_per_day = [
        (7 * 60, 60, 20),   # breakfast: 7am ± 60min, ~60g carbs
        (12 * 60, 90, 70),  # lunch: 12pm ± 90min, ~70g carbs
        (19 * 60, 60, 60),  # dinner: 7pm ± 60min, ~60g carbs
    ]

    for day in range(n_days):
        day_offset = day * int(24 * 60 / interval_minutes)
        for meal_min, jitter_min, carb_mean in meals_per_day:
            meal_idx = day_offset + int(
                (meal_min + rng.uniform(-jitter_min / 2, jitter_min / 2)) / interval_minutes
            )
            meal_idx = min(meal_idx, n_points - 1)
            meal_carbs = max(0, rng.normal(carb_mean, 15))
            carbs[meal_idx] = meal_carbs

            # glucose rises after meal, peaks ~45-90 min later
            peak_delay = int(rng.uniform(8, 18))  # steps
            peak_rise = meal_carbs * rng.uniform(0.8, 1.5)
            for k in range(min(30, n_points - meal_idx)):
                rise = peak_rise * np.exp(-0.5 * ((k - peak_delay) / 6) ** 2)
                glucose[meal_idx + k] = min(glucose[meal_idx + k] + rise, 350)

            # insulin dose
            insulin_dose = max(0, rng.normal(meal_carbs / 12, 1.0))
            if meal_idx + 5 < n_points:
                insulin[meal_idx + 5] = insulin_dose
            # insulin lowers glucose 30-120 min after injection
            for k in range(5, min(25, n_points - meal_idx)):
                drop = insulin_dose * rng.uniform(3, 6) * np.exp(-0.3 * (k - 5))
                glucose[meal_idx + k] = max(glucose[meal_idx + k] - drop, 55)

    # activity effect
    steps = np.zeros(n_points)
    for day in range(n_days):
        day_offset = day * int(24 * 60 / interval_minutes)
        walk_start = day_offset + int((rng.uniform(15, 17) * 60) / interval_minutes)
        walk_dur = int(rng.uniform(20, 60) / interval_minutes)
        walk_start = min(walk_start, n_points - 1)
        for k in range(min(walk_dur, n_points - walk_start)):
            steps[walk_start + k] = rng.uniform(80, 130)
            glucose[walk_start + k] = max(glucose[walk_start + k] - rng.uniform(0.5, 2), 55)

    # heart rate (correlated with activity + small meal bump)
    hr = 70 + steps * 0.3 + carbs * 0.05 + rng.normal(0, 3, n_points)
    hr = np.clip(hr, 45, 200)

    # final glucose clamp
    glucose = np.clip(glucose, 55, 350)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "glucose_mgdl": np.round(glucose, 1),
            "carbs_g": np.round(carbs, 1),
            "insulin_units": np.round(insulin, 2),
            "activity_steps": np.round(steps, 0),
            "heart_rate_bpm": np.round(hr, 1),
        }
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Synthetic data saved to {output_path}  ({len(df)} rows)")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic glucose data.")
    parser.add_argument("--days", type=int, default=14, help="Number of days")
    parser.add_argument("--interval", type=int, default=5, help="CGM interval in minutes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/synthetic_glucose.csv")
    args = parser.parse_args()
    generate(
        n_days=args.days,
        interval_minutes=args.interval,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
