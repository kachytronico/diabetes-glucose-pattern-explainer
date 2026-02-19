"""Generate explanations and visualizations for glucose predictions."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from diabetes_explainer import data_schema, synth_data
from diabetes_explainer.features import FEATURE_DESCRIPTIONS, build_features
from diabetes_explainer.train import MODEL_PATH


def _feature_importance_narrative(importances: dict[str, float]) -> str:
    """Convert feature importances to a human-readable narrative."""
    lines = ["## Top Factors Influencing Predicted Glucose\n"]
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    for rank, (feat, imp) in enumerate(sorted_feats[:5], 1):
        desc = FEATURE_DESCRIPTIONS.get(feat, feat)
        lines.append(f"{rank}. **{desc}** — importance score: {imp:.4f}")
    lines.append(
        "\n> ℹ️  These importances reflect patterns in *synthetic* data and are for "
        "educational illustration only."
    )
    return "\n".join(lines)


def explain(
    data_path: str | Path | None = None,
    model_path: str | Path = MODEL_PATH,
    output_dir: str | Path = "output",
    plot: bool = True,
) -> str:
    """Generate an explanation report.

    Parameters
    ----------
    data_path:
        CSV path. If None, generates synthetic data.
    model_path:
        Path to trained model pickle.
    output_dir:
        Directory for output files.
    plot:
        If True, generate matplotlib plots (requires matplotlib).

    Returns
    -------
    Explanation text string.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if data_path is None:
        print("No data provided — using synthetic data...")
        df = synth_data.generate()
    else:
        df = pd.read_csv(data_path)

    df = data_schema.validate(df)
    X, y = build_features(df)

    with open(model_path, "rb") as f:
        artefact = pickle.load(f)
    model = artefact["model"]
    feature_names = artefact["feature_names"]

    X = X[feature_names]
    importances = dict(zip(feature_names, model.feature_importances_))

    narrative = _feature_importance_narrative(importances)

    # Summary stats
    preds = model.predict(X)
    mae = float(np.mean(np.abs(preds - y.values)))

    report_lines = [
        "# Glucose Pattern Explanation Report",
        "",
        "> ⚠️ **DISCLAIMER**: This is an educational prototype. Output is NOT medical advice.",
        "",
        "## Model Performance (on validation slice)",
        f"- Mean Absolute Error: **{mae:.2f} mg/dL**",
        "- Prediction horizon: **30 minutes**",
        f"- Data points evaluated: **{len(X)}**",
        "",
        narrative,
    ]

    report = "\n".join(report_lines)
    report_path = output_dir / "explanation_report.md"
    report_path.write_text(report)
    print(f"Explanation report saved to {report_path}")

    if plot:
        try:
            import matplotlib.pyplot as plt

            sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
            feat_names = [FEATURE_DESCRIPTIONS.get(k, k) for k, _ in sorted_items]
            feat_vals = [v for _, v in sorted_items]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].barh(feat_names[::-1], feat_vals[::-1], color="steelblue")
            axes[0].set_xlabel("Feature Importance")
            axes[0].set_title(
                "Top 10 Feature Importances\n(Synthetic data — educational only)"
            )
            axes[0].tick_params(axis="y", labelsize=8)

            n_plot = min(200, len(preds))
            axes[1].plot(y.values[:n_plot], label="Actual", alpha=0.7)
            axes[1].plot(preds[:n_plot], label="Predicted (30 min ahead)", alpha=0.7)
            axes[1].set_xlabel("Time steps (5 min each)")
            axes[1].set_ylabel("Glucose (mg/dL)")
            axes[1].set_title(
                "Predicted vs Actual Glucose\n(Synthetic data — educational only)"
            )
            axes[1].legend()

            plt.tight_layout()
            plot_path = output_dir / "explanation_plot.png"
            plt.savefig(plot_path, dpi=100)
            plt.close()
            print(f"Plot saved to {plot_path}")
        except ImportError:
            print("matplotlib not available — skipping plot")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate explanation report.")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    explain(
        data_path=args.data,
        model_path=args.model,
        output_dir=args.output,
        plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
