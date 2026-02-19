# 🩺 Diabetes Glucose Pattern Explainer

**Explainable AI for glucose education — prototype (NOT medical advice)**

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/diabetes-glucose-pattern-explainer/diabetes-glucose-pattern-explainer/actions/workflows/ci.yml/badge.svg)

---

> **⚠️ SAFETY DISCLAIMER**
>
> This is an **educational / research prototype only**.
> It is **NOT medical advice**, **NOT a medical device**, and is **NOT intended for clinical use**.
> **Do NOT** use this tool to make any health or treatment decisions.
> Always consult a qualified healthcare professional.
> See [DISCLAIMER.md](DISCLAIMER.md) for the full disclaimer.

---

## What is this?

The **Diabetes Glucose Pattern Explainer** is an open-source Python project that demonstrates how machine learning can model short-horizon blood glucose dynamics from continuous glucose monitor (CGM)-style time series. It is designed as an **educational prototype** to help developers, researchers, and students understand:

- How lag features and rolling statistics can represent physiological signals
- How gradient boosted trees learn temporal patterns from tabular health data
- How feature importances can illustrate which factors (meals, insulin, activity, time-of-day) correlate with glucose changes in a model

**Everything runs on fully synthetic data by default.** No real patient data is ever required.

---

## Quickstart

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install the package
pip install -e ".[dev]"

# 3. Generate synthetic glucose data (saved to data/synthetic_glucose.csv)
python -m diabetes_explainer.synth_data

# 4. Train the prediction model (saved to models/glucose_model.pkl)
python -m diabetes_explainer.train

# 5. Generate explanation report and plots (saved to output/)
python -m diabetes_explainer.explain
```

---

## Project Structure

```
diabetes-glucose-pattern-explainer/
├── src/
│   └── diabetes_explainer/
│       ├── __init__.py          # Package version
│       ├── data_schema.py       # CSV schema validation
│       ├── synth_data.py        # Synthetic data generator
│       ├── features.py          # Feature engineering
│       ├── train.py             # Model training
│       ├── predict.py           # Inference
│       └── explain.py           # Explanation report + plots
├── tests/
│   ├── test_synth_data.py
│   ├── test_features.py
│   └── test_train_smoke.py
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   └── 02_modeling.ipynb        # Modeling walkthrough
├── docs/
│   └── ROADMAP.md
├── .github/workflows/ci.yml
├── pyproject.toml
├── DISCLAIMER.md
├── MODEL_CARD.md
├── CONTRIBUTING.md
└── CODE_OF_CONDUCT.md
```

---

## Data

By default, the project generates **fully synthetic** CGM-like data via `synth_data.py`. This data encodes qualitative patterns (post-meal glucose rise, insulin dip, circadian variation, activity effect) but does not represent any real patient.

### CSV Schema (for optional real data)

If you wish to use real exported CGM data, prepare a CSV with the following columns:

| Column | Required | Type | Description |
|---|---|---|---|
| `timestamp` | ✅ | datetime | ISO 8601 or parseable datetime string |
| `glucose_mgdl` | ✅ | float | Blood glucose in mg/dL (valid range: 40–400) |
| `carbs_g` | ☐ | float | Carbohydrate intake in grams (default: 0) |
| `insulin_units` | ☐ | float | Insulin dose in units (default: 0) |
| `activity_steps` | ☐ | float | Step count (default: 0) |
| `heart_rate_bpm` | ☐ | float | Heart rate in BPM (default: 0) |

> You are solely responsible for de-identification and regulatory compliance when using real health data.

---

## Safety & Ethics

This project takes data safety and responsible AI seriously:

- 📄 [DISCLAIMER.md](DISCLAIMER.md) — Full disclaimer, no-warranty statement
- 🃏 [MODEL_CARD.md](MODEL_CARD.md) — Model details, intended use, limitations, ethical considerations
- 🤝 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — Community standards

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setting up your development environment, running tests, and submitting pull requests.

```bash
# Run tests
pytest

# Run linter
ruff check src tests
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
