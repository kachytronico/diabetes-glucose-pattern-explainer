# Roadmap

This document outlines planned enhancements for future versions.

## v0.2 — Data & Validation
- [ ] Support ingestion of OhioT1DM, Dexcom share, or Nightscout-exported CSV
- [ ] Stricter pydantic-based schema validation with detailed error messages
- [ ] Unit tests for data edge cases (gaps, sensor errors, duplicates)

## v0.3 — Modeling
- [ ] Hyperparameter tuning via cross-validation
- [ ] LSTM / Transformer baseline for sequence modeling
- [ ] Proper SHAP value integration for true per-sample explanations

## v0.4 — Explainability & UX
- [ ] Interactive Streamlit dashboard
- [ ] Natural-language pattern summaries ("Your glucose tends to spike 45–90 min after lunch")
- [ ] Personalisation layer (fine-tune on individual user history)

## v1.0 — Research-Grade
- [ ] Full reproducibility pipeline (DVC or similar)
- [ ] Peer-reviewed citation list
- [ ] Optional anonymised community data submission pipeline (with IRB guidance)

## Non-Goals
- Clinical decision support
- FDA/CE-mark compliance
- Real-time CGM device integration
