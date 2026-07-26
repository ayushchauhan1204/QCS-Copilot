# QCS Copilot

An AI operator co-pilot for paper machine grade changes — built for the Honeywell Campus Connect Hackathon.

Grade changes are the highest-loss events in paper making: even with good Machine Direction (MD) control, mills still produce off-spec paper while the process settles into a new grade. QCS Copilot predicts *before* it happens whether a transition will breach Basis Weight spec (>2.5% deviation), recommends a corrective setpoint, explains why, and tracks whether operators actually trust its suggestions over time.

## What it does

1. **Predicts** off-spec risk and stabilization time for an in-progress grade change, using only the first 300 seconds of live data (no future leakage).
2. **Recommends** a corrective setpoint via a hybrid engine: recipe rules, k-NN similarity to historical transitions, and model-confidence fallback — every suggestion tagged with its source.
3. **Explains** the prediction and recommendation in plain language, driven by SHAP feature importances (no LLM/generative text — fully deterministic).
4. **Tracks trust**: every Accept/Reject an operator makes is logged, and a calibration chart compares stated confidence against realized outcomes over time.

## Architecture

```
data/               Synthetic historian data generator (physics-informed FOPDT simulation)
  recipes.py            Grade catalog + setpoint derivation
  disturbances.py       Hidden variables (ambient temp, valve wear) + physics disturbances
  transitions.py        Per-event FOPDT simulation loop
  labels.py              offspec_label / stabilization_time_seconds derivation
  generate_data.py       Orchestrator + data quality checks

src/
  data_loader.py          Single load point for the 3 generated CSVs
  features.py             Feature engineering (300s window, leakage-safe)
  correlation_discovery.py  Statistical + ML-validated + open-ended correlation passes
  train_model.py           Classifier (off-spec risk) + Regressor (stabilization time)
  recommend.py             Hybrid recommendation engine + novelty/OOD check
  explain.py               SHAP-driven rationale generation
  feedback_store.py        SQLite-backed accept/reject logging + calibration stats
  cost_model.py            Illustrative $ translation of risk/stabilization time

dashboard/
  app.py                  Streamlit UI — no logic lives here, only presentation
```

Build order is intentionally fixed: each stage depends on the previous one being validated first (e.g. Correlation Discovery's ML-validation pass cannot run before the models are trained).

## Setup

```bash
pip install -r requirements.txt

# Generate the synthetic dataset (250 events, seed=42 — reproducible)
python -m data.generate_data --events 250 --seed 42 --output outputs

# Train the classifier + regressor
python -m src.train_model

# Launch the dashboard
streamlit run dashboard/app.py
```

`outputs/*.csv` and `models/*.pkl` are committed to this repo so a fresh clone (including a Streamlit Cloud deploy) works immediately without re-running the steps above — but they're fully reproducible from the commands if you want to regenerate.

## Honest results (5-fold cross-validated, not a single lucky/unlucky split)

| Component | Metric | Result |
|---|---|---|
| Classifier (off-spec risk) | Accuracy / ROC-AUC | 0.889 / 0.950 |
| Regressor (stabilization time) | R² | 0.302 ± 0.078 |
| Correlation Discovery | Relationships confirmed | 9 of 12 |
| Negative controls | Correctly show no correlation | 8 of 8 |

The regressor is the weakest link — stabilization time is a genuinely noisier target than a binary off-spec label, and we're upfront about that rather than hiding it. The negative controls all coming back clean is a deliberate check that the correlation-discovery method isn't hallucinating patterns.

## Known limitations

- Synthetic data (physics-informed FOPDT simulation), not a real historian — no real plant was available for this hackathon.
- The regressor's R² is moderate; treat stabilization-time predictions with more caution than off-spec risk predictions.
- Cost estimates in the dashboard use illustrative, adjustable assumptions (paper price, web width) — not real mill financials.
- The Trust Ledger's calibration curve needs a meaningful number of logged actions to be statistically informative; a handful of clicks is a demo, not a validation.

## License

Built for the Honeywell Campus Connect Hackathon.