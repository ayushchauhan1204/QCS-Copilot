# QCS Copilot

**Know before it goes off-spec.**

An AI operator co-pilot for paper machine grade changes — built for the Honeywell Campus Connect Hackathon.

Grade changes are the highest-loss events in paper making: even with good Machine Direction (MD) control, mills still produce off-spec paper while the process settles into a new grade. Current QCS automation calculates targets and trajectories and executes coordinated ramps, but it doesn't learn from historical transitions, explain its own behavior, or help scarce experienced operators pass on their intuition to newer staff.

QCS Copilot is a co-pilot, not an autopilot — it watches every grade change alongside the operator and tells them, in plain language and before it happens, when a transition is at risk of going off-spec: what's driving that risk, what to do about it, and how much to trust the recommendation, based on the system's own tracked accuracy.

## What it does

1. **Predicts** off-spec risk (Basis Weight deviating >2.5% from setpoint) and stabilization time, using only the first 300 seconds of live data — genuinely before the outcome is known, not after the fact.
2. **Recommends** a corrective setpoint via a hybrid engine: recipe rules, k-NN similarity to historical transitions, and model-confidence fallback — every suggestion tagged with its source.
3. **Explains** every prediction and recommendation in deterministic, SHAP-driven plain language — no generative AI/LLM inside the product.
4. **Tracks trust**: every operator Accept/Reject is logged, and a calibration view compares stated confidence against realized outcomes over time.

---

## Pipeline Architecture

The pipeline is a fixed, linear build order — each stage is validated before the next depends on it. The one exception is Correlation Discovery, which runs in two passes: an early statistical pass (right after the data exists) and a later ML-validation pass (only after the Classifier/Regressor are trained).

![QCS Copilot pipeline architecture](https://drive.google.com/file/d/1-qrb0Tp33KUVmKQ8Aa4jLdNUKKpvI4_E/view?usp=drive_link)

### Module-by-module breakdown

| Module | Purpose | Inputs | Outputs |
|---|---|---|---|
| `data/*.py` | Physics-informed synthetic historian generator (FOPDT dynamics) | None — self-contained, seed=42 | 3 CSVs: `recipe_reference`, `grade_change_events`, `historian_timeseries` |
| `data_loader.py` | Single load point for the 3 CSVs — no other module reads them directly | The 3 generated CSVs | `load_events()` / `load_timeseries()` / `load_recipes()` |
| `features.py` | Leakage-safe feature engineering: 300s window cutoff, planned-vs-actual split, one-hot grades | Raw CSVs via `data_loader` | Feature table keyed by `event_id` |
| `correlation_discovery.py` | Pass 1: statistical relationship tests + negative controls + open-ended scan. Pass 2: cross-checks Pass 1 against trained model feature importances | Pass 1: features. Pass 2: + trained models | Tagged findings table (statistically found / ML-confirmed / negative control) |
| `train_model.py` | Trains Classifier (off-spec risk) and Regressor (stabilization time) | Feature table + both targets | 2 trained models (`.pkl`) + feature importances |
| `recommend.py` | Hybrid recommendation engine: recipe rules + k-NN historical similarity + model-confidence fallback, plus a novelty/out-of-distribution check | Trained models, `recipe_reference`, historical events | Ranked, source-tagged `Recommendation` objects |
| `explain.py` | SHAP-driven, deterministic plain-language rationale for both predictions and recommendations | SHAP values, recommendation evidence | Rationale strings for the dashboard's "Why?" panel |
| `feedback_store.py` | SQLite-backed accept/reject logging and calibration statistics | Operator Accept/Reject actions | Logged feedback + calibration curve data |
| `cost_model.py` | Illustrative $ translation of risk/stabilization time, using adjustable assumptions | Basis weight, machine speed, duration | Estimated broke cost (USD) |
| `dashboard/app.py` | Streamlit UI only — consolidates every module above into one operator-facing view | All upstream module outputs | Rendered UI; writes Accept/Reject events to `feedback_store` |

### Communication between modules

Two rules govern every cross-module call in this system, deliberately, to keep the pipeline auditable:

- **No hidden dependencies.** `dashboard/app.py` contains no prediction, recommendation, or statistical logic of its own — it only calls into `src/` modules and renders their output. Every dependency is an explicit function import, never a shared global or a side-channel file read.
- **A single data choke point.** No module outside `data/` ever reads or writes the three canonical CSVs directly — every downstream consumer goes through `data_loader.py`. This is what would let real Honeywell historian data be substituted in later with no changes to `src/`, provided it's reshaped to match the same schema.

Read the diagram above left-to-right: raw synthetic data becomes leakage-safe features, features train two models, those models' own feature importances get cross-checked against the statistical correlation pass, and the same trained models feed both the recommendation engine and the explainability layer. The feedback store is the one bidirectional edge — the dashboard writes Accept/Reject events into it, and reads calibration statistics back out.

---

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

---

## Honest model performance

Reported via 5-fold cross-validation, not a single train/test split — with only ~217 valid training rows, a single 80/20 split leaves ~43 test rows, and metrics from a split that small can swing widely depending on which rows happen to land in the test set (verified empirically during development).

| Component | Metric | Result |
|---|---|---|
| Classifier (off-spec risk) | Accuracy / ROC-AUC | 0.889 / 0.950 |
| Regressor (stabilization time) | R² (5-fold CV) | 0.302 ± 0.078 |
| Correlation Discovery | Relationships confirmed | 9 of 12 |
| Negative controls | Correctly show no correlation | 8 of 8 |

The regressor is the weakest link in the system, stated here deliberately rather than left for a judge to discover: stabilization time is a genuinely noisier target than a binary off-spec label. The negative controls all coming back clean is a deliberate rigor check — a correlation-discovery method that reports spurious findings on variables engineered to be undisturbed would be a sign the method itself is flawed, not a real discovery.

---

## Known limitations

- Synthetic data (physics-informed FOPDT simulation), not a real plant historian — no real Honeywell site data was available for this hackathon.
- The regressor's R² is moderate; stabilization-time predictions warrant more caution than off-spec risk predictions.
- Cost estimates use illustrative, user-adjustable assumptions (paper price, web width) — not real mill financials.
- The Trust Ledger's calibration curve needs a meaningful number of logged actions to be statistically informative; a handful of demo clicks illustrates the mechanism, not a validated calibration.
- The correlation-discovery module's open-ended scan surfaces exploratory leads at an uncorrected significance threshold — some flagged variables (e.g. machine speed, basis weight averages) are plausibly confounded by grade-pair identity rather than being independent causal drivers, and are reported as leads, not confirmed findings.

---

## License

Built for the Honeywell Campus Connect Hackathon.