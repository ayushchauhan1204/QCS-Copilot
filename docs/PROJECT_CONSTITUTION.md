# PROJECT CONSTITUTION

**Status: FROZEN.** This document consolidates every finalized decision for this project. It is the single source of truth for any human or AI agent picking up implementation work. Planning is complete — this document records decisions, it does not make new ones.

**If a future step seems to require deviating from this document, STOP and flag it explicitly rather than silently deviating.**

---

## 1. PROJECT OVERVIEW

**Project Name:** QCS Copilot

**Tagline:** Know before it goes off-spec.

**Elevator Pitch:** QCS Copilot is an AI assistant that watches every paper-machine grade change alongside the operator and tells them, in plain language and before it happens, when a transition is at risk of going off-spec — what's driving that risk, what to do about it, and how much to trust the recommendation, based on its own tracked accuracy.

**Problem Statement:** Honeywell's QCS system executes coordinated grade changes well, but grade transitions remain high-loss events — mills produce off-spec paper, broke, or cull material while quality variables stabilize. Current automation calculates targets and trajectories and executes ramps, but does not learn from historical transitions, explain its behavior, or help scarce experienced operators pass on their intuition to newer staff. Site historian data (QCS history, DCS trends, operator actions, alarms, outcomes) is collected but underused.

**Target Users:**
- **Primary:** Control-room operators, especially less experienced ones, who need real-time, explainable risk guidance during a live transition.
- **Secondary:** Process/quality engineers who need to validate discovered correlations and audit recommendation quality over time.

**Business Value:**
- Reduces off-spec/broke material and stabilization time during grade changes.
- Converts underused historian data into actionable, explained guidance instead of passive logs.
- Captures and surfaces institutional knowledge that would otherwise depend on scarce experienced operators.
- Builds justified trust in AI recommendations by tracking and exposing their own accuracy over time, rather than asking operators to trust a black box.

**Success Criteria:**
- End-to-end working demo that visibly satisfies every bullet in the original Hackathon Challenge and Deliverables sections (off-spec prediction, setpoint recommendation, stabilization-time reduction, rationale, correlation discovery, accept/reject logging).
- All outputs traceable to a labeled source of inference (recipe, historical pattern, or model).
- Fully explainable: every prediction and every recommendation has a plain-language "why."
- Achievable end-to-end by one developer in 48 hours without unresolved technical risk at demo time.

---

## 2. PRODUCT CONCEPT (FROZEN)

**Product Vision:** QCS Copilot is not an autopilot — it is an assistant that keeps the operator in control while making the invisible visible: emerging off-spec risk, its root causes, and what similar past transitions did about it. It differentiates itself by treating its own recommendation quality as a first-class, visible product feature, not an internal metric.

**Core Workflow:**
1. Operator starts or replays a grade-change event.
2. Dashboard shows a live trend chart and a risk gauge that updates as the transition progresses.
3. As predicted off-spec risk crosses a threshold, a recommendation card surfaces: suggested setpoint, expected impact, confidence, and source of inference.
4. A "Why?" panel explains the prediction and the recommendation in plain language, grounded in real SHAP values and/or historical/rule evidence.
5. Operator accepts or rejects the recommendation; the decision and eventual outcome are logged.
6. Over time, the dashboard shows a calibration view — how well QCS Copilot's stated confidence has matched real outcomes — building (or appropriately limiting) operator trust.

**Key Features:**
- Off-spec risk classifier (binary: will basis weight exceed ±2.5% during this transition?).
- Stabilization-time regressor (continuous: how long until the process settles?).
- Hybrid recommendation engine combining historical similarity (k-NN over past events), domain/business rules, and model confidence.
- Dual-validated correlation discovery (statistical methods cross-checked against model feature importance), surfacing both expected and previously unknown relationships.
- Explainability layer reused across both predictions and recommendations ("Why?" panel).
- Accept/reject feedback logging with a calibration view (stated confidence vs. actual accuracy).
- Streamlit dashboard consolidating all of the above into one operator-facing view.

**Why It Solves Honeywell's Challenge:** Maps directly onto every stated pain point — high-loss grade changes (risk prediction + recommendations), operators juggling many variables (single consolidated view), automation that executes but doesn't learn (correlation discovery + historical similarity), scarce operator expertise (plain-language rationale), and underused historian data (the correlation engine's entire premise).

**Why Judges Will Remember It:** The live risk-gauge-to-recommendation moment is the most watchable demo beat this architecture can produce, and the calibration/trust framing is a maturity signal — "the system proves whether you should trust it" — that stands out because almost no competing team will present accountability as the product thesis rather than a backend metric.

**Features Intentionally NOT Included** (explicit scope boundary — do not add without a new decision):
- Reinforcement learning of any kind.
- Large language models inside the product (explainability uses deterministic SHAP-driven templates, not generative text).
- Digital twin platforms.
- Cloud infrastructure or deployment.
- Autonomous/closed-loop process control — the operator always makes the final accept/reject decision.
- True real-time streaming dashboards — replay-mode stepping through stored events is used instead (visually near-identical in a demo, removes an entire class of timing/threading risk).
- Gamified training/onboarding mode ("Grade Change Academy" concept — considered, not selected).
- Fleet-level ROI/business dashboard ("ROI Command Center" concept — considered, not selected).
- Sophisticated constrained-optimization recommender (a rule + historical-similarity + model-confidence hybrid is used instead).
- Custom dashboard theming/CSS polish beyond Streamlit defaults.
- Proper time-series forecasting for "future state" — simple trend/linear extrapolation is used instead.
- Survival/censored regression modeling for stabilization time — capped regression is used instead (see §3, Regressor).

---

## 3. SYSTEM ARCHITECTURE (FROZEN)

```
Synthetic Data Generator
        ↓
Feature Engineering
        ↓
Correlation Discovery  ←──────────────┐
        ↓                              │ (validation pass, runs AFTER
Classifier                             │  models exist — see Dependencies)
        ↓                              │
Regressor  ─────────────────────────────┘
        ↓
Hybrid Recommendation Engine
        ↓
Explainability
        ↓
Streamlit Dashboard
        ↓
SQLite Feedback
```

Note: Correlation Discovery runs in **two passes**. Its position in the linear diagram reflects when its primary statistical output is first available (early, right after data generation); its secondary "ML-validated" output depends on the trained Classifier and Regressor and is computed after them. This is a function-call ordering within the module's own responsibility, not a circular import — see its Dependencies entry below.

### 3.1 Synthetic Data Generator
- **Purpose:** Produce historian-style time-series and event-level data for grade-change transitions, standing in for real historian data until/unless it becomes available, per the exact contract in `DATASET_SPECIFICATION.md`.
- **Inputs:** None (self-contained; grade catalog and physical parameters are defined internally per the frozen dataset spec).
- **Outputs:** `historian_timeseries.csv`, `grade_change_events.csv`, `recipe_reference.csv`.
- **Dependencies:** None. This is the root of the pipeline.

### 3.2 Feature Engineering
- **Purpose:** Transform raw historian time series into model-ready features (rolling stats, rate-of-change, time-since-ramp-start, etc.) for both the Classifier and Regressor.
- **Inputs:** `historian_timeseries.csv`, `grade_change_events.csv`.
- **Outputs:** A feature table keyed by `event_id` (and optionally `t_seconds` for within-event features).
- **Dependencies:** Synthetic Data Generator output. Must exclude `aggressive_ramp` (and any future rename such as `operator_ramp_choice`) from model-facing features — see §6.

### 3.3 Correlation Discovery
- **Purpose:** Surface both expected (recipe-explainable) and hidden relationships between process variables, quality outcomes, and the two target variables, using two independent methods so findings can be cross-validated.
- **Inputs (pass 1 — statistical):** `historian_timeseries.csv`, `grade_change_events.csv`. Computes Pearson, Spearman, and lagged correlations (including cross-event lag-1 joins for carryover effects).
- **Inputs (pass 2 — validation):** Pass 1 output + trained Classifier/Regressor feature importances (SHAP or native importances).
- **Outputs:** A correlation/finding table, each entry tagged "statistically found" or "statistically found + ML-confirmed."
- **Dependencies:** Synthetic Data Generator (pass 1); Classifier and Regressor (pass 2 only).

### 3.4 Classifier
- **Purpose:** Predict `offspec_label` — will basis weight exceed the ±2.5% band at any point during this transition?
- **Inputs:** Feature table from Feature Engineering, `offspec_label` from `grade_change_events.csv`.
- **Outputs:** Trained model artifact; predicted probability (used downstream as recommendation confidence); feature importances (consumed by Correlation Discovery pass 2 and Explainability).
- **Dependencies:** Feature Engineering.

### 3.5 Regressor
- **Purpose:** Predict `stabilization_time_seconds` (capped at 2400s per the dataset spec's censoring rule).
- **Inputs:** Feature table from Feature Engineering, `stabilization_time_seconds` from `grade_change_events.csv`.
- **Outputs:** Trained model artifact; predicted stabilization time; feature importances (consumed by Correlation Discovery pass 2 and Explainability).
- **Dependencies:** Feature Engineering. Trained independently from the Classifier (two single-output models, not one joint multi-task model — frozen decision).

### 3.6 Hybrid Recommendation Engine
- **Purpose:** For an at-risk transition, generate setpoint recommendations by combining three evidence sources into one ranked, tagged output.
- **Inputs:** Current event state/features; Classifier probability and Regressor estimate; `recipe_reference.csv` (domain rules); historical `grade_change_events.csv` (for k-NN similarity search).
- **Outputs:** Recommendation objects, each with: suggested setpoint, expected impact (stated against **both** off-spec risk reduction and stabilization-time reduction — see §6), confidence, and source tag (historical pattern / rule / model).
- **Dependencies:** Classifier, Regressor, Synthetic Data Generator output (for historical similarity and recipe rules).

### 3.7 Explainability
- **Purpose:** Convert SHAP values and recommendation evidence into deterministic, plain-language rationale strings for both predictions and recommendations — the single "Why?" logic reused across both.
- **Inputs:** Classifier/Regressor SHAP values; Hybrid Recommendation Engine evidence (which source(s) contributed, similarity matches, rule triggers).
- **Outputs:** Rationale strings consumed by the dashboard's "Why?" section.
- **Dependencies:** Classifier, Regressor, Hybrid Recommendation Engine.

### 3.8 Streamlit Dashboard
- **Purpose:** Single operator-facing view consolidating trend, risk gauge, stabilization-time estimate, correlation panels (split into "what's driving off-spec risk" and "what's driving slow stabilization" — two explicitly separate sections, not one merged panel), recommendation cards, the "Why?" section, and the calibration/trust view.
- **Inputs:** All upstream module outputs.
- **Outputs:** None (terminal UI layer). Writes accept/reject events to SQLite Feedback.
- **Dependencies:** All other modules. Must not compute predictions/recommendations itself — it calls into `src/` modules (see §6, no hidden dependencies).

### 3.9 SQLite Feedback
- **Purpose:** Persist every accept/reject decision against its recommendation, enabling the calibration view and satisfying the explicit "record responses to evaluate suggestion quality" deliverable.
- **Inputs:** Recommendation ID, accept/reject decision, (eventually) realized outcome.
- **Outputs:** Feedback records; aggregate accuracy/calibration statistics consumed by the Dashboard.
- **Dependencies:** Hybrid Recommendation Engine (for recommendation IDs/metadata).

---

## 4. PROJECT STRUCTURE

```
qcs_copilot/
├── dashboard/
│   ├── __init__.py
│   └── app.py                    # Streamlit dashboard — primary operator-facing UI entrypoint
├── data/
│   ├── recipes.py            # Grade catalog and recipe-implied setpoint formulas
│   ├── disturbances.py       # Hidden-variable generators: valve wear, ambient temp, carryover, overshoot events
│   ├── transitions.py        # FOPDT ramp dynamics and the per-event simulation loop
│   ├── labels.py             # offspec_label, stabilization_time_seconds, outcome_category derivation
│   ├── generate_data.py      # Orchestrator: wires recipes/disturbances/transitions/labels, writes CSVs
│   └── output/
│       ├── historian_timeseries.csv
│       ├── grade_change_events.csv
│       └── recipe_reference.csv
├── src/
│   ├── data_loader.py            # Single load_events()/load_timeseries() interface — abstracts sim vs. real data
│   ├── features.py               # Feature engineering for both models
│   ├── correlation_discovery.py  # Both correlation passes (statistical + ML-validated)
│   ├── train_model.py            # Trains and saves both Classifier and Regressor
│   ├── explain.py                # SHAP extraction + rationale templates (shared by predictions and recommendations)
│   ├── recommend.py              # Hybrid recommendation engine (historical + rules + model confidence)
│   ├── feedback_store.py         # SQLite wrapper: log_feedback(), get_accuracy_stats()
│   └── app.py                    # Streamlit dashboard entrypoint wrapper / re-exporter
├── models/
│   ├── offspec_classifier.pkl
│   └── stabilization_regressor.pkl
├── db/
│   └── feedback.db
├── docs/
│   ├── DATASET_SPECIFICATION.md
│   ├── PROJECT_CONSTITUTION.md
│   └── PRESENTATION_NOTES.txt    # Running bullet notes for the final Idea Submission deck, filled in as built
├── requirements.txt
└── README.md
```

**Directory responsibilities:**
- **`dashboard/`** — user interface layer containing the primary Streamlit operator terminal (`dashboard/app.py`).
- **`data/`** — everything needed to produce the three canonical CSVs. Nothing outside `data/` may generate or mutate historian-shaped data.
- **`data/output/`** — generated artifacts only; never hand-edited.
- **`src/`** — all analysis, modeling, and UI backend logic. Each file maps to exactly one architecture module from §3 (`train_model.py` covers both Classifier and Regressor).
- **`models/`** — serialized trained artifacts only; never hand-edited; regenerated by `train_model.py`.
- **`db/`** — SQLite feedback database; owned exclusively by `feedback_store.py`.
- **`docs/`** — all planning and reference documents, including this one.

---

## 5. CODING STANDARDS

- **Python version:** 3.11.
- **Style:** PEP8 throughout; no exceptions without a documented reason.
- **Type hints:** Required on every function signature (parameters and return type), including within Streamlit callbacks.
- **Modular functions:** Single-responsibility; a function that does two distinct things gets split. Target under ~40 lines per function as a guideline, not a hard rule.
- **Small files:** Each `src/` file stays close to its original LOC estimate from planning (roughly 50–250 lines). A file that grows far beyond that is a signal its responsibility has drifted — split it, don't let it grow.
- **Meaningful naming:** `snake_case` for functions/variables, no unexplained abbreviations, names should make a docstring unnecessary for trivial functions.
- **Docstrings:** Every public function gets a one-line summary; non-trivial functions (anything with more than one parameter or non-obvious behavior) get a full docstring with `Args:` and `Returns:`.
- **Logging:** Use Python's `logging` module in all `data/` and `src/` pipeline code — no bare `print()` for anything other than the CLI entry point's final summary. The Streamlit dashboard may use `st.write`/`st.toast` for user-facing messages, but its underlying calls into `src/` still log normally.
- **Error handling:** Wrap I/O and model-loading calls in `try/except` with specific exception types and clear messages. For a hackathon prototype: **fail loudly and immediately** rather than silently degrading — a silent fallback that produces a plausible-looking wrong number during judging is worse than a visible crash during development.

---

## 6. NON-NEGOTIABLE RULES

1. **Never redesign the architecture.** §3 is frozen. If a genuine blocker requires a change, stop and flag it explicitly for human sign-off before proceeding — do not silently adapt.
2. **Never rename, remove, or retype a column** from `DATASET_SPECIFICATION.md` without updating that document first, in the same change.
3. **Never modify the folder structure** in §4 without updating this document in the same change.
4. **Never change a module's responsibility** as defined in §3 (e.g., no model training logic inside `app.py`; no UI code inside `train_model.py`).
5. **Never implement multiple modules simultaneously.** Follow the frozen build order: Data Generator → Feature Engineering → Correlation Discovery (pass 1) → Classifier + Regressor → Correlation Discovery (pass 2) → Recommendation Engine → Explainability → Feedback Store → Dashboard.
6. **Never create hidden dependencies.** `app.py` calls into `src/` modules; it never computes predictions, recommendations, or statistics itself. Every cross-module dependency listed in §3 must be an explicit function call/import, never a shared global or side-channel file read.
7. **Never feed `aggressive_ramp`** (or any future rename, e.g. `operator_ramp_choice`) **to the Classifier or Regressor as a model feature.** It is simulator ground truth a real historian would not log, and using it directly would leak the label. Models must learn from the actual computed ramp rate instead.
8. **Never change the off-spec threshold (±2.5%) or the settled-band definition (±1.5%, 60-second continuous hold)** without updating `DATASET_SPECIFICATION.md` and `data/labels.py` together, in the same change.
9. **Never run Correlation Discovery's validation pass before the Classifier and Regressor are trained.** Its two-pass order (§3.3) is fixed.
10. **Never regenerate the final 500-event dataset** until the full pipeline has been validated end-to-end on the 250-event development set.
11. **Never skip accept/reject feedback logging.** It is an explicit graded deliverable, not an optional feature.
12. **Never add a new external dependency** without confirming it fits the 48-hour budget and can be justified to a judge in one sentence.
13. **Never introduce real-time streaming, reinforcement learning, in-product LLM calls, a digital twin platform, cloud infrastructure, or autonomous/closed-loop control.** These are explicitly out of scope per §2 and are not to be added under any framing (defensive, "just for the demo," "just a stub," etc.).
14. **Never let the dashboard depend on live, potentially-slow, or potentially-failing computation during judging.** Use replay mode over stored/cached results, not on-the-fly simulation.
15. **Always keep `offspec_label` and `stabilization_time_seconds`** exactly as defined in `DATASET_SPECIFICATION.md` §12 — any model, dashboard panel, or recommendation logic that references these must use these exact definitions, not an approximation.
