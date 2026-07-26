# DATASET SPECIFICATION

**Status: FROZEN.** This document is the single source of truth for everything data-related in this project — `data/*.py`, `src/features.py`, `src/correlation_discovery.py`, and `src/train_model.py` must all conform to it exactly. It consolidates and supersedes all prior dataset design discussion.

**Two items in this document (§8's prediction-cutoff mechanism, and encoding choices) fill genuine gaps that hadn't been made numerically explicit before now — they operationalize existing decisions, they do not introduce new architecture. Both are flagged inline for confirmation before implementation begins.**

---

## 1. DATASET OVERVIEW

**Purpose:** Provide a historian-equivalent dataset for paper-machine grade-change transitions, standing in for real Honeywell QCS/DCS historian data until (or unless) it becomes available. The dataset is synthetic but physics-informed (first-order-lag-plus-dead-time loop dynamics) and deliberately embeds hidden, non-recipe relationships so the correlation-discovery module has real signal to find.

**Files:** Three CSVs, each with one clear responsibility:
- `recipe_reference.csv` — static grade catalog and recipe-implied control limits.
- `grade_change_events.csv` — one row per grade-change event: metadata, labels, diagnostics.
- `historian_timeseries.csv` — one row per (event, timestep): the full sensor/actuator trace.

**Relationships:**
- `event_id` is the primary/foreign key linking `grade_change_events.csv` (1 row) to `historian_timeseries.csv` (480 rows).
- `grade_name` in `recipe_reference.csv` is referenced by `from_grade`/`to_grade` in both other files.

**Workflow:** `data/generate_data.py` orchestrates `recipes.py` → `disturbances.py` → `transitions.py` → `labels.py` and writes all three CSVs to `data/output/`. **No module outside `data/` may read or write these files directly** — every downstream consumer goes through `src/data_loader.py`'s `load_events()` / `load_timeseries()` interface. This single choke point is what lets real historian data be substituted later with no changes to `src/`, provided it's reshaped to match this schema exactly.

**Reproducibility:** Random seed fixed at `42` for the generator. Changing it requires regenerating and revalidating the entire downstream pipeline (§10).

---

## 2. CSV FILES

### 2.1 `recipe_reference.csv`

**Purpose:** The single source of truth for recipe-implied targets and business-rule limits — used by the rule-based arm of the recommendation engine and by the correlation engine to distinguish "known/recipe-explainable" from "new" correlations.

| Column | Type | Units | Range | Description |
|---|---|---|---|---|
| `grade_name` | string | — | one of 5 grades | Primary key |
| `basis_weight_setpoint` | float | gsm | 45–100 | Recipe target |
| `moisture_setpoint` | float | % | 6.0–8.0 | Recipe target |
| `ash_setpoint` | float | % | 4.0–12.0 | Recipe target |
| `caliper_setpoint` | float | µm | 55–120 | Recipe target |
| `offspec_band_pct` | float | % | fixed = 2.5 | Global spec threshold, stored per-row for future per-grade tuning |
| `recommended_max_ramp_rate` | float | gsm/min | 3–8 (grade-dependent) | Recipe-book "safe" ramp rate — exact per-grade value finalized during `recipes.py` implementation, within this band |
| `typical_transition_time_min` | float | min | 15–35 | Recipe-book expected duration baseline |

### 2.2 `grade_change_events.csv`

**Purpose:** Event-level metadata, diagnostics, and both ML targets — the table `train_model.py` trains against.

| Column | Type | Units | Range | Description |
|---|---|---|---|---|
| `event_id` | int | — | 0..N-1 | Primary key |
| `event_start_timestamp` | datetime | — | — | Synthetic absolute time, for historian realism |
| `from_grade` / `to_grade` | string | — | one of 5 grades | FK to `recipe_reference.csv`; always different |
| `operator_ramp_choice` | bool | — | — | Generation-control flag — **never a model feature**, see §8 |
| `ramp_duration_seconds` | int | s | 240–1430 (±10% jitter applied) | Actual ramp window used |
| `valve_wear_index` | float | 0–1 | 0–1 | Hidden variable |
| `ambient_temp_c` | float | °C | 15–35 | Environmental/hidden variable |
| `prior_ash` | float | % | 4–12 | Hidden variable (cross-event carryover) |
| `max_abs_basis_weight_pct_dev` | float | % | 0–15+ | Diagnostic only — never a model feature |
| `offspec_label` | int {0,1} | — | — | **Classifier target** |
| `time_to_offspec_s` | float, nullable | s | — | Diagnostic only — null if never breached |
| `stabilization_time_seconds` | float | s | 0–2400 | **Regressor target**, capped |
| `stabilized_within_window` | bool | — | — | Diagnostic/censoring flag |
| `outcome_category` | string | — | {clean, recovered, unresolved} | Diagnostic only — never a model feature |
| `event_duration_seconds` | int | s | fixed = 2400 | Observation window length |

### 2.3 `historian_timeseries.csv`

**Purpose:** The full sensor/actuator trace, sampled every 5 seconds — the raw material for feature engineering and correlation discovery.

| Column | Type | Units | Range | Description |
|---|---|---|---|---|
| `event_id` | int | — | — | FK |
| `timestamp` | datetime | — | — | `event_start_timestamp + t_seconds` |
| `t_seconds` | int | s | 0–2395, step 5 | Elapsed time within event |
| `from_grade` / `to_grade` | string | — | — | — |
| `stock_flow` / `stock_flow_setpoint` | float | L/min | 412–550 | Control variable |
| `filler_flow` / `filler_flow_setpoint` | float | L/min | 32–56 | Control variable |
| `steam_pressure` / `steam_pressure_setpoint` | float | kPa | 310–330 | Control variable |
| `machine_speed` / `machine_speed_setpoint` | float | m/min | 700–810 | Control variable |
| `basis_weight` / `basis_weight_setpoint` | float | gsm | 40–105 | **Primary quality variable** |
| `basis_weight_pct_dev` | float | % | -15–+15 | Derived: defines both targets |
| `moisture` | float | % | 5.5–8.5 | Quality variable |
| `ash` | float | % | 3–13 | Quality variable |
| `caliper` | float | µm | 50–125 | Quality variable |
| `ambient_temp_c` | float | °C | 15–35 | Environmental/hidden (repeated per row) |
| `valve_wear_index` | float | 0–1 | 0–1 | Hidden (repeated per row) |
| `operator_ramp_choice` | bool | — | — | Never a model feature |

---

## 3. GRADE CHANGE EVENTS

- **Event duration:** fixed 2400 seconds (40 minutes) per event — long enough to observe full settling even under the slowest, most disturbed conditions.
- **Sampling interval:** 5 seconds → 480 rows/event in `historian_timeseries.csv`.
- **Grades (5, fixed):**

| Grade | Basis Weight (gsm) | Moisture (%) | Ash (%) | Caliper (µm) |
|---|---|---|---|---|
| G1_LightBond | 45 | 6.0 | 8.0 | 55 |
| G2_StandardBond | 60 | 6.5 | 10.0 | 70 |
| G3_HeavyBond | 80 | 7.0 | 12.0 | 95 |
| G4_Kraft | 100 | 7.5 | 6.0 | 120 |
| G5_Newsprint | 52 | 8.0 | 4.0 | 65 |

- **Transition logic:** `from_grade`/`to_grade` sampled uniformly without replacement per event (20 ordered pairs possible). `operator_ramp_choice` (Bernoulli p=0.35) sets the ramp duration to a short fraction (18% of the window) vs. a calm fraction (50%), each with ±10% random jitter so the flag isn't a perfectly clean separator (see §8 leakage note). All four control-variable setpoints ramp **linearly** from their source recipe-implied value to their destination value over that window, then hold; each control variable and the basis weight itself lags its setpoint via an independent first-order response (§4).

---

## 4. PROCESS VARIABLES

### Control Variables (Manipulated Variables)

| Variable | Range | Noise (σ) | Disturbance effects | Expected behavior |
|---|---|---|---|---|
| `stock_flow` | 412–550 L/min | 1.2 L/min | Weak secondary effect from `ambient_temp_c` (small addition to its own lag time constant) | First-order lag toward its ramping setpoint |
| `filler_flow` | 32–56 L/min | 0.4 L/min | `valve_wear_index` elevates its lag time constant and can trigger a mid-ramp overshoot bump | First-order lag toward its ramping setpoint, occasional oscillatory overshoot |
| `steam_pressure` | 310–330 kPa | 2.0 kPa | **None** — intentionally left clean | First-order lag toward its ramping setpoint |
| `machine_speed` | 700–810 m/min | 1.0 m/min | **None** — intentionally left clean | First-order lag toward its ramping setpoint |

`steam_pressure` and `machine_speed` are deliberately left free of hidden disturbances. They function as **negative controls**: a correlation-discovery module that reports spurious "hidden relationships" for these two should be treated as a signal of a flawed method, not a real finding.

### Quality Variables

| Variable | Range | Noise (σ) | Disturbance effects | Expected behavior |
|---|---|---|---|---|
| `basis_weight` | 40–105 gsm | 0.25 gsm | **Primary target of both hidden relationships A and E** (ambient temp, valve wear — see §6), plus a small direct cross-coupling bump from valve overshoot | First-order lag (τ_bw) toward its own ramping setpoint — the dominant dynamic in the whole dataset |
| `moisture` | 5.5–8.5 % | 0.08 % | None direct — inherits `steam_pressure`'s clean status | First-order lag toward a target derived from `steam_pressure` |
| `ash` | 3–13 % | 0.15 % | `valve_wear_index` (via `filler_flow`'s lag and overshoot) and `prior_ash` carryover at event start (hidden relationship D) | First-order lag toward a target derived from `filler_flow`, initial condition blended 15% toward the previous event's destination ash |
| `caliper` | 50–125 µm | 0.5 µm | **None** — intentionally left clean | Simple first-order lag (fixed τ=100s) toward the linearly interpolated src→dst target |

`caliper` is a third negative control, alongside `steam_pressure` and `machine_speed`.

### Environmental / Hidden Variables

| Variable | Range | Behavior |
|---|---|---|
| `ambient_temp_c` | 15–35 °C | Drawn once per event from Normal(23, 3), clipped. **Static for the entire event** (same value on all 480 rows). |
| `valve_wear_index` | 0–1 | Drawn once per event from Beta(2,5), clipped, **plus a linear drift of up to +0.3 as a function of position in the generation log** — static within an event, systematically trending upward across `event_id`. |
| `prior_ash` | 4–12 % | Deterministic: equals the destination-grade ash setpoint of the *immediately preceding* event in generation order (first event seeded with a default of 10.0%). |

---

## 5. DISTURBANCES

| Disturbance | Root Cause | Affected Variables | Severity | Duration | Probability |
|---|---|---|---|---|---|
| Ambient temperature drift | Wet-end/plant ambient conditions | `basis_weight` (strong, via τ_bw), `stock_flow` (weak, via τ_stock) | τ_bw increases by up to ~2×(temp−22)°C seconds; τ_stock by up to ~0.5×(temp−22) | Entire event (static per-event draw, not a transient) | Continuous — every event, drawn from Normal(23,3) |
| Valve wear / stiction (steady-state) | Mechanical wear of the filler valve actuator | `filler_flow` (τ_filler), `basis_weight` (τ_bw) | τ_filler increases by up to 40×`valve_wear_index` seconds; τ_bw by up to 18×`valve_wear_index` seconds | Entire event | Continuous — every event has a `valve_wear_index` draw |
| Valve overshoot event (transient) | Consequence of high `valve_wear_index` | `filler_flow` (sinusoidal bump), `basis_weight` (small cross-coupling bump) | Amplitude ~Normal(0.25, 0.05), truncated positive | Middle third of the ramp window only, when active | Bernoulli(0.5), **conditional on** `valve_wear_index` > 0.6 |
| Valve wear campaign drift | Cumulative wear without a maintenance reset across the logged campaign | `valve_wear_index` itself | Up to +0.3 added linearly as a function of event position in the log | Entire dataset generation run | Deterministic (not randomly sampled) |
| Ash carryover | Residual furnish composition not fully purged between grades | `ash` (initial condition only) | 15% blend weight toward the previous event's destination ash | Transient — decays via `ash`'s own first-order lag over the event | Deterministic — always present |
| Measurement/sensor noise | Normal scanner/sensor noise floor | Every tag | Per-tag σ as specified in §4 | Continuous, every sample | Always present (Gaussian) |

---

## 6. HIDDEN RELATIONSHIPS

These are the relationships intentionally engineered so the correlation-discovery module has real, non-recipe signal to recover. **Strength values below are design intent, not measured — they must be confirmed empirically against the generated dataset once produced, and the confirmed values should be recorded back into this document.**

| ID | Variables Involved | Direction | Design-Intent Strength | Reasoning | Expected Discoverability |
|---|---|---|---|---|---|
| A | `ambient_temp_c` → `stabilization_time_seconds` / `basis_weight_pct_dev` | Positive (hotter → slower settling, higher risk) | Moderate–strong | Raises τ_bw directly | Lagged correlation on the time series; confirmed by feature importance in both models |
| B | `valve_wear_index` → `ash` oscillation and intermittent `basis_weight` bumps | Positive | Moderate | Wear elevates τ_filler/τ_bw and triggers a time-localized overshoot | Requires **windowed/lagged** correlation — whole-event Pearson under-detects this because the bump only occupies the mid-ramp third |
| C | `event_id` (time-in-campaign proxy) → `valve_wear_index` → off-spec rate | Positive | Moderate | Systematic drift term, not noise | Correlation between `event_id` and `valve_wear_index`/off-spec rate — a "campaign degradation" pattern absent from any spec table |
| D | `prior_ash` (previous event's destination ash) → current event's initial `ash` trajectory | Positive | Weak–moderate | Cross-event memory/carryover effect | Requires a **lag-1 join across events**, not a within-event time lag — a distinct analysis method from A–C |
| E | `ambient_temp_c` → `stock_flow` settling | Positive, weak (secondary channel) | Weak | Deliberately subtle secondary effect, distinct from A's strong channel into `basis_weight` | Tests the correlation engine's sensitivity to a genuinely weak signal — should be findable but with lower confidence/significance than A |

---

## 7. LABEL GENERATION

### Classifier Target — `offspec_label`

**Threshold:** ±2.5% of the ramping `basis_weight_setpoint` (the quality threshold from the problem statement).

**Ground truth rule:** For each event, compute `basis_weight_pct_dev` at every one of the 480 timesteps. If `|basis_weight_pct_dev|` exceeds 2.5 at **any** timestep, `offspec_label = 1`; otherwise `0`. This is a deterministic function of the full time series, computed once at generation time in `labels.py` — never re-derived downstream.

### Regression Target — `stabilization_time_seconds`

**Settled-band threshold:** ±1.5% of the ramping setpoint (tighter than the 2.5% quality threshold — mirrors an MPC readiness check), held continuously for **≥60 seconds** (12 consecutive samples).

**Ground truth rule (Fixed & Harmonized with `labels.py`):**
1. Check if `|basis_weight_pct_dev|` ever breaches 1.5% during the transition (`abs_dev > 1.5%`).
2. If the process **never** leaves the ±1.5% band, it was continuously stable — `stabilization_time_seconds = 0.0` and `stabilized_within_window = True`.
3. If a breach occurs, set the scan start index `i` to the timestamp of the **first breach**.
4. From that first breach point forward, find the first timestep where the process enters the ±1.5% band and stays inside for at least 12 consecutive samples (60s).
5. If a qualifying recovery window is found: `stabilization_time_seconds` = `t_seconds` at the start of that window; `stabilized_within_window = True`.
6. If no qualifying window exists within the full 2400s: `stabilization_time_seconds` is **capped at 2400**; `stabilized_within_window = False`.

### `outcome_category` (diagnostic, derived from the two targets above)

- `offspec_label = 0` → `"clean"`
- `offspec_label = 1` and `stabilized_within_window = True` → `"recovered"`
- `offspec_label = 1` and `stabilized_within_window = False` → `"unresolved"`

---

## 8. FEATURE ENGINEERING

### Prediction cutoff mechanism (fills a gap — confirm before implementation)

The product promise is "predict off-spec risk **before it happens**." That requires the model to make its prediction using only data available up to some point *during* the transition — not full-event statistics computed after the fact, which would be leakage against the model's own purpose.

**Frozen choice: `feature_window_seconds = 300`** (5 minutes). Every training example's *actual/measured* features are computed using only `historian_timeseries.csv` rows with `t_seconds ≤ 300` for that event; the label (`offspec_label`, `stabilization_time_seconds`) is still computed from the full 2400s event, per §7.

**Events whose real breach occurs before the cutoff** (`time_to_offspec_s < 300`) are flagged `early_breach = True` and **excluded from the primary training/evaluation set** for the early-warning use case (the model cannot claim to predict something that already happened before its observation window closed). They are retained for the full-event correlation-discovery pass, which is not subject to this restriction. This flag and its exclusion rule live in `features.py`, not in `labels.py` — it is a modeling-time restriction, not a change to the dataset's ground truth.

*Not implemented, by design (avoid unnecessary complexity):* multiple prediction cutoffs (e.g., 120s/300s/600s) generating several training rows per event. Single fixed cutoff is the frozen MVP choice; multi-cutoff is a documented future extension only.

### Input Features

- **Actual/measured features** (windowed to `t_seconds ≤ 300` only): rolling mean/std of each control and quality variable, rate-of-change of `basis_weight` and `basis_weight_pct_dev`, current deviation from setpoint per variable, `valve_wear_index`, `ambient_temp_c` (both legitimately observable — see §6, these are "hidden" from the recipe, not from the historian).
- **Planned/setpoint features** (known in full at `t=0` — **not subject to the windowing restriction**, since Honeywell's own target/trajectory calculation computes the full planned ramp upfront): total planned `basis_weight` change (`to_grade` − `from_grade` setpoint), planned ramp duration, planned ramp rate, and the equivalent for the other three control variables. Using future *setpoint* values is not leakage; using future *actual/measured* values is.
- **Categorical:** `from_grade`, `to_grade`.

### Excluded Features (never fed to Classifier or Regressor)

- `operator_ramp_choice` — simulator ground truth; a real historian would never log an explicit "aggressiveness" flag. Models must learn from the *planned ramp rate* feature instead.
- `max_abs_basis_weight_pct_dev`, `time_to_offspec_s`, `outcome_category`, `stabilized_within_window` — all are post-hoc, full-event diagnostics computed from data the model wouldn't have at prediction time.
- Any *actual/measured* value from `t_seconds > 300` for the corresponding event.
- The underlying physical time constants (τ_stock, τ_filler, τ_steam, τ_speed, τ_bw) and the raw overshoot-activation flag/amplitude — these are simulator-internal mechanics, not observable in real life under any circumstance, real or synthetic. Only their observable proxies (`valve_wear_index`, `ambient_temp_c`) may be used.

### Leakage Prevention Summary

Two distinct rules, not one: (1) *actual* process values are windowed to the prediction cutoff; (2) *planned* setpoint trajectories are exempt from windowing because they are genuinely known in advance by the existing trajectory-calculation system. Conflating these two would either cripple the model (over-restricting known setpoints) or leak the future (under-restricting actual values) — both are real mistakes worth guarding against explicitly.

### Scaling

**None required for the Classifier/Regressor.** Both are gradient-boosted tree models (frozen choice, §2 of the architecture discussion) — scale-invariant by construction, and applying scaling would be unnecessary complexity.

**Standardization (zero mean, unit variance) is required** for the feature set used by the Hybrid Recommendation Engine's k-NN historical-similarity search in `recommend.py`, since Euclidean distance is scale-sensitive. This is a separate feature set/consumer from the trained models and must be standardized independently.

### Encoding

`from_grade` and `to_grade` are **one-hot encoded** (5 categories each → 10 binary columns). Target encoding is explicitly avoided — with only ~500 events across 20 grade pairs, target-encoding these categoricals risks overfitting/leakage; one-hot is simpler and sufficient at this scale. Learned embeddings are unnecessary complexity for 5 categories.

---

## 9. DATA QUALITY CHECKS

**Structural validation (run after every `generate_data.py` execution):**
- Every `event_id` has exactly 480 rows in `historian_timeseries.csv` (2400s ÷ 5s).
- `t_seconds` is strictly increasing, no gaps or duplicates, 0 to 2395 in steps of 5, per event.
- Every `event_id` in `historian_timeseries.csv` has exactly one matching row in `grade_change_events.csv`, and vice versa — no orphans.
- `from_grade` ≠ `to_grade` for every event.
- Every `from_grade`/`to_grade` value exists in `recipe_reference.csv`.

**Missing values:** No `NaN` is permitted anywhere except by design: `time_to_offspec_s` is null for events with `offspec_label = 0`. `stabilization_time_seconds` is **never** null — it is capped at 2400, not left missing (see §7). Any other `NaN` indicates a generation bug, not expected data.

**Consistency checks:**
- `offspec_label == 1` if and only if `max(|basis_weight_pct_dev|)` over the event exceeds 2.5 — recompute independently at validation time and cross-check against the stored label.
- `stabilized_within_window == False` if and only if `stabilization_time_seconds == 2400`.
- `outcome_category` matches the derivation rule in §7 exactly, for every row.

**Expected distributions (sanity ranges, not hard failures — investigate if violated):**
- `offspec_label` positive rate: 35–45%.
- `valve_wear_index` vs. `event_id`: positive correlation expected (campaign drift, §6 relationship C).
- `ambient_temp_c`: approximately Normal(23, 3), truncated to [15, 35].
- Grade-pair coverage: every one of the 20 ordered pairs should have at least 5 examples at the 500-event volume — fewer would break the k-NN historical-similarity search's `k=5` assumption in `recommend.py`.

**Range/anomaly bounds** (looser than §4's "normal operating range," used to catch generation bugs rather than flag ordinary transient excursions): `basis_weight` outside [30, 115] gsm, `moisture` outside [4, 10]%, `ash` outside [1, 16]%, or `caliper` outside [40, 135] µm should be treated as a likely bug, not a real disturbance, and investigated before the dataset is used downstream.

---

## 10. FROZEN DATASET RULES

1. **Never rename a column** in any of the three CSVs without updating this document in the same change, and propagating the rename to every consumer (`data_loader.py`, `features.py`, `correlation_discovery.py`, `train_model.py`, `recommend.py`, `app.py`).
2. **Never alter target definitions** — the 2.5% off-spec threshold, the ±1.5%/60-second settled-band rule, or the 2400s cap — without updating both this document and `data/labels.py` together.
3. **Never change event duration (2400s) or sampling frequency (5s)** without regenerating and revalidating the entire dataset, and updating every downstream module that assumes 480 rows/event.
4. **Never leak simulator ground truth into model inputs** — `operator_ramp_choice` (or any future rename) is permanently excluded, per §8.
5. **Never use hidden simulator mechanics as model inputs.** Only the observable proxies `valve_wear_index` and `ambient_temp_c` may be used — never the underlying τ values, disturbance-activation flags, or overshoot amplitudes.
6. **Never use full-event diagnostic columns as model features** (`max_abs_basis_weight_pct_dev`, `time_to_offspec_s`, `outcome_category`, `stabilized_within_window`) — they are computed from the future relative to any prediction point.
7. **Never use actual/measured values beyond `feature_window_seconds` (300s)** as model input features — only planned/setpoint trajectories are exempt from this restriction, per §8.
8. **Never change the grade catalog** (the 5 grades and their recipe values) without updating `recipe_reference.csv` and re-deriving every recipe-implied setpoint formula consistently.
9. **Never treat `stabilization_time_seconds` as uncensored** in any reported metric, model evaluation, or dashboard claim without acknowledging the 2400s capping simplification.
10. **Never regenerate the "final" 500-event dataset with a different random seed** without re-running and revalidating the full downstream pipeline (models, correlation findings, dashboard) — the frozen seed is `42`.
11. **Never bypass `src/data_loader.py`** — no module other than `data/generate_data.py` writes these CSVs, and no module other than `data_loader.py` reads them directly; this is what keeps real-data substitution cheap.
