"""
correlation_discovery.py — Both correlation passes (statistical + ML-validated).

Pass 1 (statistical): runs independently of any trained model, right after the
data generator. Tests for the five engineered Hidden Relationships (A-E) plus
negative controls, using Pearson/Spearman and windowed/lagged methods where
the relationship specifically requires them (see DATASET_SPECIFICATION.md §6).

Pass 2 (ML-validated): MUST run after the Classifier and Regressor are trained
(PROJECT_CONSTITUTION.md §6 rule 9). Cross-checks Pass 1's statistical findings
against native feature importances from the saved models, tagging each finding
"statistically found" or "statistically found + ML-confirmed".

IMPORTANT CAVEAT (documented, not hidden): Pass 1 operates on the FULL event
history (historian_timeseries.csv, unrestricted — correlation discovery is
explicitly exempt from the 300s feature-window restriction, §8). Pass 2,
however, can only confirm a relationship to the extent it is reflected in the
Feature Engineering module's windowed (t<=300s) feature set, because that is
the only thing the trained models ever see. Relationship E in particular
(ambient_temp_c -> stock_flow settling, measured in the LATE window t>1000s)
has no faithful windowed analogue in the model's features — its ML
"confirmation" below is therefore approximate and flagged as such rather than
overclaimed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from src.data_loader import load_events, load_timeseries

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
CLASSIFIER_PATH = MODELS_DIR / "offspec_classifier.pkl"
REGRESSOR_PATH = MODELS_DIR / "stabilization_regressor.pkl"

# Top-K cutoff used to decide whether a proxy feature counts as "important"
# enough for a relationship to be considered ML-confirmed.
TOP_K_IMPORTANCE: int = 10


# ---------------------------------------------------------------------------
# Pass 1 — Statistical
# ---------------------------------------------------------------------------

def _midramp_window(ts: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Restrict the timeseries to each event's middle-third ramp window.

    Relationship B's overshoot bump is only active in the middle third of the
    ramp (see disturbances.py / transitions.py), so a whole-event correlation
    under-detects it. This helper builds the windowed subset needed to test it
    properly, per DATASET_SPECIFICATION.md §6.

    Args:
        ts: Full historian_timeseries.csv.
        events: grade_change_events.csv (for ramp_duration_seconds per event).

    Returns:
        Timeseries rows falling inside each event's own mid-ramp third.
    """
    merged = ts.merge(
        events[["event_id", "ramp_duration_seconds"]], on="event_id", how="left"
    )
    window_start = merged["ramp_duration_seconds"] / 3.0
    window_end = 2.0 * merged["ramp_duration_seconds"] / 3.0
    mask = (merged["t_seconds"] >= window_start) & (merged["t_seconds"] <= window_end)
    return merged[mask]


def _relationship_b(events: pd.DataFrame, ts: pd.DataFrame) -> list[dict]:
    """Test relationship B using both a naive whole-event and a windowed method.

    Demonstrates the exact failure mode the spec calls out: whole-event Pearson
    under-detects this relationship because the bump only occupies the
    mid-ramp third; the windowed version should find it.

    Args:
        events: grade_change_events.csv.
        ts: historian_timeseries.csv.

    Returns:
        List of finding dicts (whole-event and windowed variants).
    """
    findings = []

    # --- Whole-event (naive) baseline ---
    ash_std_whole = ts.groupby("event_id")["ash"].std().rename("ash_std_whole")
    bw_maxdev_whole = (
        ts.groupby("event_id")["basis_weight_pct_dev"].apply(lambda s: s.abs().max())
        .rename("bw_maxdev_whole")
    )
    merged_whole = events.join(ash_std_whole, on="event_id").join(bw_maxdev_whole, on="event_id")

    r_ash_whole, p_ash_whole = stats.pearsonr(merged_whole["valve_wear_index"], merged_whole["ash_std_whole"])
    findings.append({
        "Relationship": "B (Wear -> Ash Oscillation) [whole-event]",
        "Method": "Pearson (naive, full event)",
        "Coefficient": r_ash_whole, "P-Value": p_ash_whole, "Found": p_ash_whole < 0.05,
    })

    r_bw_whole, p_bw_whole = stats.pearsonr(merged_whole["valve_wear_index"], merged_whole["bw_maxdev_whole"])
    findings.append({
        "Relationship": "B (Wear -> BW Bump) [whole-event]",
        "Method": "Pearson (naive, full event)",
        "Coefficient": r_bw_whole, "P-Value": p_bw_whole, "Found": p_bw_whole < 0.05,
    })

    # --- Windowed (mid-ramp third) — the method the spec says is required ---
    mid = _midramp_window(ts, events)
    ash_std_mid = mid.groupby("event_id")["ash"].std().rename("ash_std_mid")
    bw_maxdev_mid = (
        mid.groupby("event_id")["basis_weight_pct_dev"].apply(lambda s: s.abs().max())
        .rename("bw_maxdev_mid")
    )
    merged_mid = events.join(ash_std_mid, on="event_id").join(bw_maxdev_mid, on="event_id").dropna(
        subset=["ash_std_mid", "bw_maxdev_mid"]
    )

    r_ash_mid, p_ash_mid = stats.pearsonr(merged_mid["valve_wear_index"], merged_mid["ash_std_mid"])
    findings.append({
        "Relationship": "B (Wear -> Ash Oscillation) [windowed mid-ramp]",
        "Method": "Pearson (windowed, mid-ramp third)",
        "Coefficient": r_ash_mid, "P-Value": p_ash_mid, "Found": p_ash_mid < 0.05,
    })

    r_bw_mid, p_bw_mid = stats.pearsonr(merged_mid["valve_wear_index"], merged_mid["bw_maxdev_mid"])
    findings.append({
        "Relationship": "B (Wear -> BW Bump) [windowed mid-ramp]",
        "Method": "Pearson (windowed, mid-ramp third)",
        "Coefficient": r_bw_mid, "P-Value": p_bw_mid, "Found": p_bw_mid < 0.05,
    })

    # --- Filler flow (fixed: was never tested directly, despite being the
    # FIRST/primary disturbed variable per disturbances.py's own docstring —
    # "valve wear elevates filler_flow lag (drives ash carryover)". Testing
    # ash alone was one mechanistic step removed from the actual disturbance. ---
    ff_dev_whole = (
        ts.assign(ff_abs_dev=(ts["filler_flow"] - ts["filler_flow_setpoint"]).abs())
        .groupby("event_id")["ff_abs_dev"].max().rename("ff_maxdev_whole")
    )
    merged_ff_whole = events.join(ff_dev_whole, on="event_id")
    r_ff_whole, p_ff_whole = stats.pearsonr(merged_ff_whole["valve_wear_index"], merged_ff_whole["ff_maxdev_whole"])
    findings.append({
        "Relationship": "B (Wear -> Filler Flow Deviation) [whole-event]",
        "Method": "Pearson (naive, full event)",
        "Coefficient": r_ff_whole, "P-Value": p_ff_whole, "Found": p_ff_whole < 0.05,
    })

    mid_ff = mid.assign(ff_abs_dev=(mid["filler_flow"] - mid["filler_flow_setpoint"]).abs())
    ff_dev_mid = mid_ff.groupby("event_id")["ff_abs_dev"].max().rename("ff_maxdev_mid")
    merged_ff_mid = events.join(ff_dev_mid, on="event_id").dropna(subset=["ff_maxdev_mid"])
    r_ff_mid, p_ff_mid = stats.pearsonr(merged_ff_mid["valve_wear_index"], merged_ff_mid["ff_maxdev_mid"])
    findings.append({
        "Relationship": "B (Wear -> Filler Flow Deviation) [windowed mid-ramp]",
        "Method": "Pearson (windowed, mid-ramp third)",
        "Coefficient": r_ff_mid, "P-Value": p_ff_mid, "Found": p_ff_mid < 0.05,
    })

    return findings


def _negative_controls(events: pd.DataFrame, ts: pd.DataFrame) -> list[dict]:
    """Test all combinations of {valve_wear_index, ambient_temp_c} x the four
    deliberately-clean variables (steam_pressure, machine_speed, caliper,
    moisture).

    None of these should show significant correlation — a correlation-discovery
    method that reports spurious findings here is a signal the method itself is
    flawed (DATASET_SPECIFICATION.md §4).

    Fixed: moisture was the one variable named in the hackathon problem
    statement's constraints list ("Stock flow, filler flow, steam pressure,
    machine speed, moisture, ash, caliper, recipe limits") that was never
    tested at all — not as a relationship, not even as a negative control,
    despite tau_moisture never being modified by any hidden variable in the
    simulator (see disturbances.py — it's exactly as "clean" as the other three).

    Args:
        events: grade_change_events.csv.
        ts: historian_timeseries.csv.

    Returns:
        List of finding dicts, one per (hidden_var, clean_var) combination.
    """
    findings = []
    hidden_vars = ["valve_wear_index", "ambient_temp_c"]
    clean_vars = ["steam_pressure", "machine_speed", "caliper", "moisture"]

    means = ts.groupby("event_id")[clean_vars].mean()
    merged = events.join(means, on="event_id")

    for hv in hidden_vars:
        for cv in clean_vars:
            r, p = stats.pearsonr(merged[hv], merged[cv])
            findings.append({
                "Relationship": f"Negative Control ({hv} -> {cv})",
                "Method": "Pearson",
                "Coefficient": r, "P-Value": p,
                "Found": p < 0.05,  # expected False for every row here
            })
    return findings


def run_statistical_pass() -> pd.DataFrame:
    """Run Pass 1 of Correlation Discovery (statistical correlations).

    Tests all five engineered Hidden Relationships (A-E) plus a full set of
    negative controls, using windowed/lagged methods where the relationship
    specifically requires them.

    Returns:
        pd.DataFrame: A table of findings, one row per statistical test.
    """
    events = load_events()
    ts = load_timeseries()

    findings: list[dict] = []

    # --- Relationship A ---
    settled_events = events[events["stabilized_within_window"] == True]
    r_a_stab, p_a_stab = stats.pearsonr(settled_events["ambient_temp_c"], settled_events["stabilization_time_seconds"])
    r_a_off, p_a_off = stats.pearsonr(events["ambient_temp_c"], events["offspec_label"])
    findings.append({"Relationship": "A (Temp -> Stabilization)", "Method": "Pearson (subset: stabilized)",
                      "Coefficient": r_a_stab, "P-Value": p_a_stab, "Found": p_a_stab < 0.05})
    findings.append({"Relationship": "A (Temp -> Offspec Risk)", "Method": "Pearson (all events)",
                      "Coefficient": r_a_off, "P-Value": p_a_off, "Found": p_a_off < 0.05})

    # --- Relationship B (fixed: was entirely missing) ---
    findings.extend(_relationship_b(events, ts))

    # --- Relationship C (fixed: off-spec-rate half was computed but dropped) ---
    r_c_wear, p_c_wear = stats.pearsonr(events["event_id"], events["valve_wear_index"])
    r_c_off, p_c_off = stats.spearmanr(events["event_id"], events["offspec_label"])
    findings.append({"Relationship": "C (Time in Campaign -> Valve Wear)", "Method": "Pearson",
                      "Coefficient": r_c_wear, "P-Value": p_c_wear, "Found": p_c_wear < 0.05})
    findings.append({"Relationship": "C (Time in Campaign -> Offspec Rate)", "Method": "Spearman",
                      "Coefficient": r_c_off, "P-Value": p_c_off, "Found": p_c_off < 0.05})

    # --- Relationship D ---
    ts_t0 = ts[ts["t_seconds"] == 0].set_index("event_id")
    events_with_t0 = events.join(ts_t0[["ash"]], on="event_id", rsuffix="_t0")
    r_d, p_d = stats.pearsonr(events_with_t0["prior_ash"], events_with_t0["ash"])
    findings.append({"Relationship": "D (Prior Ash -> Initial Ash)", "Method": "Pearson (Cross-event Lag-1)",
                      "Coefficient": r_d, "P-Value": p_d, "Found": p_d < 0.05})

    # --- Relationship E ---
    ts_late = ts[ts["t_seconds"] > 1000].groupby("event_id")["stock_flow"].std()
    events_sf = events.join(ts_late.rename("sf_late_std"), on="event_id").dropna(subset=["sf_late_std"])
    r_e, p_e = stats.pearsonr(events_sf["ambient_temp_c"], events_sf["sf_late_std"])
    findings.append({"Relationship": "E (Temp -> Stock Flow Settling)", "Method": "Pearson (Temp vs Late-Window StdDev)",
                      "Coefficient": r_e, "P-Value": p_e, "Found": p_e < 0.05})

    # --- Negative controls (fixed: was only 1 of 6 combinations) ---
    findings.extend(_negative_controls(events, ts))

    return pd.DataFrame(findings)


# ---------------------------------------------------------------------------
# Pass 2 — ML-validated
# ---------------------------------------------------------------------------

# Maps each Pass-1 relationship to the model feature(s) that would carry its
# signal IF the trained models could see it. Caveat column documents where
# the correspondence between pass-1's full-event signal and the model's
# 300s-windowed features is imperfect (see module docstring).
_RELATIONSHIP_FEATURE_MAP: dict[str, dict] = {
    "A": {"features": ["ambient_temp_c"], "caveat": None},
    "B": {"features": ["ash_std_300s", "filler_flow_std_300s"],
          "caveat": "Proxy only: model sees ash/filler_flow variability in the "
                    "FIRST 300s, not the mid-ramp-third window Pass 1 used."},
    "C": {"features": ["valve_wear_index"],
          "caveat": "event_id itself is not a model feature (correctly excluded "
                    "as a leak-prone id); valve_wear_index is used as the proxy "
                    "for the underlying campaign-drift signal."},
    "D": {"features": ["ash_mean_300s"],
          "caveat": "Proxy only: reflects blended initial ash, not prior_ash directly."},
    "E": {"features": ["stock_flow_std_300s"],
          "caveat": "WEAK CORRESPONDENCE: Pass 1 measures LATE-window (t>1000s) "
                    "settling; the model only ever sees the first 300s, so this "
                    "confirmation is approximate at best."},
}


def _load_importances(model_path: Path) -> pd.Series | None:
    """Load a trained model and return its feature importances as a Series.

    Args:
        model_path: Path to a joblib-serialized sklearn model.

    Returns:
        Series indexed by feature name, or None if the model file is missing.
    """
    if not model_path.exists():
        logger.warning("Model not found at %s — run train_model.py first.", model_path)
        return None
    model = joblib.load(model_path)
    return pd.Series(model.feature_importances_, index=model.feature_names_in_)


def run_ml_validation_pass(pass1_findings: pd.DataFrame) -> pd.DataFrame:
    """Run Pass 2: cross-check Pass 1 findings against trained model importances.

    Must only be called after both models are trained and saved
    (PROJECT_CONSTITUTION.md §6 rule 9).

    Args:
        pass1_findings: Output of run_statistical_pass().

    Returns:
        pass1_findings with three added columns: ML_Confirmed (bool/None),
        Caveat (str/None), and Tag (the required "statistically found" /
        "statistically found + ML-confirmed" label).
    """
    clf_importances = _load_importances(CLASSIFIER_PATH)
    reg_importances = _load_importances(REGRESSOR_PATH)

    df = pass1_findings.copy()
    df["ML_Confirmed"] = None
    df["Caveat"] = None

    for letter, spec in _RELATIONSHIP_FEATURE_MAP.items():
        row_mask = df["Relationship"].str.startswith(letter + " ")
        if not row_mask.any():
            continue

        confirmed = False
        for feat in spec["features"]:
            for importances in (clf_importances, reg_importances):
                if importances is None or feat not in importances.index:
                    continue
                top_k = importances.sort_values(ascending=False).head(TOP_K_IMPORTANCE).index
                if feat in top_k:
                    confirmed = True

        df.loc[row_mask, "ML_Confirmed"] = confirmed
        df.loc[row_mask, "Caveat"] = spec["caveat"]

    def _tag(row: pd.Series) -> str:
        if not row["Found"]:
            return "not statistically found"
        if row["ML_Confirmed"] is True:
            return "statistically found + ML-confirmed"
        if row["ML_Confirmed"] is False:
            return "statistically found (not ML-confirmed)"
        return "statistically found (no ML cross-check defined)"

    df["Tag"] = df.apply(_tag, axis=1)
    return df


def run_full_discovery() -> pd.DataFrame:
    """Single entry point for dashboard/app.py: runs Pass 1 + Pass 2 and adds
    a dashboard-ready 'Status' column.

    IMPORTANT (bug fix): a negative control that finds NO correlation is the
    CORRECT/expected outcome — good news, not a miss. It is tagged distinctly
    from a real relationship (A-E) that simply wasn't found, so the dashboard
    doesn't display a successful negative control as if it were a failure.

    Returns:
        DataFrame with all Pass 1/2 columns plus a 'Status' column.
    """
    pass1 = run_statistical_pass()
    combined = run_ml_validation_pass(pass1)

    def _status(row: pd.Series) -> str:
        is_negative_control = row["Relationship"].startswith("Negative Control")
        if is_negative_control:
            return (
                "Negative control (expected: not found)" if not row["Found"]
                else "Negative control FAILED (unexpected correlation)"
            )
        if not row["Found"]:
            return "Not found (weak signal at this data scale)"
        if row["ML_Confirmed"] is True:
            return "Statistically found + ML-confirmed"
        if row["ML_Confirmed"] is False:
            return "Statistically found (not ML-confirmed)"
        return "Statistically found (no ML cross-check defined)"

    combined["Status"] = combined.apply(_status, axis=1)
    return combined


def run_open_ended_scan(alpha: float = 0.01) -> pd.DataFrame:
    """Generic, undirected correlation scan — the literal "find new correlations
    not defined in the system" requirement from the hackathon problem statement.

    Distinct from everything above: run_statistical_pass() only tests the FIVE
    relationships (A-E) that were deliberately engineered into the simulator —
    it's a confirmation engine, not a discovery engine. This function instead
    builds a broad set of event-level summary features (mean/std of every
    process/quality variable, plus the known hidden variables and planned
    trajectory parameters) and tests ALL of them against both ML targets,
    without presupposing which ones matter.

    HONESTY NOTE: running many pairwise tests means some will show p < 0.05
    by chance alone even under the null hypothesis (multiple-comparisons
    problem). This uses a stricter alpha=0.01 by default and reports the total
    number of tests run in the returned DataFrame's .attrs, so the expected
    false-positive count is visible rather than silently presenting every hit
    as a confirmed discovery. Anything surfaced here should be treated as an
    exploratory lead worth a follow-up look, not a validated finding the way
    A-E (cross-checked against the trained models in Pass 2) are.

    Args:
        alpha: Significance threshold, stricter than the 0.05 used elsewhere
            specifically because this function runs many more tests.

    Returns:
        DataFrame of (Variable, Target, Coefficient, P-Value) rows that cleared
        the threshold, sorted by p-value. Empty if nothing cleared it. The
        `.attrs["n_tests_run"]` and `.attrs["alpha"]` fields record how many
        tests were run, for transparency about false-positive risk.
    """
    events = load_events()
    ts = load_timeseries()

    raw_vars = ["stock_flow", "filler_flow", "steam_pressure", "machine_speed",
                "moisture", "ash", "caliper", "basis_weight"]
    ts_summary = ts.groupby("event_id")[raw_vars].agg(["mean", "std"])
    ts_summary.columns = [f"{col}_{stat}" for col, stat in ts_summary.columns]

    merged = events.set_index("event_id").join(ts_summary)

    candidate_cols = list(ts_summary.columns) + [
        "valve_wear_index", "ambient_temp_c", "ramp_duration_seconds", "prior_ash"
    ]
    target_cols = ["offspec_label", "stabilization_time_seconds"]

    findings = []
    n_tests = 0
    for col in candidate_cols:
        for target in target_cols:
            if merged[col].nunique() < 2:
                continue
            r, p = stats.pearsonr(merged[col], merged[target])
            n_tests += 1
            if p < alpha:
                findings.append({
                    "Variable": col, "Target": target,
                    "Coefficient": round(r, 4), "P-Value": p,
                })

    result = pd.DataFrame(findings).sort_values("P-Value") if findings else pd.DataFrame(
        columns=["Variable", "Target", "Coefficient", "P-Value"]
    )
    result.attrs["n_tests_run"] = n_tests
    result.attrs["alpha"] = alpha
    return result


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    full = run_full_discovery()
    print("=== Correlation Discovery: Full (Pass 1 + Pass 2) ===")
    print(full[["Relationship", "Method", "Coefficient", "P-Value", "Status", "Caveat"]].round(4).to_string(index=False))

    print("\n=== Open-Ended Scan (genuinely undirected — not one of the 5 engineered relationships) ===")
    scan = run_open_ended_scan()
    print(f"Tests run: {scan.attrs['n_tests_run']}  |  alpha={scan.attrs['alpha']}  |  "
          f"expected false positives by chance alone: ~{scan.attrs['n_tests_run'] * scan.attrs['alpha']:.1f}")
    if scan.empty:
        print("No variable cleared the threshold beyond the 5 engineered relationships.")
    else:
        print(scan.to_string(index=False))