"""
labels.py — offspec_label, stabilization_time_seconds, and outcome_category
derivation.

Takes the per-event time-series DataFrame (480 rows) produced by transitions.py
and computes the two ML targets and one diagnostic column, exactly as defined
in DATASET_SPECIFICATION.md §7.

IMPORTANT: These definitions are frozen.  Do NOT modify without updating
DATASET_SPECIFICATION.md and the non-negotiable rules in PROJECT_CONSTITUTION.md.

Targets:
  - offspec_label (int, 0/1):
      1 iff |basis_weight_pct_dev| > 2.5 at any sample.
  - stabilization_time_seconds (float, 0–2400):
      First t at which process enters ±1.5% band AND stays for ≥60s (12 samples).
      Capped at 2400 if never achieved.

Diagnostics (never model features):
  - time_to_offspec_s (float, nullable): First t at which threshold is breached.
  - max_abs_basis_weight_pct_dev (float): Max absolute deviation observed.
  - stabilized_within_window (bool): False iff stabilization_time_seconds == 2400.
  - outcome_category (str): 'clean' | 'recovered' | 'unresolved'.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen threshold constants (non-negotiable per constitution §6, rule 8 and 15)
# ---------------------------------------------------------------------------
OFFSPEC_THRESHOLD_PCT: float = 2.5
SETTLED_BAND_PCT: float = 1.5
SETTLED_DURATION_S: int = 60
SETTLED_MIN_SAMPLES: int = SETTLED_DURATION_S // 5  # 12 consecutive samples
EVENT_DURATION_S: int = 2400


def compute_offspec_label(pct_dev_series: pd.Series) -> int:
    """Return 1 if absolute deviation exceeds OFFSPEC_THRESHOLD_PCT at any point.

    Args:
        pct_dev_series: basis_weight_pct_dev column for a single event (480 values).

    Returns:
        1 if off-spec threshold breached, 0 otherwise.
    """
    return int((pct_dev_series.abs() > OFFSPEC_THRESHOLD_PCT).any())


def compute_time_to_offspec(
    t_seconds_series: pd.Series,
    pct_dev_series: pd.Series,
) -> float | None:
    """Return the first t_seconds at which off-spec threshold is breached.

    Args:
        t_seconds_series: Elapsed-time column for a single event.
        pct_dev_series: basis_weight_pct_dev column for a single event.

    Returns:
        First breach time in seconds, or None if threshold never breached.
    """
    breach_mask = pct_dev_series.abs() > OFFSPEC_THRESHOLD_PCT
    if not breach_mask.any():
        return None
    first_breach_idx = breach_mask.idxmax()
    return float(t_seconds_series.loc[first_breach_idx])


def compute_stabilization_time(
    t_seconds_series: pd.Series,
    pct_dev_series: pd.Series,
) -> tuple[float, bool]:
    """Find the first t at which the process RE-enters ±1.5% band for ≥60s.

    DEVIATION FROM ORIGINAL DATASET_SPECIFICATION.md §7 (documented, not silent
    — see PROGRESS.md "Bug #1"): the original rule scanned from t=0 with no
    floor, which is trivially satisfied by nearly every event because the
    process starts *exactly* at its own source-grade setpoint at t=0 (zero
    deviation before the ramp/disturbance has had any chance to act). That
    made stabilization_time_seconds ≈ 0 for ~80% of events regardless of
    whether they later went off-spec, which is a degenerate regression target.

    Fixed definition: if the process never leaves the ±1.5% band at all
    during the event, it never needed to "stabilize" — return 0.0 (still
    trivially true, and now correctly rare/expected for genuinely clean
    transitions rather than the default outcome for everything). Otherwise,
    scanning for the qualifying 60s settled window starts at the first
    breach of the ±1.5% band, so the metric measures *recovery time after
    a real excursion* — which is what "reduce stabilization time" means in
    the product pitch.

    Args:
        t_seconds_series: Elapsed-time column for a single event (0..2395, step 5).
        pct_dev_series: basis_weight_pct_dev column for a single event.

    Returns:
        Tuple of (stabilization_time_seconds, stabilized_within_window).
        stabilization_time_seconds is capped at EVENT_DURATION_S (2400) if
        no qualifying window is found after the first breach.
    """
    abs_dev = pct_dev_series.abs().values
    t_vals = t_seconds_series.values
    n = len(abs_dev)

    breach_mask = abs_dev > SETTLED_BAND_PCT
    if not breach_mask.any():
        # Process never left the settled band — nothing to recover from.
        return 0.0, True

    i = int(np.argmax(breach_mask))  # start scanning at the first real breach
    while i < n:
        # Skip until the process enters the settled band
        if abs_dev[i] > SETTLED_BAND_PCT:
            i += 1
            continue
        # Found an entry point — check if the next SETTLED_MIN_SAMPLES-1 are also in band
        entry_i = i
        end_i = min(i + SETTLED_MIN_SAMPLES, n)
        window = abs_dev[entry_i:end_i]
        if len(window) < SETTLED_MIN_SAMPLES:
            # Not enough samples left in the event to confirm settling
            break
        if np.all(window <= SETTLED_BAND_PCT):
            # Qualifying window found
            return float(t_vals[entry_i]), True
        # Band was exited before 60s — advance to the first out-of-band sample
        # within the checked window so we don't redundantly re-check
        exit_offset = int(np.argmax(window > SETTLED_BAND_PCT))
        i = entry_i + exit_offset + 1

    return float(EVENT_DURATION_S), False


def derive_outcome_category(offspec_label: int, stabilized: bool) -> str:
    """Derive human-readable outcome category from the two ML targets.

    Args:
        offspec_label: 0 or 1 — classifier target.
        stabilized: True iff stabilized_within_window.

    Returns:
        One of 'clean', 'recovered', 'unresolved'.
    """
    if offspec_label == 0:
        return "clean"
    return "recovered" if stabilized else "unresolved"


def compute_event_labels(ts_df: pd.DataFrame) -> dict:
    """Compute all label and diagnostic columns for one grade-change event.

    Args:
        ts_df: 480-row DataFrame for a single event (from transitions.py),
            containing at minimum 't_seconds' and 'basis_weight_pct_dev'.

    Returns:
        Dict with keys: offspec_label, time_to_offspec_s,
        stabilization_time_seconds, stabilized_within_window,
        max_abs_basis_weight_pct_dev, outcome_category.
    """
    pct_dev = ts_df["basis_weight_pct_dev"]
    t_s = ts_df["t_seconds"]

    offspec = compute_offspec_label(pct_dev)
    t_to_offspec = compute_time_to_offspec(t_s, pct_dev)
    stab_time, stabilized = compute_stabilization_time(t_s, pct_dev)
    outcome = derive_outcome_category(offspec, stabilized)
    max_dev = float(pct_dev.abs().max())

    return {
        "offspec_label": offspec,
        "time_to_offspec_s": t_to_offspec,
        "stabilization_time_seconds": stab_time,
        "stabilized_within_window": stabilized,
        "max_abs_basis_weight_pct_dev": round(max_dev, 6),
        "outcome_category": outcome,
    }