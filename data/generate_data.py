"""
generate_data.py — Orchestrator: wires recipes / disturbances / transitions /
labels together and writes the three canonical CSVs.

Usage:
    python -m data.generate_data [--events 250] [--seed 42] [--output outputs/]

Frozen defaults per PROJECT_CONSTITUTION.md:
  - Development set: 250 events
  - Final set: 500 events (only after full pipeline validation — Rule 10)
  - Random seed: 42

Build order: this file is the LAST module in data/, and the ONLY module that
calls all four siblings. It must not be imported by any module outside data/.
Downstream consumers use src/data_loader.py exclusively.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Project-relative imports — run as: python -m data.generate_data from repo root
from data.recipes import export_recipe_reference, GRADES, get_setpoints_for_grade
from data.disturbances import make_event_disturbances
from data.transitions import (
    simulate_event,
    sample_grade_pair,
    sample_ramp_duration,
    EVENT_DURATION_S,
    N_SAMPLES,
    SAMPLE_INTERVAL_S,
)
from data.labels import compute_event_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data quality checks — §9 of DATASET_SPECIFICATION.md
# ---------------------------------------------------------------------------

def validate_timeseries(ts_df: pd.DataFrame, n_events: int) -> None:
    """Run structural and consistency checks on historian_timeseries.csv.

    Raises:
        ValueError: On any structural violation.
        AssertionError: On any consistency violation.
    """
    logger.info("Running timeseries validation …")
    expected_rows = n_events * N_SAMPLES

    if len(ts_df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} timeseries rows, found {len(ts_df)}"
        )

    for eid, grp in ts_df.groupby("event_id"):
        if len(grp) != N_SAMPLES:
            raise ValueError(
                f"Event {eid} has {len(grp)} rows; expected {N_SAMPLES}"
            )
        expected_t = list(range(0, EVENT_DURATION_S, SAMPLE_INTERVAL_S))
        actual_t = sorted(grp["t_seconds"].tolist())
        if actual_t != expected_t:
            raise ValueError(
                f"Event {eid} has unexpected t_seconds sequence"
            )

    logger.info("  ✓ Row counts and t_seconds sequences OK")


def validate_events(events_df: pd.DataFrame, ts_df: pd.DataFrame) -> None:
    """Cross-validate event and timeseries tables."""
    logger.info("Running events-vs-timeseries cross-validation …")

    ts_event_ids = set(ts_df["event_id"].unique())
    ev_event_ids = set(events_df["event_id"].unique())

    orphan_ts = ts_event_ids - ev_event_ids
    orphan_ev = ev_event_ids - ts_event_ids
    if orphan_ts:
        raise ValueError(f"Timeseries event_ids not in events table: {orphan_ts}")
    if orphan_ev:
        raise ValueError(f"Events table event_ids not in timeseries: {orphan_ev}")

    # from_grade != to_grade
    bad = events_df[events_df["from_grade"] == events_df["to_grade"]]
    if len(bad):
        raise ValueError(f"Events with from_grade == to_grade: {bad['event_id'].tolist()}")

    # NaN checks — only time_to_offspec_s may be null (when offspec_label == 0)
    for col in events_df.columns:
        if col == "time_to_offspec_s":
            continue
        null_count = events_df[col].isna().sum()
        if null_count:
            raise ValueError(f"Unexpected NaN in events_df['{col}']: {null_count} nulls")

    logger.info("  ✓ Cross-validation OK")


def validate_consistency(events_df: pd.DataFrame, ts_df: pd.DataFrame) -> None:
    """Check label consistency between event rows and derived-from time series."""
    logger.info("Running label consistency checks …")

    for _, row in events_df.iterrows():
        eid = row["event_id"]
        grp = ts_df[ts_df["event_id"] == eid]
        max_dev = grp["basis_weight_pct_dev"].abs().max()
        expected_label = int(max_dev > 2.5)
        if row["offspec_label"] != expected_label:
            raise ValueError(
                f"Event {eid}: stored offspec_label={row['offspec_label']} "
                f"but recomputed={expected_label} (max_dev={max_dev:.3f}%)"
            )

    # stabilized_within_window ↔ stabilization_time == 2400
    wrong = events_df[
        (events_df["stabilized_within_window"]) &
        (events_df["stabilization_time_seconds"] == 2400.0)
    ]
    if len(wrong):
        raise ValueError(
            f"{len(wrong)} events claim stabilized=True but stabilization_time=2400"
        )

    logger.info("  ✓ Label consistency OK")


def print_distribution_summary(events_df: pd.DataFrame) -> None:
    """Log distribution statistics for sanity checks (§9 expected distributions)."""
    logger.info("=== Distribution Summary ===")

    offspec_rate = events_df["offspec_label"].mean() * 100
    logger.info("  offspec_label positive rate: %.1f%%  (expected 35–45%%)", offspec_rate)
    if not (30.0 <= offspec_rate <= 55.0):
        logger.warning("  ⚠ offspec rate %.1f%% outside 30–55%% — investigate", offspec_rate)

    temp_mean = events_df["ambient_temp_c"].mean()
    temp_std = events_df["ambient_temp_c"].std()
    logger.info("  ambient_temp_c: mean=%.1f, std=%.1f  (expected ~23, ~3)", temp_mean, temp_std)

    pair_counts = events_df.groupby(["from_grade", "to_grade"]).size()
    min_pair = pair_counts.min()
    logger.info(
        "  Grade-pair coverage: %d pairs, min=%d occurrences (need ≥5 for k-NN k=5)",
        len(pair_counts), min_pair,
    )
    if min_pair < 5:
        logger.warning("  ⚠ Some grade pairs have < 5 examples — k-NN may degrade")

    wear_corr = events_df[["event_id", "valve_wear_index"]].corr().iloc[0, 1]
    logger.info(
        "  valve_wear_index vs event_id correlation: %.3f  (expected positive)",
        wear_corr,
    )

    outcome_cts = events_df["outcome_category"].value_counts().to_dict()
    logger.info("  outcome_category: %s", outcome_cts)


# ---------------------------------------------------------------------------
# Main generation routine
# ---------------------------------------------------------------------------

def generate(n_events: int, seed: int, output_dir: Path) -> None:
    """Generate all three canonical CSVs for the QCS Copilot dataset.

    Args:
        n_events: Number of grade-change events to simulate.
        seed: Random seed for full reproducibility.
        output_dir: Directory to write the three CSVs into.

    Raises:
        ValueError: On data quality check failure.
    """
    logger.info(
        "QCS Copilot synthetic data generation: n_events=%d, seed=%d, out=%s",
        n_events, seed, output_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Step 1: Write recipe_reference.csv (pure recipe metadata — no randomness)
    recipe_df = export_recipe_reference(output_dir)

    # Synthetic campaign start timestamp for historian realism
    campaign_start = datetime(2024, 1, 15, 6, 0, 0)

    # Carry-over state across events
    prior_ash: float = 10.0   # seed default per spec
    current_time = campaign_start

    all_ts_rows: list[pd.DataFrame] = []
    event_rows: list[dict] = []

    for event_idx in range(n_events):
        event_id = event_idx

        # Step 2: Sample grade pair and ramp
        from_grade, to_grade = sample_grade_pair(rng)
        operator_ramp_choice, ramp_duration_s = sample_ramp_duration(rng)

        # Step 3: Draw per-event disturbances
        dist = make_event_disturbances(rng, event_idx, n_events, prior_ash)

        # Step 4: Simulate event
        ts_df = simulate_event(
            event_id=event_id,
            from_grade=from_grade,
            to_grade=to_grade,
            operator_ramp_choice=operator_ramp_choice,
            ramp_duration_s=ramp_duration_s,
            disturbances=dist,
            rng=rng,
            event_start_dt=current_time,
        )

        # Step 5: Compute labels
        labels = compute_event_labels(ts_df)

        # Assemble event row
        event_rows.append(
            {
                "event_id": event_id,
                "event_start_timestamp": current_time,
                "from_grade": from_grade,
                "to_grade": to_grade,
                "operator_ramp_choice": operator_ramp_choice,
                "ramp_duration_seconds": ramp_duration_s,
                "valve_wear_index": dist.valve_wear_index,
                "ambient_temp_c": dist.ambient_temp_c,
                "prior_ash": dist.prior_ash,
                "max_abs_basis_weight_pct_dev": labels["max_abs_basis_weight_pct_dev"],
                "offspec_label": labels["offspec_label"],
                "time_to_offspec_s": labels["time_to_offspec_s"],
                "stabilization_time_seconds": labels["stabilization_time_seconds"],
                "stabilized_within_window": labels["stabilized_within_window"],
                "outcome_category": labels["outcome_category"],
                "event_duration_seconds": EVENT_DURATION_S,
            }
        )

        all_ts_rows.append(ts_df)

        # Update carryover state: next event's prior_ash = this event's destination ash
        dst_setpoints = get_setpoints_for_grade(to_grade)
        prior_ash = dst_setpoints["ash_setpoint"]

        # Advance simulation clock (event + 10-min buffer between events)
        current_time += timedelta(seconds=EVENT_DURATION_S + 600)

        if (event_idx + 1) % 50 == 0 or event_idx == n_events - 1:
            logger.info("  Generated event %d / %d", event_idx + 1, n_events)

    # Concatenate and write timeseries CSV
    ts_full = pd.concat(all_ts_rows, ignore_index=True)
    ts_path = output_dir / "historian_timeseries.csv"
    ts_full.to_csv(ts_path, index=False)
    logger.info("Wrote historian_timeseries.csv (%d rows) → %s", len(ts_full), ts_path)

    # Write events CSV
    events_df = pd.DataFrame(event_rows)
    ev_path = output_dir / "grade_change_events.csv"
    events_df.to_csv(ev_path, index=False)
    logger.info("Wrote grade_change_events.csv (%d rows) → %s", len(events_df), ev_path)

    # Step 6: Data quality checks
    validate_timeseries(ts_full, n_events)
    validate_events(events_df, ts_full)
    validate_consistency(events_df, ts_full)
    print_distribution_summary(events_df)

    logger.info("✅ Generation complete — all checks passed.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the QCS Copilot synthetic historian dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--events", type=int, default=250,
        help="Number of grade-change events to generate (250=dev, 500=final).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (frozen at 42 per PROJECT_CONSTITUTION.md).",
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="Directory to write the three CSV files into.",
    )
    args = parser.parse_args()

    if args.events not in (250, 500):
        logger.warning(
            "Non-standard event count %d requested. "
            "Constitution mandates 250 (dev) or 500 (final).",
            args.events,
        )

    try:
        generate(
            n_events=args.events,
            seed=args.seed,
            output_dir=Path(args.output),
        )
    except Exception as exc:
        logger.error("Data generation FAILED: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()