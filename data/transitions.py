"""
transitions.py — FOPDT ramp dynamics and the per-event simulation loop.

Runs the 2400-second, 5-second-sampled simulation for a single grade-change
event.  Accepts pre-computed disturbance parameters from ``disturbances.py``
and grade setpoints from ``recipes.py``; returns a DataFrame of 480 rows that
becomes historian_timeseries.csv rows for this event.

FOPDT discrete update:
    x(t + Δt) = x(t) + (Δt / (τ + Δt)) * (SP(t) - x(t)) + ε(t)

where SP(t) is the linearly ramping setpoint between source and destination
values, and ε(t) is Gaussian sensor noise.

No randomness is generated inside this module — all stochastic values are
received from the caller (disturbances, rng).  This keeps simulation logic
separate from probability-distribution concerns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from data.disturbances import (
    EventDisturbances, overshoot_bump_at, sample_noise,
    TAU_STEAM_BASE, TAU_SPEED_BASE, TAU_CALIPER_BASE, TAU_MOISTURE_BASE,
    K_TEMP_BW_PCT, K_WEAR_BW_PCT,
)
from data.recipes import GRADES, get_setpoints_for_grade

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event-timing constants — must match DATASET_SPECIFICATION.md §3
# ---------------------------------------------------------------------------
EVENT_DURATION_S: int = 2400   # fixed 40-minute observation window
SAMPLE_INTERVAL_S: int = 5     # sampling frequency
N_SAMPLES: int = EVENT_DURATION_S // SAMPLE_INTERVAL_S  # 480 per event

# Ramp fractions of the full window: aggressive vs. calm.
RAMP_FRACTION_AGGRESSIVE: float = 0.18   # 18% of 2400s ≈ 432s
RAMP_FRACTION_CALM: float = 0.50         # 50% of 2400s = 1200s
RAMP_JITTER: float = 0.10               # ±10% jitter on ramp duration
OPERATOR_RAMP_CHOICE_PROB: float = 0.35   # Bernoulli p for operator_ramp_choice


# ---------------------------------------------------------------------------
# Setpoint ramp helper
# ---------------------------------------------------------------------------

def linear_ramp_setpoint(
    t_s: int,
    sp_src: float,
    sp_dst: float,
    ramp_end_s: int,
) -> float:
    """Return the linearly interpolated setpoint value at elapsed time t_s.

    Args:
        t_s: Elapsed seconds since event start.
        sp_src: Setpoint at t=0 (source grade value).
        sp_dst: Setpoint at ramp completion (destination grade value).
        ramp_end_s: The second at which the ramp completes (after which SP
            holds at sp_dst).

    Returns:
        Setpoint value at t_s.
    """
    if ramp_end_s <= 0 or t_s >= ramp_end_s:
        return sp_dst
    fraction = t_s / ramp_end_s
    return sp_src + fraction * (sp_dst - sp_src)


# ---------------------------------------------------------------------------
# Ramp duration sampling
# ---------------------------------------------------------------------------

def sample_ramp_duration(rng: np.random.Generator) -> tuple[bool, int]:
    """Sample operator_ramp_choice flag and ramp_duration_seconds for one event.

    The Bernoulli draw for operator_ramp_choice uses p=0.35.  Ramp duration is
    expressed as a fraction of EVENT_DURATION_S, with ±10% uniform jitter so
    the flag is not a perfectly clean separator.

    Args:
        rng: Seeded numpy random generator.

    Returns:
        Tuple of (operator_ramp_choice: bool, ramp_duration_seconds: int).
    """
    aggressive = rng.random() < OPERATOR_RAMP_CHOICE_PROB
    base_fraction = RAMP_FRACTION_AGGRESSIVE if aggressive else RAMP_FRACTION_CALM
    jitter = rng.uniform(-RAMP_JITTER, RAMP_JITTER)
    raw_duration = EVENT_DURATION_S * base_fraction * (1.0 + jitter)
    # Snap to nearest SAMPLE_INTERVAL_S multiple; clamp within the window.
    duration = int(round(raw_duration / SAMPLE_INTERVAL_S) * SAMPLE_INTERVAL_S)
    duration = max(SAMPLE_INTERVAL_S, min(duration, EVENT_DURATION_S))
    return aggressive, duration


# ---------------------------------------------------------------------------
# Grade-pair sampling
# ---------------------------------------------------------------------------

GRADE_NAMES: list[str] = list(GRADES.keys())


def sample_grade_pair(rng: np.random.Generator) -> tuple[str, str]:
    """Sample a from_grade/to_grade pair (always different) uniformly.

    Args:
        rng: Seeded numpy random generator.

    Returns:
        Tuple of (from_grade, to_grade) — always from ≠ to.
    """
    from_idx, to_idx = rng.choice(len(GRADE_NAMES), size=2, replace=False)
    return GRADE_NAMES[int(from_idx)], GRADE_NAMES[int(to_idx)]


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def simulate_event(
    event_id: int,
    from_grade: str,
    to_grade: str,
    operator_ramp_choice: bool,
    ramp_duration_s: int,
    disturbances: EventDisturbances,
    rng: np.random.Generator,
    event_start_dt: datetime,
) -> pd.DataFrame:
    """Simulate one grade-change event and return its time-series DataFrame.

    Runs 480 timesteps (2400 s at 5-second intervals) for the given grade pair
    using first-order lag (FOPDT) dynamics on all process variables, applying
    the disturbances from ``disturbances`` and Gaussian noise.

    Args:
        event_id: Unique integer identifier for this event.
        from_grade: Source grade name.
        to_grade: Destination grade name.
        operator_ramp_choice: Whether this event used an aggressive ramp operator choice.
        ramp_duration_s: Actual ramp window in seconds (already jittered).
        disturbances: Pre-computed per-event hidden variables and time constants.
        rng: Seeded numpy random generator for sensor noise.
        event_start_dt: Absolute datetime stamp for t=0 of this event.

    Returns:
        DataFrame with 480 rows conforming to historian_timeseries.csv schema.
    """
    src = get_setpoints_for_grade(from_grade)
    dst = get_setpoints_for_grade(to_grade)

    dt = SAMPLE_INTERVAL_S  # 5 seconds

    # Effective time constants (pre-computed in disturbances; clean ones from constants)
    tau_bw = disturbances.tau_bw
    tau_stock = disturbances.tau_stock
    tau_filler = disturbances.tau_filler
    tau_steam = TAU_STEAM_BASE     # clean, no disturbance
    tau_speed = TAU_SPEED_BASE     # clean, no disturbance
    tau_caliper = TAU_CALIPER_BASE # clean, no disturbance
    tau_moisture = TAU_MOISTURE_BASE

    # Initial conditions — process starts at source-grade equilibrium
    bw = src["basis_weight_setpoint"]
    sf = src["stock_flow_setpoint"]
    ff = src["filler_flow_setpoint"]
    sp = src["steam_pressure_setpoint"]
    ms = src["machine_speed_setpoint"]
    cal = src["caliper_setpoint"]
    mois = src["moisture_setpoint"]
    # Ash initial condition: 15% blend toward prior_ash (relationship D)
    ash = (0.85 * src["ash_setpoint"]) + (0.15 * disturbances.prior_ash)

    rows: list[dict] = []

    for step in range(N_SAMPLES):
        t_s = step * SAMPLE_INTERVAL_S
        ts = event_start_dt + timedelta(seconds=t_s)

        # --- Setpoint ramps (linear, known in advance) ---
        bw_sp = linear_ramp_setpoint(t_s, src["basis_weight_setpoint"],
                                     dst["basis_weight_setpoint"], ramp_duration_s)
        sf_sp = linear_ramp_setpoint(t_s, src["stock_flow_setpoint"],
                                     dst["stock_flow_setpoint"], ramp_duration_s)
        ff_sp = linear_ramp_setpoint(t_s, src["filler_flow_setpoint"],
                                     dst["filler_flow_setpoint"], ramp_duration_s)
        steam_sp = linear_ramp_setpoint(t_s, src["steam_pressure_setpoint"],
                                        dst["steam_pressure_setpoint"], ramp_duration_s)
        speed_sp = linear_ramp_setpoint(t_s, src["machine_speed_setpoint"],
                                        dst["machine_speed_setpoint"], ramp_duration_s)
        cal_sp = linear_ramp_setpoint(t_s, src["caliper_setpoint"],
                                      dst["caliper_setpoint"], ramp_duration_s)
        ash_sp = linear_ramp_setpoint(t_s, src["ash_setpoint"],
                                      dst["ash_setpoint"], ramp_duration_s)
        mois_sp_target = linear_ramp_setpoint(t_s, src["moisture_setpoint"],
                                              dst["moisture_setpoint"], ramp_duration_s)

        # --- Filler-flow overshoot bump (relationship B, transient) ---
        ff_bump = 0.0
        if disturbances.overshoot_active:
            ff_bump = overshoot_bump_at(t_s, 0, ramp_duration_s,
                                        disturbances.overshoot_amplitude * 5.0)

        # --- Direct basis-weight disturbance (mechanism 2, drives offspec_label) ---
        # A triangular window (0→1→0 over the ramp) models a process upset that
        # peaks at mid-ramp then resolves as the new grade steady-state is reached.
        # The process FOPDT tracks (bw_sp + disturbance), so the nominal-setpoint
        # deviation equals the disturbance at steady state.
        if ramp_duration_s > 0 and t_s <= ramp_duration_s:
            ramp_progress = t_s / ramp_duration_s          # 0 → 1
            # Triangular: 0 at start, 1 at mid-ramp, 0 at end
            tri_window = min(ramp_progress, 1.0 - ramp_progress) * 2.0
        else:
            tri_window = 0.0  # no upset outside the ramp window
            
        bw_direct_pct = (
            K_TEMP_BW_PCT * max(0.0, disturbances.ambient_temp_c - 22.0)
            + K_WEAR_BW_PCT * disturbances.valve_wear_index
        ) * tri_window
        bw_direct = bw_direct_pct * (bw_sp / 100.0)

        # --- FOPDT updates (Euler forward) ---
        alpha_bw = dt / (tau_bw + dt)
        alpha_sf = dt / (tau_stock + dt)
        alpha_ff = dt / (tau_filler + dt)
        alpha_steam = dt / (tau_steam + dt)
        alpha_speed = dt / (tau_speed + dt)
        alpha_cal = dt / (tau_caliper + dt)
        alpha_mois = dt / (tau_moisture + dt)

        # bw tracks (bw_sp + direct disturbance); pct_dev measured vs nominal bw_sp
        bw_effective_sp = bw_sp + bw_direct
        bw_new = bw + alpha_bw * (bw_effective_sp - bw)
        # Cross-coupling: filler-flow overshoot nudges basis weight (raised coeff -> more discoverable)
        bw_new += 0.25 * ff_bump if disturbances.overshoot_active else 0.0
        sf_new = sf + alpha_sf * (sf_sp - sf)
        ff_new = ff + alpha_ff * (ff_sp + ff_bump - ff)
        sp_new = sp + alpha_steam * (steam_sp - sp)    # clean
        ms_new = ms + alpha_speed * (speed_sp - ms)    # clean
        cal_new = cal + alpha_cal * (cal_sp - cal)     # clean
        # Ash: driven toward ash_sp through filler_flow lag
        ash_new = ash + alpha_ff * (ash_sp - ash)
        mois_new = mois + alpha_mois * (mois_sp_target - mois)

        # --- Add sensor noise ---
        bw_obs = bw_new + sample_noise(rng, "basis_weight")
        sf_obs = sf_new + sample_noise(rng, "stock_flow")
        ff_obs = ff_new + sample_noise(rng, "filler_flow")
        sp_obs = sp_new + sample_noise(rng, "steam_pressure")
        ms_obs = ms_new + sample_noise(rng, "machine_speed")
        cal_obs = cal_new + sample_noise(rng, "caliper")
        ash_obs = ash_new + sample_noise(rng, "ash")
        mois_obs = mois_new + sample_noise(rng, "moisture")

        # --- Percentage deviation from ramping setpoint ---
        bw_pct_dev = (
            (bw_obs - bw_sp) / bw_sp * 100.0 if bw_sp != 0 else 0.0
        )

        rows.append(
            {
                "event_id": event_id,
                "timestamp": ts,
                "t_seconds": t_s,
                "from_grade": from_grade,
                "to_grade": to_grade,
                # Control variables (actual observed)
                "stock_flow": round(sf_obs, 4),
                "stock_flow_setpoint": round(sf_sp, 4),
                "filler_flow": round(ff_obs, 4),
                "filler_flow_setpoint": round(ff_sp, 4),
                "steam_pressure": round(sp_obs, 4),
                "steam_pressure_setpoint": round(steam_sp, 4),
                "machine_speed": round(ms_obs, 4),
                "machine_speed_setpoint": round(speed_sp, 4),
                # Quality variables
                "basis_weight": round(bw_obs, 4),
                "basis_weight_setpoint": round(bw_sp, 4),
                "basis_weight_pct_dev": round(bw_pct_dev, 6),
                "moisture": round(mois_obs, 4),
                "ash": round(ash_obs, 4),
                "caliper": round(cal_obs, 4),
                # Hidden / environmental (repeated per row as spec requires)
                "ambient_temp_c": disturbances.ambient_temp_c,
                "valve_wear_index": disturbances.valve_wear_index,
                # Ground-truth flag — never a model feature
                "operator_ramp_choice": operator_ramp_choice,
            }
        )

        # Advance state
        bw = bw_new
        sf = sf_new
        ff = ff_new
        sp = sp_new
        ms = ms_new
        cal = cal_new
        ash = ash_new
        mois = mois_new

    df = pd.DataFrame(rows)
    logger.debug(
        "Simulated event %d (%s→%s), %d rows, operator_ramp_choice=%s",
        event_id, from_grade, to_grade, len(df), operator_ramp_choice,
    )
    return df
