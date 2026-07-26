"""
disturbances.py — Hidden-variable generators and physics disturbances.

Responsible for:
  1. Drawing per-event static hidden variables (ambient_temp_c, valve_wear_index,
     prior_ash) from their specified distributions.
  2. Computing disturbed FOPDT time-constants given those variables.
  3. Generating the transient mid-ramp valve-overshoot bump when activated.
  4. Returning per-sample Gaussian sensor noise vectors.

All randomness in this module is driven by a caller-supplied ``numpy.random.Generator``
so that reproducibility is fully controlled by the seed in ``generate_data.py``.

DATASET_SPECIFICATION.md §4 and §5 define every numeric parameter used here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base FOPDT time constants (undisturbed) — in seconds
# ---------------------------------------------------------------------------
# These are the "clean" time constants before any hidden-variable modification.
# They are simulator internals — MUST NOT appear as model features (§8).
#
# Design decision (§6 empirical update):
#   Two-mechanism architecture:
#   (1) TAU modification (tau_bw += 1×temp + 10×wear) — adjusted empirically — is the
#       primary driver of stabilization_time_seconds (relationship A via tau).
#       Values reduced from spec (2/18) because they mathematically forced off-spec
#       on all aggressive large-delta ramps (e.g., G4->G1 threshold is ~14.2s).
#   (2) Direct basis-weight perturbation (K_TEMP_BW_PCT, K_WEAR_BW_PCT
#       below) is the primary driver of offspec_label (relationship A/B via
#       process upset during ramp), calibrated to produce the 35–45% target.
#   This separation lets both target variables respond to the same hidden
#   variables through physically distinct channels.

TAU_BW_BASE: float = 4.0        # basis_weight  first-order lag (s)
TAU_STOCK_BASE: float = 5.0     # stock_flow first-order lag (s)
TAU_FILLER_BASE: float = 4.0    # filler_flow first-order lag (s)
TAU_STEAM_BASE: float = 3.0     # steam_pressure first-order lag (s)  [clean]
TAU_SPEED_BASE: float = 2.0     # machine_speed first-order lag (s)   [clean]
TAU_CALIPER_BASE: float = 4.0   # caliper first-order lag (s)         [clean]
TAU_MOISTURE_BASE: float = 4.0  # moisture first-order lag (s)

# ---------------------------------------------------------------------------
# Direct basis-weight disturbance coefficients (mechanism 2, see above)
# ---------------------------------------------------------------------------
# During the ramp, hidden variables produce a process upset that shifts the
# actual basis weight value away from the ramping setpoint.  This is physically
# distinct from the tau-lag mechanism and represents unmeasured feedforward
# errors (fibre swelling from ambient heat, filler over-dosing from worn valves).
# Coefficients calibrated so P(|pct_dev| > 2.5 at any ramp timestep) ≈ 35-45%.
# Applied as a percentage of the current setpoint to ensure lighter grades
# don't disproportionately go off-spec.
K_TEMP_BW_PCT: float = 0.30  # % deviation per excess degC above 22  (relationship A)
K_WEAR_BW_PCT: float = 2.20  # % deviation per unit valve_wear_index (relationship B)

# Campaign drift ceiling added to valve_wear_index across all N events.
VALVE_WEAR_DRIFT_MAX: float = 0.3

# Overshoot conditional threshold.
VALVE_OVERSHOOT_THRESHOLD: float = 0.6  # wear index must exceed this
VALVE_OVERSHOOT_PROB: float = 0.5       # Bernoulli p given threshold exceeded


# ---------------------------------------------------------------------------
# Per-event hidden-variable draw
# ---------------------------------------------------------------------------

@dataclass
class EventDisturbances:
    """Holds all per-event static disturbance values.

    Attributes:
        ambient_temp_c: Drawn once per event; repeated on every row.
        valve_wear_index: Drawn once per event (with campaign drift baked in).
        prior_ash: Deterministic carryover from the previous event.
        overshoot_active: True if a mid-ramp overshoot event is triggered.
        overshoot_amplitude: Amplitude of the sinusoidal bump (gsm equivalent).
        tau_bw: Disturbed basis-weight time constant (s).
        tau_stock: Disturbed stock-flow time constant (s).
        tau_filler: Disturbed filler-flow time constant (s).
    """

    ambient_temp_c: float
    valve_wear_index: float
    prior_ash: float
    overshoot_active: bool
    overshoot_amplitude: float
    tau_bw: float
    tau_stock: float
    tau_filler: float


def draw_ambient_temp(rng: np.random.Generator) -> float:
    """Draw ambient temperature from Normal(23, 3) clipped to [15, 35] °C.

    Args:
        rng: Seeded numpy random generator.

    Returns:
        Ambient temperature in °C.
    """
    temp = rng.normal(loc=23.0, scale=3.0)
    return float(np.clip(temp, 15.0, 35.0))


def draw_valve_wear(
    rng: np.random.Generator,
    event_index: int,
    total_events: int,
) -> float:
    """Draw valve wear index from Beta(2,5) plus a linear campaign drift.

    The drift adds up to VALVE_WEAR_DRIFT_MAX (0.3) by the final event,
    encoding relationship C (campaign degradation) from the spec.

    Args:
        rng: Seeded numpy random generator.
        event_index: 0-based position of this event in the generation run.
        total_events: Total number of events being generated.

    Returns:
        Valve wear index in [0, 1].
    """
    base_wear = rng.beta(a=2, b=5)
    drift = VALVE_WEAR_DRIFT_MAX * (event_index / max(total_events - 1, 1))
    return float(np.clip(base_wear + drift, 0.0, 1.0))


def compute_disturbed_time_constants(
    ambient_temp_c: float,
    valve_wear_index: float,
) -> dict[str, float]:
    """Compute FOPDT time constants after applying hidden-variable disturbances.

    Relationship A: ambient_temp_c > 22 increases tau_bw (strong) and
        tau_stock (weak, secondary channel E).
    Relationship B: valve_wear_index increases tau_filler and tau_bw.

    Args:
        ambient_temp_c: Current event's ambient temperature in °C.
        valve_wear_index: Current event's wear index in [0, 1].

    Returns:
        Dict of disturbed time constants keyed by variable name.
    """
    excess_temp = max(0.0, ambient_temp_c - 22.0)

    # Relationship A — tau channel into basis_weight (drives stabilization_time)
    # Scaled down empirically to avoid forcing off-spec via lag alone.
    tau_bw = TAU_BW_BASE + 1.0 * excess_temp + 10.0 * valve_wear_index
    # Relationship E — weak secondary channel into stock_flow
    tau_stock = TAU_STOCK_BASE + 0.5 * excess_temp
    # Relationship B — valve wear elevates filler_flow lag (drives ash carryover)
    tau_filler = TAU_FILLER_BASE + 40.0 * valve_wear_index

    return {
        "tau_bw": float(tau_bw),
        "tau_stock": float(tau_stock),
        "tau_filler": float(tau_filler),
        "tau_steam": TAU_STEAM_BASE,    # clean — no disturbance
        "tau_speed": TAU_SPEED_BASE,    # clean — no disturbance
        "tau_caliper": TAU_CALIPER_BASE,  # clean — no disturbance
        "tau_moisture": TAU_MOISTURE_BASE,
    }


def draw_overshoot(
    rng: np.random.Generator,
    valve_wear_index: float,
) -> tuple[bool, float]:
    """Determine if a mid-ramp overshoot event is triggered and its amplitude.

    Overshoot is a Bernoulli(0.5) draw conditional on valve_wear_index > 0.6,
    per spec §5 (relationship B, transient).

    Args:
        rng: Seeded numpy random generator.
        valve_wear_index: Current event's wear index.

    Returns:
        Tuple of (overshoot_active, amplitude).
        amplitude is 0.0 when overshoot_active is False.
    """
    if valve_wear_index <= VALVE_OVERSHOOT_THRESHOLD:
        return False, 0.0
    active = rng.random() < VALVE_OVERSHOOT_PROB
    if not active:
        return False, 0.0
    amplitude = float(np.clip(rng.normal(loc=0.25, scale=0.05), 0.05, 0.5))
    return True, amplitude


def make_event_disturbances(
    rng: np.random.Generator,
    event_index: int,
    total_events: int,
    prior_ash: float,
) -> EventDisturbances:
    """Assemble a complete EventDisturbances object for one grade-change event.

    Args:
        rng: Seeded numpy random generator.
        event_index: 0-based position of this event in the generation run.
        total_events: Total number of events being generated.
        prior_ash: Destination ash setpoint of the immediately preceding event.

    Returns:
        Fully populated EventDisturbances for this event.
    """
    ambient_temp = draw_ambient_temp(rng)
    wear = draw_valve_wear(rng, event_index, total_events)
    overshoot_active, overshoot_amp = draw_overshoot(rng, wear)
    taus = compute_disturbed_time_constants(ambient_temp, wear)

    logger.debug(
        "Event %d: temp=%.1f°C, wear=%.3f, overshoot=%s, tau_bw=%.1f",
        event_index, ambient_temp, wear, overshoot_active, taus["tau_bw"],
    )

    return EventDisturbances(
        ambient_temp_c=ambient_temp,
        valve_wear_index=wear,
        prior_ash=prior_ash,
        overshoot_active=overshoot_active,
        overshoot_amplitude=overshoot_amp,
        tau_bw=taus["tau_bw"],
        tau_stock=taus["tau_stock"],
        tau_filler=taus["tau_filler"],
    )


# ---------------------------------------------------------------------------
# Transient overshoot waveform
# ---------------------------------------------------------------------------

def overshoot_bump_at(
    t_seconds: int,
    ramp_start_s: int,
    ramp_end_s: int,
    amplitude: float,
) -> float:
    """Return the sinusoidal filler-flow overshoot bump magnitude at time t.

    The bump is active only during the middle third of the ramp window and
    has a sinusoidal shape that peaks at the center.

    Args:
        t_seconds: Current elapsed time within the event (s).
        ramp_start_s: Ramp start time in the event (always 0 for our sim).
        ramp_end_s: Ramp end time in the event (s).
        amplitude: Peak amplitude of the bump (fractional of nominal filler flow).

    Returns:
        Additive bump value to apply to filler_flow at this timestep.
    """
    ramp_duration = ramp_end_s - ramp_start_s
    window_start = ramp_start_s + ramp_duration // 3
    window_end = ramp_start_s + 2 * ramp_duration // 3

    if t_seconds < window_start or t_seconds > window_end:
        return 0.0

    # Sine wave peaks at the center of the active window
    progress = (t_seconds - window_start) / max(window_end - window_start, 1)
    return float(amplitude * np.sin(progress * np.pi))


# ---------------------------------------------------------------------------
# Sensor noise
# ---------------------------------------------------------------------------

# Per-tag Gaussian noise standard deviations (from DATASET_SPECIFICATION.md §4)
NOISE_SIGMA: dict[str, float] = {
    "stock_flow": 1.2,
    "filler_flow": 0.4,
    "steam_pressure": 2.0,
    "machine_speed": 1.0,
    "basis_weight": 0.25,
    "moisture": 0.08,
    "ash": 0.15,
    "caliper": 0.5,
}


def sample_noise(rng: np.random.Generator, tag: str) -> float:
    """Draw one Gaussian noise sample for a given process tag.

    Args:
        rng: Seeded numpy random generator.
        tag: Process variable name — must be a key in NOISE_SIGMA.

    Returns:
        Noise sample in the same units as the process variable.

    Raises:
        KeyError: If tag is not in the NOISE_SIGMA table.
    """
    if tag not in NOISE_SIGMA:
        raise KeyError(f"No noise sigma defined for tag '{tag}'.")
    return float(rng.normal(loc=0.0, scale=NOISE_SIGMA[tag]))
