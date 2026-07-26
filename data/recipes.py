"""
recipes.py — Grade catalog and recipe-implied setpoint formulas.

Defines the 5 paper grades, their quality targets, control-variable setpoint
derivations, and metadata (ramp rates, typical transition times).  Also exports
the static ``recipe_reference.csv`` that downstream modules consume for domain
rules and correlation-discovery context.

This module has NO external dependencies within the project.  It is the first
module executed by ``generate_data.py``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grade catalog
# ---------------------------------------------------------------------------
# Keys are canonical grade names used throughout the project.
# All numeric targets are in the physical units documented in
# DATASET_SPECIFICATION.md §3.

GRADES: Final[dict[str, dict]] = {
    "G1_LightBond": {
        "basis_weight_setpoint": 45.0,   # gsm
        "moisture_setpoint": 6.0,         # %
        "ash_setpoint": 8.0,              # %
        "caliper_setpoint": 55.0,         # µm
        "recommended_max_ramp_rate": 3.0, # gsm/min
        "typical_transition_time_min": 15.0,
    },
    "G2_StandardBond": {
        "basis_weight_setpoint": 60.0,
        "moisture_setpoint": 6.5,
        "ash_setpoint": 10.0,
        "caliper_setpoint": 70.0,
        "recommended_max_ramp_rate": 4.0,
        "typical_transition_time_min": 20.0,
    },
    "G3_HeavyBond": {
        "basis_weight_setpoint": 80.0,
        "moisture_setpoint": 7.0,
        "ash_setpoint": 12.0,
        "caliper_setpoint": 95.0,
        "recommended_max_ramp_rate": 5.0,
        "typical_transition_time_min": 25.0,
    },
    "G4_Kraft": {
        "basis_weight_setpoint": 100.0,
        "moisture_setpoint": 7.5,
        "ash_setpoint": 6.0,
        "caliper_setpoint": 120.0,
        "recommended_max_ramp_rate": 7.0,
        "typical_transition_time_min": 30.0,
    },
    "G5_Newsprint": {
        "basis_weight_setpoint": 52.0,
        "moisture_setpoint": 8.0,
        "ash_setpoint": 4.0,
        "caliper_setpoint": 65.0,
        "recommended_max_ramp_rate": 8.0,
        "typical_transition_time_min": 35.0,
    },
}

# Spec-mandated global off-spec band (%) — stored per-row for future per-grade tuning.
OFFSPEC_BAND_PCT: Final[float] = 2.5


# ---------------------------------------------------------------------------
# Control-variable setpoint derivation
# ---------------------------------------------------------------------------
# These linear relationships map from recipe quality targets to the
# actuator setpoints visible in historian_timeseries.csv.  They are
# deliberately simple (linear regression-style coefficients) so the simulator
# stays physics-adjacent without requiring a full paper-machine process model.

def derive_stock_flow_setpoint(basis_weight_gsm: float) -> float:
    """Derive stock-flow setpoint from basis-weight target.

    Args:
        basis_weight_gsm: Recipe basis weight target in gsm.

    Returns:
        Corresponding stock-flow setpoint in L/min (412–550 range).
    """
    # Linear mapping: 45 gsm → 412 L/min; 100 gsm → 550 L/min.
    return 412.0 + (basis_weight_gsm - 45.0) * (550.0 - 412.0) / (100.0 - 45.0)


def derive_filler_flow_setpoint(ash_pct: float) -> float:
    """Derive filler-flow setpoint from ash target.

    Args:
        ash_pct: Recipe ash target in %.

    Returns:
        Corresponding filler-flow setpoint in L/min (32–56 range).
    """
    # Linear mapping: 4 % ash → 32 L/min; 12 % ash → 56 L/min.
    return 32.0 + (ash_pct - 4.0) * (56.0 - 32.0) / (12.0 - 4.0)


def derive_steam_pressure_setpoint(moisture_pct: float) -> float:
    """Derive steam-pressure setpoint from moisture target.

    Args:
        moisture_pct: Recipe moisture target in %.

    Returns:
        Corresponding steam-pressure setpoint in kPa (310–330 range).
    """
    # Linear mapping: 6.0 % moisture → 330 kPa; 8.0 % → 310 kPa.
    # (Drier paper requires more steam — inverse relationship.)
    return 330.0 + (moisture_pct - 6.0) * (310.0 - 330.0) / (8.0 - 6.0)


def derive_machine_speed_setpoint(basis_weight_gsm: float) -> float:
    """Derive machine-speed setpoint from basis-weight target.

    Args:
        basis_weight_gsm: Recipe basis weight target in gsm.

    Returns:
        Corresponding machine-speed setpoint in m/min (700–810 range).
    """
    # Heavier grades run slower — inverse linear mapping.
    return 810.0 - (basis_weight_gsm - 45.0) * (810.0 - 700.0) / (100.0 - 45.0)


def get_setpoints_for_grade(grade_name: str) -> dict[str, float]:
    """Return all recipe-implied setpoints (quality + control) for a grade.

    Args:
        grade_name: One of the 5 canonical grade names in GRADES.

    Returns:
        Dict with basis_weight, moisture, ash, caliper setpoints plus the
        four derived control-variable setpoints.

    Raises:
        KeyError: If grade_name is not in the grade catalog.
    """
    if grade_name not in GRADES:
        raise KeyError(
            f"Unknown grade '{grade_name}'. Valid grades: {list(GRADES.keys())}"
        )
    g = GRADES[grade_name]
    return {
        "basis_weight_setpoint": g["basis_weight_setpoint"],
        "moisture_setpoint": g["moisture_setpoint"],
        "ash_setpoint": g["ash_setpoint"],
        "caliper_setpoint": g["caliper_setpoint"],
        "stock_flow_setpoint": derive_stock_flow_setpoint(g["basis_weight_setpoint"]),
        "filler_flow_setpoint": derive_filler_flow_setpoint(g["ash_setpoint"]),
        "steam_pressure_setpoint": derive_steam_pressure_setpoint(g["moisture_setpoint"]),
        "machine_speed_setpoint": derive_machine_speed_setpoint(g["basis_weight_setpoint"]),
    }


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_recipe_reference(output_dir: str | Path) -> pd.DataFrame:
    """Build and write ``recipe_reference.csv`` to *output_dir*.

    Each row contains all static recipe metadata for one grade, including
    both quality targets and derived control-variable setpoints.

    Args:
        output_dir: Directory path for the output CSV.

    Returns:
        The DataFrame written to disk (also useful for in-process consumers).

    Raises:
        OSError: If the output directory cannot be created or the file
            cannot be written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for grade_name, meta in GRADES.items():
        setpoints = get_setpoints_for_grade(grade_name)
        rows.append(
            {
                "grade_name": grade_name,
                "basis_weight_setpoint": meta["basis_weight_setpoint"],
                "moisture_setpoint": meta["moisture_setpoint"],
                "ash_setpoint": meta["ash_setpoint"],
                "caliper_setpoint": meta["caliper_setpoint"],
                "stock_flow_setpoint": setpoints["stock_flow_setpoint"],
                "filler_flow_setpoint": setpoints["filler_flow_setpoint"],
                "steam_pressure_setpoint": setpoints["steam_pressure_setpoint"],
                "machine_speed_setpoint": setpoints["machine_speed_setpoint"],
                "offspec_band_pct": OFFSPEC_BAND_PCT,
                "recommended_max_ramp_rate": meta["recommended_max_ramp_rate"],
                "typical_transition_time_min": meta["typical_transition_time_min"],
            }
        )

    df = pd.DataFrame(rows)
    out_path = output_dir / "recipe_reference.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote recipe_reference.csv (%d rows) → %s", len(df), out_path)
    return df
